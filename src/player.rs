// SPDX-License-Identifier: MIT OR Apache-2.0

//! `HOWL` music sequencer and 24-voice PlayStation 1 SPU player driving
//! [`libspu2::Device::new_ps1()`].

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::fmt;

use libspu2::registers::{ps1, CONTROL_ENABLE, CONTROL_REVERB, CONTROL_UNMUTE};
use libspu2::{
    Device, EnvelopePhase, StereoI16, VoiceState, PS1_RAM_SIZE, PS1_SAMPLE_RATE, VOICES_PER_CORE,
};

use crate::bank::{HowlArchiveView, SfxEntry};
use crate::reverb::{ReverbParams, ReverbPreset};
use crate::sequence::{HowlCommand, SongSet};
pub use crate::tables::compute_tempo_fp16;
use crate::tables::{
    compute_melodic_pitch, compute_percussion_pitch, compute_stereo_volume, CENTER_PARAM,
};
use crate::{Error, Result};

/// Default PS1 SPU RAM base address (`0x1010` = `0x202 << 3`), immediately after
/// the hardware capture area and 16-byte silent dummy block.
pub const DEFAULT_SPU_RAM_BASE: u32 = 0x1010;

/// Hardware upper bound on SPU RAM uploads before reverb work area (`0x7E000`).
pub const MAX_SPU_RAM_END: u32 = 0x7E000;

/// Default PS1 SPU `VP_ADSR1` register used for percussion and SFX voices.
pub const DEFAULT_PERCUSSION_ADSR1: u16 = 0x80FF;

/// Default PS1 SPU `VP_ADSR2` register used for percussion and SFX voices.
pub const DEFAULT_PERCUSSION_ADSR2: u16 = 0x1FC2;

/// Native `HOWL` sequencer tick rate in Hz (NTSC 60 Hz frame tick).
pub const HOWL_TICK_HZ: u32 = 60;

/// Default global music volume (`0xAF` = 175).
pub const DEFAULT_MUSIC_VOLUME: u8 = 0xAF;

/// Default global SFX volume (`0xD7` = 215).
pub const DEFAULT_SFX_VOLUME: u8 = 0xD7;

/// Default sequence master volume (`0xBE` = 190).
pub const DEFAULT_SEQUENCE_VOLUME: u8 = 0xBE;

// Per-voice register staging dirty bits (matching the hardware driver's commit pass).
const DIRTY_KOFF: u32 = 0x01;
const DIRTY_KON: u32 = 0x02;
const DIRTY_SSA: u32 = 0x04;
const DIRTY_ADSR: u32 = 0x08;
const DIRTY_PITCH: u32 = 0x10;
const DIRTY_REVERB: u32 = 0x20;
const DIRTY_VOLUME: u32 = 0x40;
const DIRTY_ALL_NOTE_ON: u32 = DIRTY_KON | DIRTY_SSA | DIRTY_ADSR | DIRTY_PITCH | DIRTY_REVERB | DIRTY_VOLUME;

/// Configuration for [`HowlPlayer`] rendering and hardware-faithful fade-out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderConfig {
    /// Full sequence loops to play before fade-out (`0` = play once, then tail).
    pub target_loops: u32,
    /// Per-sequence voice-volume fade-out duration in seconds (default `8.0`).
    pub fade_seconds: f64,
    /// Post-fade release / reverb decay tail in seconds (default `2.0`).
    pub tail_seconds: f64,
    /// Hard safety ceiling on total rendered duration in seconds (default `300.0`).
    pub max_seconds: f64,
    /// Output PCM sample rate in Hz (default `44100`, PS1 SPU native rate).
    pub sample_rate: u32,
    /// Stereo panning enabled (`true`) or mono center (`false`).
    pub stereo: bool,
    /// Global music volume (`0..=255`, default `0xAF`).
    pub music_volume: u8,
    /// Global SFX volume (`0..=255`, default `0xD7`).
    pub sfx_volume: u8,
    /// Initial sequence volume (`0..=255`, default `0xBE`).
    pub sequence_volume: u8,
    /// Whether SPU reverb master write is enabled (default `true`).
    pub reverb_enabled: bool,
    /// Fixed PS1 SPU master volume (`MVOLL`/`MVOLR`, default `0x3FFF`). Never modified by fade-out.
    pub master_volume: u16,
    /// Hardware reverb preset programmed at startup (default [`ReverbPreset::Off`]).
    pub reverb_preset: ReverbPreset,
    /// SPU reverb return depth (`EVOLL`/`EVOLR`, default `0x2000`).
    pub reverb_depth: i16,
    /// Sequencer tick rate in Hz (default `60`).
    pub tick_hz: u32,
}

impl RenderConfig {
    /// Returns the standard BGM playback configuration (2 loops, 8 s linear fade, 2 s tail, 44.1 kHz).
    #[inline]
    #[must_use]
    pub const fn bgm_default() -> Self {
        Self {
            target_loops: 2,
            fade_seconds: 8.0,
            tail_seconds: 2.0,
            max_seconds: 300.0,
            sample_rate: PS1_SAMPLE_RATE,
            stereo: true,
            music_volume: DEFAULT_MUSIC_VOLUME,
            sfx_volume: DEFAULT_SFX_VOLUME,
            sequence_volume: DEFAULT_SEQUENCE_VOLUME,
            reverb_enabled: true,
            master_volume: 0x3FFF,
            reverb_preset: ReverbPreset::Off,
            reverb_depth: 0x2000,
            tick_hz: HOWL_TICK_HZ,
        }
    }
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self::bgm_default()
    }
}

/// Summary and interleaved stereo PCM returned by [`HowlPlayer::render_all`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderResult {
    /// Rendered interleaved stereo 16-bit PCM (`[L0, R0, L1, R1, ...]`).
    pub samples: Vec<i16>,
    /// Total number of stereo frames rendered (`samples.len() / 2`).
    pub frames_rendered: u64,
    /// Number of completed sequence loops observed during playback.
    pub loops_completed: u32,
    /// `true` if playback completed a fade-out cycle.
    pub faded_out: bool,
    /// `true` if all tracks halted naturally before hitting `max_seconds`.
    pub reached_end: bool,
}

/// Read-only runtime snapshot of a `HOWL` sequence track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrackSnapshot {
    /// Track index within the active sequence.
    pub index: usize,
    /// Track ID byte from the track header.
    pub id: u8,
    /// `true` if this track addresses the percussion table.
    pub percussion: bool,
    /// `true` if this track is still active.
    pub active: bool,
    /// `true` if this track is currently muted via mute/solo masks.
    pub muted: bool,
    /// Current melodic instrument index (`0x09`).
    pub instrument: u8,
    /// Current track volume (`0x06`, `0..=255`).
    pub volume: u8,
    /// Current track stereo pan (`0x07`, `0..=255`, `0x80` = center).
    pub pan: u8,
    /// Current track pitch bend (`0x0A`, `0..=255`, `0x80` = center).
    pub pitch_bend: u8,
    /// Current reverb send flag (`0x08`).
    pub reverb: bool,
    /// Most recently assigned hardware voice index, if any.
    pub voice: Option<usize>,
    /// Live hardware voice state from `libspu2`, if a hardware voice is assigned.
    pub voice_state: Option<VoiceState>,
}

/// Summary of ADPCM samples uploaded to PS1 SPU RAM for a [`HowlPlayer`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpuUploadSummary {
    /// Base byte address in PS1 SPU RAM where samples were uploaded.
    pub base_addr: u32,
    /// End byte address (exclusive) of uploaded ADPCM data.
    pub end_addr: u32,
    /// Number of distinct SPU sample IDs uploaded.
    pub sample_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct TrackRuntime {
    active: bool,
    percussion: bool,
    loop_enabled: bool,
    pass_had_time: bool,
    id: u8,
    instrument: u8,
    reverb: bool,
    volume: u8,
    pitch_bend: u8,
    pan: u8,
    event_index: usize,
    wait_ticks: u32,
    elapsed_ticks: u32,
    loops_completed: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VoiceKind {
    Music,
    Sfx,
}

#[derive(Debug, Clone, Copy)]
struct ActiveVoice {
    kind: VoiceKind,
    track_index: usize,
    note: u8,
    velocity: u8,
    instrument_volume: u8,
    base_pitch: u16,
    duration_remaining: u16,
    age: u64,
    released: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct StagedVoiceRegs {
    dirty: u32,
    ssa_bytes: u32,
    adsr1: u16,
    adsr2: u16,
    pitch: u16,
    reverb: bool,
    vol_l: i16,
    vol_r: i16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlayPhase {
    Playing,
    Fading,
    Tail,
    Finished,
}

/// PlayStation 1 `HOWL` music and SFX player driving [`libspu2::Device`] in PS1 SPU mode.
pub struct HowlPlayer {
    song_set: SongSet,
    sequence_index: usize,
    config: RenderConfig,
    device: Device,
    sample_addrs: BTreeMap<u16, u32>,
    upload: SpuUploadSummary,
    tracks: Vec<TrackRuntime>,
    voices: [Option<ActiveVoice>; VOICES_PER_CORE],
    staged: [StagedVoiceRegs; VOICES_PER_CORE],
    eon_mask: u32,
    voice_clock: u64,
    mute_mask: u64,
    solo_mask: u64,
    reverb_active: bool,
    tempo_fp16: u32,
    tempo_acc: u32,
    sample_accum_num: u64,
    samples_per_tick_num: u64,
    samples_per_tick_den: u64,
    sequence_volume_fp16: u32,
    initial_sequence_volume_fp16: u32,
    fade_step_fp16: u32,
    fade_elapsed_ticks: u32,
    fade_total_ticks: u32,
    tail_frames_remaining: u64,
    max_frames: u64,
    frames_rendered: u64,
    loops_completed: u32,
    faded_out: bool,
    reached_end: bool,
    phase: PlayPhase,
}

impl fmt::Debug for HowlPlayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HowlPlayer")
            .field("sequence_index", &self.sequence_index)
            .field("tracks", &self.tracks.len())
            .field("phase", &self.phase)
            .field("frames_rendered", &self.frames_rendered)
            .field("loops_completed", &self.loops_completed)
            .field("reverb_active", &self.reverb_active)
            .finish()
    }
}

impl HowlPlayer {
    /// Creates a new [`HowlPlayer`] from a [`HowlArchiveView`], selecting
    /// `song_set_index` and `sequence_index`, uploading all ADPCM samples
    /// referenced by that song set into a fresh PS1 [`Device`].
    pub fn from_archive(
        archive: &HowlArchiveView<'_>,
        song_set_index: usize,
        sequence_index: usize,
        config: RenderConfig,
    ) -> Result<Self> {
        let song_set = archive.song_set(song_set_index)?.to_owned()?;
        if sequence_index >= song_set.sequences.len() {
            return Err(Error::OutOfRange("sequence index out of range"));
        }

        // Collect all unique SPU sample IDs referenced by instruments and percussion.
        let mut needed_ids = BTreeSet::new();
        for inst in &song_set.instruments {
            needed_ids.insert(inst.spu_index);
        }
        for drum in &song_set.percussion {
            needed_ids.insert(drum.spu_index);
        }

        // Single-pass scan over each sample bank, borrowing ADPCM slices directly without cloning.
        let mut samples: BTreeMap<u16, &[u8]> = BTreeMap::new();
        let bank_count = archive.header().bank_count as usize;
        for bi in 0..bank_count {
            if samples.len() >= needed_ids.len() {
                break;
            }
            let Ok(bank) = archive.bank(bi) else {
                continue;
            };
            let adpcm_data = bank.adpcm_data();
            let mut cursor = 0usize;
            for slot in 0..bank.sample_count() {
                let Some(sid) = bank.sample_id(slot) else {
                    break;
                };
                let Some(entry) = archive.spu_entry(usize::from(sid)) else {
                    break;
                };
                let len = entry.byte_len();
                let Some(end) = cursor.checked_add(len) else {
                    break;
                };
                if end > adpcm_data.len() {
                    break;
                }
                if needed_ids.contains(&sid) && !samples.contains_key(&sid) {
                    samples.insert(sid, &adpcm_data[cursor..end]);
                }
                cursor = end;
            }
        }

        Self::from_slices(song_set, sequence_index, samples.into_iter(), config)
    }

    /// Creates a new [`HowlPlayer`] from an owned [`SongSet`] and an explicit
    /// map of `spu_index -> ADPCM bytes`.
    pub fn from_parts(
        song_set: SongSet,
        sequence_index: usize,
        samples: &BTreeMap<u16, Vec<u8>>,
        config: RenderConfig,
    ) -> Result<Self> {
        Self::from_slices(
            song_set,
            sequence_index,
            samples.iter().map(|(&sid, buf)| (sid, buf.as_slice())),
            config,
        )
    }

    fn from_slices<'a>(
        song_set: SongSet,
        sequence_index: usize,
        samples: impl Iterator<Item = (u16, &'a [u8])>,
        config: RenderConfig,
    ) -> Result<Self> {
        validate_config(&config)?;
        if sequence_index >= song_set.sequences.len() {
            return Err(Error::OutOfRange("sequence index out of range"));
        }

        let seq = &song_set.sequences[sequence_index];
        if seq.bpm == 0 || seq.tpqn == 0 {
            return Err(Error::InvalidArgument(
                "sequence bpm and tpqn must be non-zero",
            ));
        }
        let tempo_fp16 = compute_tempo_fp16(seq.bpm, seq.tpqn, config.tick_hz);
        if tempo_fp16 == 0 {
            return Err(Error::InvalidArgument(
                "sequence tempo produces zero ticks per frame",
            ));
        }

        let mut device = Device::new_ps1();
        if config.sample_rate != PS1_SAMPLE_RATE {
            device.set_output_sample_rate(config.sample_rate)?;
        }

        let mut sample_addrs = BTreeMap::new();
        let mut cursor = DEFAULT_SPU_RAM_BASE;
        for (sid, adpcm) in samples {
            if adpcm.is_empty() {
                continue;
            }
            let aligned_len = adpcm
                .len()
                .checked_add(7)
                .map(|n| n & !7)
                .ok_or(Error::OutOfRange("ADPCM sample length overflow"))?;
            let aligned_u32 = u32::try_from(aligned_len)
                .map_err(|_| Error::OutOfRange("ADPCM sample length exceeds 32-bit SPU RAM"))?;
            let end = cursor
                .checked_add(aligned_u32)
                .ok_or(Error::OutOfRange("SPU RAM upload overflow"))?;
            if end > MAX_SPU_RAM_END {
                return Err(Error::OutOfRange("SPU RAM capacity exceeded"));
            }
            device.write_ram(cursor, adpcm)?;
            sample_addrs.insert(sid, cursor);
            cursor = end;
        }

        let upload = SpuUploadSummary {
            base_addr: DEFAULT_SPU_RAM_BASE,
            end_addr: cursor,
            sample_count: sample_addrs.len(),
        };

        let initial_sequence_volume_fp16 = u32::from(config.sequence_volume) << 16;
        let fade_total_ticks = ((config.fade_seconds * f64::from(config.tick_hz)) + 0.5) as u32;
        let fade_step_fp16 = initial_sequence_volume_fp16
            .checked_div(fade_total_ticks)
            .map_or(initial_sequence_volume_fp16, |step| step.max(1));
        let max_frames = (config.max_seconds * f64::from(config.sample_rate)) as u64;

        let mut player = Self {
            song_set,
            sequence_index,
            config,
            device,
            sample_addrs,
            upload,
            tracks: Vec::new(),
            voices: [None; VOICES_PER_CORE],
            staged: [StagedVoiceRegs::default(); VOICES_PER_CORE],
            eon_mask: 0,
            voice_clock: 0,
            mute_mask: 0,
            solo_mask: 0,
            reverb_active: false,
            tempo_fp16,
            tempo_acc: 0,
            sample_accum_num: 0,
            samples_per_tick_num: u64::from(config.sample_rate),
            samples_per_tick_den: u64::from(config.tick_hz),
            sequence_volume_fp16: initial_sequence_volume_fp16,
            initial_sequence_volume_fp16,
            fade_step_fp16,
            fade_elapsed_ticks: 0,
            fade_total_ticks,
            tail_frames_remaining: 0,
            max_frames,
            frames_rendered: 0,
            loops_completed: 0,
            faded_out: false,
            reached_end: false,
            phase: PlayPhase::Playing,
        };
        player.reset_tracks();
        player.init_spu()?;
        Ok(player)
    }

    /// Returns a read-only reference to the underlying [`libspu2::Device`].
    #[inline]
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the active [`RenderConfig`].
    #[inline]
    #[must_use]
    pub const fn config(&self) -> RenderConfig {
        self.config
    }

    /// Returns a read-only reference to the loaded [`SongSet`].
    #[inline]
    #[must_use]
    pub fn song_set(&self) -> &SongSet {
        &self.song_set
    }

    /// Returns the SPU RAM upload summary for this player.
    #[inline]
    #[must_use]
    pub const fn upload_summary(&self) -> &SpuUploadSummary {
        &self.upload
    }

    /// Returns `true` if hardware SPU reverb is currently active.
    #[inline]
    #[must_use]
    pub const fn reverb_active(&self) -> bool {
        self.reverb_active
    }

    /// Returns the number of tracks in the active sequence.
    #[inline]
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Returns the elapsed playback time in seconds.
    #[inline]
    #[must_use]
    pub fn elapsed_seconds(&self) -> f64 {
        self.frames_rendered as f64 / f64::from(self.config.sample_rate)
    }

    /// Returns cumulative stereo frames rendered so far.
    #[inline]
    #[must_use]
    pub const fn frames_rendered(&self) -> u64 {
        self.frames_rendered
    }

    /// Returns the number of completed sequence loops observed so far.
    #[inline]
    #[must_use]
    pub const fn loops_completed(&self) -> u32 {
        self.loops_completed
    }

    /// Returns the current per-sequence volume scalar (`0..=255`).
    #[inline]
    #[must_use]
    pub const fn sequence_volume(&self) -> u8 {
        (self.sequence_volume_fp16 >> 16) as u8
    }

    /// Returns `true` while the player is actively ramping down during fade-out.
    #[inline]
    #[must_use]
    pub const fn is_fading(&self) -> bool {
        matches!(self.phase, PlayPhase::Fading)
    }

    /// Returns `true` if playback has finished (reached natural end, completed
    /// fade-out + tail, or hit `max_seconds`).
    #[inline]
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        matches!(self.phase, PlayPhase::Finished)
    }

    /// Returns the current track mute bitmask (`bit i = 1` means track `i` is muted).
    #[inline]
    #[must_use]
    pub const fn mute_mask(&self) -> u64 {
        self.mute_mask
    }

    /// Returns the current track solo bitmask (`0` disables solo).
    #[inline]
    #[must_use]
    pub const fn solo_mask(&self) -> u64 {
        self.solo_mask
    }

    /// Sets the track mute bitmask (`bit i = 1` mutes track `i`) and immediately
    /// flushes updated voice volumes to the SPU.
    #[inline]
    pub fn set_mute_mask(&mut self, mask: u64) {
        self.mute_mask = mask;
        self.refresh_all_music_volumes();
        let _ = self.commit_staged_registers();
    }

    /// Mutes or unmutes a single track (`0..64`) and immediately flushes updated
    /// voice volumes to the SPU.
    #[inline]
    pub fn set_track_muted(&mut self, track: usize, muted: bool) {
        if track >= 64 {
            return;
        }
        let bit = 1u64 << track;
        if muted {
            self.mute_mask |= bit;
        } else {
            self.mute_mask &= !bit;
        }
        self.refresh_all_music_volumes();
        let _ = self.commit_staged_registers();
    }

    /// Sets the track solo bitmask (non-zero mutes every track whose bit is `0`)
    /// and immediately flushes updated voice volumes to the SPU.
    #[inline]
    pub fn set_solo_mask(&mut self, mask: u64) {
        self.solo_mask = mask;
        self.refresh_all_music_volumes();
        let _ = self.commit_staged_registers();
    }

    /// Manually triggers a per-sequence-voice linear fade-out over
    /// [`RenderConfig::fade_seconds`].
    pub fn start_fade_out(&mut self) {
        if self.phase == PlayPhase::Playing {
            if self.fade_total_ticks == 0 {
                self.sequence_volume_fp16 = 0;
                self.faded_out = true;
                self.refresh_all_music_volumes();
                self.key_off_all_music();
                self.enter_tail();
                let _ = self.commit_staged_registers();
            } else {
                self.phase = PlayPhase::Fading;
                self.fade_elapsed_ticks = 0;
            }
        }
    }

    /// Returns a read-only snapshot of track `index`, if in range.
    #[must_use]
    pub fn track_snapshot(&self, index: usize) -> Option<TrackSnapshot> {
        let trk = self.tracks.get(index)?;
        let voice = self
            .voices
            .iter()
            .enumerate()
            .filter_map(|(vi, slot)| {
                slot.filter(|v| v.kind == VoiceKind::Music && v.track_index == index && !v.released)
                    .map(|v| (vi, v.age))
            })
            .max_by_key(|&(_, age)| age)
            .map(|(vi, _)| vi);
        let voice_state = voice.and_then(|v| self.device.inspect_voice(0, v).ok());
        Some(TrackSnapshot {
            index,
            id: trk.id,
            percussion: trk.percussion,
            active: trk.active,
            muted: self.is_track_muted(index),
            instrument: trk.instrument,
            volume: trk.volume,
            pan: trk.pan,
            pitch_bend: trk.pitch_bend,
            reverb: trk.reverb,
            voice,
            voice_state,
        })
    }

    /// Triggers a one-shot sound effect descriptor on a free SPU voice.
    ///
    /// Returns the allocated hardware voice index, or `None` if the sample was
    /// not uploaded or all 24 voices are busy.
    pub fn trigger_sfx(
        &mut self,
        sfx: SfxEntry,
        volume: u8,
        pan: u8,
        pitch_bend: u8,
    ) -> Option<usize> {
        let &ssa_bytes = self.sample_addrs.get(&sfx.spu_index)?;
        let voice_idx = self.allocate_voice()?;
        let pitch = compute_percussion_pitch(sfx.base_pitch, pitch_bend);
        let scaled_vol =
            (u32::from(self.config.sfx_volume) * u32::from(sfx.volume) * u32::from(volume)) >> 10;
        let (vol_l, vol_r) = compute_stereo_volume(scaled_vol, pan, self.config.stereo);

        self.voice_clock = self.voice_clock.wrapping_add(1);
        self.voices[voice_idx] = Some(ActiveVoice {
            kind: VoiceKind::Sfx,
            track_index: 0,
            note: 0,
            velocity: volume,
            instrument_volume: sfx.volume,
            base_pitch: sfx.base_pitch,
            duration_remaining: sfx.duration,
            age: self.voice_clock,
            released: false,
        });
        let reg = &mut self.staged[voice_idx];
        reg.dirty |= DIRTY_ALL_NOTE_ON;
        reg.ssa_bytes = ssa_bytes;
        reg.adsr1 = DEFAULT_PERCUSSION_ADSR1;
        reg.adsr2 = DEFAULT_PERCUSSION_ADSR2;
        reg.pitch = pitch;
        reg.reverb = false;
        reg.vol_l = vol_l;
        reg.vol_r = vol_r;
        Some(voice_idx)
    }

    /// Resets playback state, tracks, and SPU hardware state (voices, envelopes,
    /// and reverb work area) back to the start of the sequence while preserving
    /// uploaded ADPCM sample RAM.
    pub fn reset(&mut self) -> Result<()> {
        let base = self.upload.base_addr as usize;
        let end = self.upload.end_addr as usize;
        let saved = if end > base {
            self.device.ram()[base..end].to_vec()
        } else {
            Vec::new()
        };
        self.device.reset();
        if self.config.sample_rate != PS1_SAMPLE_RATE {
            self.device.set_output_sample_rate(self.config.sample_rate)?;
        }
        if !saved.is_empty() {
            self.device.write_ram(self.upload.base_addr, &saved)?;
        }

        self.voices = [None; VOICES_PER_CORE];
        self.staged = [StagedVoiceRegs::default(); VOICES_PER_CORE];
        self.eon_mask = 0;
        self.voice_clock = 0;
        self.tempo_acc = 0;
        self.sample_accum_num = 0;
        self.sequence_volume_fp16 = self.initial_sequence_volume_fp16;
        self.fade_elapsed_ticks = 0;
        self.tail_frames_remaining = 0;
        self.frames_rendered = 0;
        self.loops_completed = 0;
        self.faded_out = false;
        self.reached_end = false;
        self.phase = PlayPhase::Playing;
        self.reset_tracks();
        self.init_spu()
    }

    /// Renders up to `out.len()` stereo [`StereoI16`] frames, returning the
    /// number of frames actually written.
    pub fn render_samples(&mut self, out: &mut [StereoI16]) -> Result<usize> {
        let mut written = 0usize;
        while written < out.len() && self.phase != PlayPhase::Finished {
            let step = self.advance_until_renderable(out.len() - written)?;
            if step == 0 {
                break;
            }
            self.device
                .render_stereo(&mut out[written..written + step]);
            written += step;
            self.commit_rendered_frames(step);
        }
        Ok(written)
    }

    /// Renders planar stereo `f32` PCM (`[-1.0, 1.0]`) into `left` and `right`
    /// with zero heap allocation, returning the number of frames rendered.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if `left.len() != right.len()`, or
    /// [`Error::Spu`] if an SPU register write or planar render call fails.
    pub fn render_f32_planar(&mut self, left: &mut [f32], right: &mut [f32]) -> Result<usize> {
        if left.len() != right.len() {
            return Err(Error::InvalidArgument(
                "planar left and right buffers must have equal length",
            ));
        }
        let total = left.len();
        let mut written = 0usize;
        while written < total && self.phase != PlayPhase::Finished {
            let step = self.advance_until_renderable(total - written)?;
            if step == 0 {
                break;
            }
            self.device
                .render_f32_planar(&mut left[written..written + step], &mut right[written..written + step])?;
            written += step;
            self.commit_rendered_frames(step);
        }
        Ok(written)
    }

    /// Renders interleaved stereo `f32` PCM (`[L0, R0, ...]`) with zero heap
    /// allocation, returning the number of stereo frames rendered.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if `out.len()` is odd, or
    /// [`Error::Spu`] if an SPU register write or render call fails.
    pub fn render_f32_interleaved(&mut self, out: &mut [f32]) -> Result<usize> {
        if out.len() % 2 != 0 {
            return Err(Error::InvalidArgument(
                "interleaved stereo buffer length must be even",
            ));
        }
        let total_frames = out.len() / 2;
        let mut written = 0usize;
        let mut left_buf = [0.0f32; 256];
        let mut right_buf = [0.0f32; 256];
        while written < total_frames && self.phase != PlayPhase::Finished {
            let n = (total_frames - written).min(256);
            let got = self.render_f32_planar(&mut left_buf[..n], &mut right_buf[..n])?;
            if got == 0 {
                break;
            }
            for i in 0..got {
                let dst = (written + i) * 2;
                out[dst] = left_buf[i];
                out[dst + 1] = right_buf[i];
            }
            written += got;
        }
        Ok(written)
    }

    /// Renders the complete sequence according to [`RenderConfig`] and returns
    /// interleaved 16-bit stereo PCM plus playback metadata.
    pub fn render_all(&mut self) -> Result<RenderResult> {
        let est_seconds = self.config.max_seconds.min(10.0);
        let reserve_frames = (est_seconds * f64::from(self.config.sample_rate)) as usize;
        let mut samples = Vec::with_capacity(reserve_frames.saturating_mul(2));
        let mut chunk = [StereoI16::default(); 512];
        while !self.is_finished() {
            let n = self.render_samples(&mut chunk)?;
            if n == 0 {
                break;
            }
            samples.reserve(n * 2);
            for frame in &chunk[..n] {
                samples.push(frame.left);
                samples.push(frame.right);
            }
        }
        Ok(RenderResult {
            samples,
            frames_rendered: self.frames_rendered,
            loops_completed: self.loops_completed,
            faded_out: self.faded_out,
            reached_end: self.reached_end,
        })
    }

    /// Advances all due sequencer ticks before computing the next contiguous
    /// render chunk, preserving exact fractional Bresenham remainder when
    /// `sample_rate % tick_hz != 0`.
    fn advance_until_renderable(&mut self, max_wanted: usize) -> Result<usize> {
        loop {
            if self.phase == PlayPhase::Finished || max_wanted == 0 {
                return Ok(0);
            }
            let remaining_max = self.max_frames.saturating_sub(self.frames_rendered);
            if remaining_max == 0 {
                self.phase = PlayPhase::Finished;
                return Ok(0);
            }
            if self.phase == PlayPhase::Tail {
                if self.tail_frames_remaining == 0 {
                    self.phase = PlayPhase::Finished;
                    return Ok(0);
                }
                return Ok(max_wanted
                    .min(remaining_max as usize)
                    .min(self.tail_frames_remaining as usize));
            }
            if self.sample_accum_num < self.samples_per_tick_den {
                self.step_tick()?;
                self.sample_accum_num += self.samples_per_tick_num;
                continue;
            }
            let frames_until_tick = self.sample_accum_num / self.samples_per_tick_den;
            let chunk = max_wanted
                .min(remaining_max as usize)
                .min(frames_until_tick as usize);
            return Ok(chunk);
        }
    }

    fn commit_rendered_frames(&mut self, step: usize) {
        self.frames_rendered += step as u64;
        if self.phase == PlayPhase::Tail {
            self.tail_frames_remaining = self.tail_frames_remaining.saturating_sub(step as u64);
            if self.tail_frames_remaining == 0 {
                self.phase = PlayPhase::Finished;
            }
        } else {
            let consumed_num = (step as u64) * self.samples_per_tick_den;
            self.sample_accum_num -= consumed_num;
        }
    }

    fn reset_tracks(&mut self) {
        let seq = &self.song_set.sequences[self.sequence_index];
        let loop_enabled = self.config.target_loops > 0;
        self.tracks = seq
            .tracks
            .iter()
            .map(|track| {
                let first_delta = track.events.first().map_or(0, |e| e.delta);
                TrackRuntime {
                    active: !track.events.is_empty(),
                    percussion: track.is_percussion(),
                    loop_enabled,
                    pass_had_time: false,
                    id: track.id,
                    instrument: 0,
                    reverb: false,
                    volume: 0xFF,
                    pitch_bend: CENTER_PARAM,
                    pan: CENTER_PARAM,
                    event_index: 0,
                    wait_ticks: first_delta,
                    elapsed_ticks: 0,
                    loops_completed: 0,
                }
            })
            .collect();
    }

    fn init_spu(&mut self) -> Result<()> {
        let preset = self.config.reverb_preset;
        let work_size = preset.work_area_size() as usize;
        let bank_end = self.upload.end_addr as usize;
        let work_start = PS1_RAM_SIZE.saturating_sub(work_size) & !7;
        self.reverb_active = self.config.reverb_enabled
            && preset != ReverbPreset::Off
            && work_size > 0
            && work_start >= bank_end;
        let params = if self.reverb_active {
            preset.params()
        } else {
            ReverbParams::ZERO
        };
        let regs = params.to_ps1_registers();
        for (idx, &word) in regs.iter().enumerate() {
            self.device
                .write_ps1_register(ps1::RVB_BASE + (idx as u32) * 2, word)?;
        }
        let work_addr_reg = if self.reverb_active {
            (work_start >> 3) as u16
        } else {
            ((PS1_RAM_SIZE - 8) >> 3) as u16
        };
        self.device
            .write_ps1_register(ps1::RVB_WORK_ADDR, work_addr_reg)?;
        let mvol = self.config.master_volume & 0x7FFF;
        self.device.write_ps1_register(ps1::MVOLL, mvol)?;
        self.device.write_ps1_register(ps1::MVOLR, mvol)?;
        let depth_reg = if self.reverb_active {
            self.config.reverb_depth as u16
        } else {
            0
        };
        self.device.write_ps1_register(ps1::EVOLL, depth_reg)?;
        self.device.write_ps1_register(ps1::EVOLR, depth_reg)?;
        let mut ctrl = CONTROL_ENABLE | CONTROL_UNMUTE;
        if self.reverb_active {
            ctrl |= CONTROL_REVERB;
        }
        self.device.write_ps1_register(ps1::SPUCNT, ctrl)?;
        Ok(())
    }

    fn step_tick(&mut self) -> Result<()> {
        // 1. Expire timed one-shot SFX voices (5 ticks per 60 Hz frame).
        for vi in 0..VOICES_PER_CORE {
            if let Some(ref mut v) = self.voices[vi] {
                if v.kind == VoiceKind::Sfx && !v.released {
                    if v.duration_remaining <= 5 {
                        self.staged[vi].dirty |= DIRTY_KOFF;
                        self.staged[vi].dirty &= !DIRTY_KON;
                        v.released = true;
                    } else {
                        v.duration_remaining -= 5;
                    }
                }
            }
        }

        // 2. Advance fade-out ramp if active (linear voice-volume fade, MVOLL/MVOLR untouched).
        if self.phase == PlayPhase::Fading {
            self.fade_elapsed_ticks = self.fade_elapsed_ticks.saturating_add(1);
            if self.fade_elapsed_ticks >= self.fade_total_ticks
                || self.sequence_volume_fp16 <= self.fade_step_fp16
            {
                self.sequence_volume_fp16 = 0;
                self.faded_out = true;
                self.refresh_all_music_volumes();
                self.key_off_all_music();
                self.enter_tail();
                self.commit_staged_registers()?;
                return Ok(());
            }
            let prev_seq_vol = self.sequence_volume_fp16 >> 16;
            self.sequence_volume_fp16 = self.sequence_volume_fp16.saturating_sub(self.fade_step_fp16);
            if (self.sequence_volume_fp16 >> 16) != prev_seq_vol {
                self.refresh_all_music_volumes();
            }
        }

        // 3. Advance fixed-point tempo accumulator (16.16) and process track events.
        let sum = u64::from(self.tempo_acc) + u64::from(self.tempo_fp16);
        let delta_ticks = (sum >> 16) as u32;
        self.tempo_acc = (sum & 0xFFFF) as u32;

        let track_count = self.tracks.len();
        for ti in 0..track_count {
            if !self.tracks[ti].active {
                continue;
            }
            self.tracks[ti].elapsed_ticks = self.tracks[ti].elapsed_ticks.saturating_add(delta_ticks);
            let mut guard = 0u32;
            while self.tracks[ti].active
                && self.tracks[ti].elapsed_ticks >= self.tracks[ti].wait_ticks
                && guard < 4096
            {
                guard += 1;
                let wait = self.tracks[ti].wait_ticks;
                if wait > 0 {
                    self.tracks[ti].pass_had_time = true;
                }
                self.tracks[ti].elapsed_ticks = self.tracks[ti].elapsed_ticks.saturating_sub(wait);
                let ev_idx = self.tracks[ti].event_index;
                let event = self.song_set.sequences[self.sequence_index].tracks[ti]
                    .events
                    .get(ev_idx)
                    .copied();
                let Some(event) = event else {
                    self.tracks[ti].active = false;
                    break;
                };
                self.execute_command(ti, event.command);
                if !self.tracks[ti].active {
                    break;
                }
                let next_idx = self.tracks[ti].event_index;
                if let Some(next_ev) =
                    self.song_set.sequences[self.sequence_index].tracks[ti].events.get(next_idx)
                {
                    self.tracks[ti].wait_ticks = next_ev.delta;
                } else {
                    self.tracks[ti].active = false;
                    break;
                }
            }
        }

        // Check loop / completion transitions.
        if self.phase == PlayPhase::Playing {
            let any_active = self.tracks.iter().any(|t| t.active);
            if !any_active {
                self.reached_end = true;
                self.enter_tail();
            } else if self.config.target_loops > 0 {
                let min_loops = self
                    .tracks
                    .iter()
                    .filter(|t| t.active)
                    .map(|t| t.loops_completed)
                    .min()
                    .unwrap_or(0);
                if min_loops > self.loops_completed {
                    self.loops_completed = min_loops;
                }
                if self.loops_completed >= self.config.target_loops {
                    if self.fade_total_ticks == 0 {
                        self.faded_out = true;
                        self.key_off_all_music();
                        self.enter_tail();
                    } else {
                        self.phase = PlayPhase::Fading;
                    }
                }
            }
        }

        // 4. Commit all staged register changes in hardware order: KOFF -> attrs -> KON.
        self.commit_staged_registers()?;
        Ok(())
    }

    fn execute_command(&mut self, ti: usize, cmd: HowlCommand) {
        match cmd {
            HowlCommand::Nop | HowlCommand::Reserved2(_) | HowlCommand::Reserved4(_) => {
                self.tracks[ti].event_index += 1;
            }
            HowlCommand::NoteOff { note } => {
                self.release_note(ti, note);
                self.tracks[ti].event_index += 1;
            }
            HowlCommand::EndOfTrack => {
                if self.tracks[ti].loop_enabled && self.tracks[ti].pass_had_time {
                    self.tracks[ti].loops_completed =
                        self.tracks[ti].loops_completed.saturating_add(1);
                    self.tracks[ti].event_index = 0;
                    self.tracks[ti].pass_had_time = false;
                } else {
                    self.tracks[ti].active = false;
                    self.release_track(ti);
                }
            }
            HowlCommand::NoteOn { note, velocity } => {
                self.trigger_note(ti, note, velocity);
                self.tracks[ti].event_index += 1;
            }
            HowlCommand::SetVolume(vol) => {
                self.tracks[ti].volume = vol;
                self.refresh_track_volumes(ti);
                self.tracks[ti].event_index += 1;
            }
            HowlCommand::SetPan(pan) => {
                self.tracks[ti].pan = pan;
                self.refresh_track_volumes(ti);
                self.tracks[ti].event_index += 1;
            }
            HowlCommand::SetReverb(reverb) => {
                let enabled = reverb != 0;
                self.tracks[ti].reverb = enabled;
                for vi in 0..VOICES_PER_CORE {
                    if let Some(v) = self.voices[vi] {
                        if v.kind == VoiceKind::Music && v.track_index == ti {
                            self.staged[vi].reverb = enabled;
                            self.staged[vi].dirty |= DIRTY_REVERB;
                        }
                    }
                }
                self.tracks[ti].event_index += 1;
            }
            HowlCommand::SetInstrument(inst) => {
                self.tracks[ti].instrument = inst;
                self.tracks[ti].event_index += 1;
            }
            HowlCommand::SetPitchBend(bend) => {
                self.tracks[ti].pitch_bend = bend;
                self.refresh_track_pitches(ti);
                self.tracks[ti].event_index += 1;
            }
        }
    }

    fn trigger_note(&mut self, ti: usize, note: u8, velocity: u8) {
        let trk = self.tracks[ti];
        let seq_vol = self.sequence_volume_fp16 >> 16;
        if self.config.music_volume == 0 || trk.volume == 0 || seq_vol == 0 || velocity == 0 {
            return;
        }
        let (ssa_bytes, adsr1, adsr2, base_pitch, pitch, inst_vol) = if trk.percussion {
            let Some(&drum) = self.song_set.percussion.get(usize::from(note)) else {
                return;
            };
            let Some(&ssa) = self.sample_addrs.get(&drum.spu_index) else {
                return;
            };
            let pitch = compute_percussion_pitch(drum.base_pitch, trk.pitch_bend);
            (
                ssa,
                DEFAULT_PERCUSSION_ADSR1,
                DEFAULT_PERCUSSION_ADSR2,
                drum.base_pitch,
                pitch,
                drum.volume,
            )
        } else {
            let Some(&inst) = self.song_set.instruments.get(usize::from(trk.instrument)) else {
                return;
            };
            let Some(&ssa) = self.sample_addrs.get(&inst.spu_index) else {
                return;
            };
            let pitch = compute_melodic_pitch(inst.base_pitch, note, trk.pitch_bend);
            (ssa, inst.adsr1, inst.adsr2, inst.base_pitch, pitch, inst.volume)
        };

        let Some(vi) = self.allocate_voice() else {
            return;
        };

        let (vol_l, vol_r) = self.compute_music_voice_volume(ti, inst_vol, velocity);
        self.voice_clock = self.voice_clock.wrapping_add(1);
        self.voices[vi] = Some(ActiveVoice {
            kind: VoiceKind::Music,
            track_index: ti,
            note,
            velocity,
            instrument_volume: inst_vol,
            base_pitch,
            duration_remaining: 0,
            age: self.voice_clock,
            released: false,
        });
        let reg = &mut self.staged[vi];
        reg.dirty |= DIRTY_ALL_NOTE_ON;
        reg.ssa_bytes = ssa_bytes;
        reg.adsr1 = adsr1;
        reg.adsr2 = adsr2;
        reg.pitch = pitch;
        reg.reverb = trk.reverb;
        reg.vol_l = vol_l;
        reg.vol_r = vol_r;
    }

    fn release_note(&mut self, ti: usize, note: u8) {
        for vi in 0..VOICES_PER_CORE {
            if let Some(ref mut v) = self.voices[vi] {
                if !v.released && v.kind == VoiceKind::Music && v.track_index == ti && v.note == note
                {
                    self.staged[vi].dirty |= DIRTY_KOFF;
                    self.staged[vi].dirty &= !DIRTY_KON;
                    v.released = true;
                }
            }
        }
    }

    fn release_track(&mut self, ti: usize) {
        for vi in 0..VOICES_PER_CORE {
            if let Some(ref mut v) = self.voices[vi] {
                if !v.released && v.kind == VoiceKind::Music && v.track_index == ti {
                    self.staged[vi].dirty |= DIRTY_KOFF;
                    self.staged[vi].dirty &= !DIRTY_KON;
                    v.released = true;
                }
            }
        }
    }

    fn key_off_all_music(&mut self) {
        for vi in 0..VOICES_PER_CORE {
            if let Some(ref mut v) = self.voices[vi] {
                if !v.released && v.kind == VoiceKind::Music {
                    self.staged[vi].dirty |= DIRTY_KOFF;
                    self.staged[vi].dirty &= !DIRTY_KON;
                    v.released = true;
                }
            }
        }
    }

    /// 3-tier 24-voice allocator (skipping slots already pending `KON` on this tick):
    /// 1. First unowned or hardware-stopped slot (`Stopped` or `reached_end && envx == 0`).
    /// 2. Oldest released slot (actively decaying in ADSR release).
    /// 3. Oldest active slot.
    fn allocate_voice(&self) -> Option<usize> {
        // Tier 1: unowned or hardware-stopped slot.
        for vi in 0..VOICES_PER_CORE {
            if (self.staged[vi].dirty & DIRTY_KON) != 0 {
                continue;
            }
            if self.voices[vi].is_none() {
                return Some(vi);
            }
            if let Ok(st) = self.device.inspect_voice(0, vi) {
                if st.envelope_phase == EnvelopePhase::Stopped || (st.reached_end && st.envx == 0) {
                    return Some(vi);
                }
            }
        }

        // Tier 2: oldest released slot.
        let mut best_released: Option<(usize, u64)> = None;
        for vi in 0..VOICES_PER_CORE {
            if (self.staged[vi].dirty & DIRTY_KON) != 0 {
                continue;
            }
            if let Some(v) = self.voices[vi] {
                if v.released && best_released.map_or(true, |(_, age)| v.age < age) {
                    best_released = Some((vi, v.age));
                }
            }
        }
        if let Some((vi, _)) = best_released {
            return Some(vi);
        }

        // Tier 3: oldest active slot.
        let mut best_active: Option<(usize, u64)> = None;
        for vi in 0..VOICES_PER_CORE {
            if (self.staged[vi].dirty & DIRTY_KON) != 0 {
                continue;
            }
            if let Some(v) = self.voices[vi] {
                if best_active.map_or(true, |(_, age)| v.age < age) {
                    best_active = Some((vi, v.age));
                }
            }
        }
        best_active.map(|(vi, _)| vi)
    }

    fn is_track_muted(&self, ti: usize) -> bool {
        if ti >= 64 {
            return self.solo_mask != 0;
        }
        let bit = 1u64 << ti;
        if (self.mute_mask & bit) != 0 {
            return true;
        }
        self.solo_mask != 0 && (self.solo_mask & bit) == 0
    }

    fn compute_music_voice_volume(&self, ti: usize, inst_vol: u8, velocity: u8) -> (i16, i16) {
        if self.is_track_muted(ti) {
            return (0, 0);
        }
        let trk = self.tracks[ti];
        let seq_vol = (self.sequence_volume_fp16 >> 16) as u64;
        let base = (u64::from(self.config.music_volume) * seq_vol * u64::from(trk.volume)) >> 10;
        let vol = ((base * u64::from(inst_vol) * u64::from(velocity)) >> 15) as u32;
        compute_stereo_volume(vol, trk.pan, self.config.stereo)
    }

    fn refresh_track_volumes(&mut self, ti: usize) {
        for vi in 0..VOICES_PER_CORE {
            if let Some(v) = self.voices[vi] {
                if v.kind == VoiceKind::Music && v.track_index == ti {
                    let (vol_l, vol_r) =
                        self.compute_music_voice_volume(ti, v.instrument_volume, v.velocity);
                    if (vol_l, vol_r) != (self.staged[vi].vol_l, self.staged[vi].vol_r) {
                        self.staged[vi].vol_l = vol_l;
                        self.staged[vi].vol_r = vol_r;
                        self.staged[vi].dirty |= DIRTY_VOLUME;
                    }
                }
            }
        }
    }

    fn refresh_all_music_volumes(&mut self) {
        for vi in 0..VOICES_PER_CORE {
            if let Some(v) = self.voices[vi] {
                if v.kind == VoiceKind::Music {
                    let (vol_l, vol_r) = self.compute_music_voice_volume(
                        v.track_index,
                        v.instrument_volume,
                        v.velocity,
                    );
                    if (vol_l, vol_r) != (self.staged[vi].vol_l, self.staged[vi].vol_r) {
                        self.staged[vi].vol_l = vol_l;
                        self.staged[vi].vol_r = vol_r;
                        self.staged[vi].dirty |= DIRTY_VOLUME;
                    }
                }
            }
        }
    }

    fn refresh_track_pitches(&mut self, ti: usize) {
        let trk = self.tracks[ti];
        for vi in 0..VOICES_PER_CORE {
            if let Some(v) = self.voices[vi] {
                if v.kind == VoiceKind::Music && v.track_index == ti && !v.released {
                    let pitch = if trk.percussion {
                        compute_percussion_pitch(v.base_pitch, trk.pitch_bend)
                    } else {
                        compute_melodic_pitch(v.base_pitch, v.note, trk.pitch_bend)
                    };
                    if pitch != self.staged[vi].pitch {
                        self.staged[vi].pitch = pitch;
                        self.staged[vi].dirty |= DIRTY_PITCH;
                    }
                }
            }
        }
    }

    fn enter_tail(&mut self) {
        let tail_frames = (self.config.tail_seconds * f64::from(self.config.sample_rate)) as u64;
        if tail_frames == 0 {
            self.phase = PlayPhase::Finished;
        } else {
            self.tail_frames_remaining = tail_frames;
            self.phase = PlayPhase::Tail;
        }
    }

    /// Commits staged 24-voice register updates in exact hardware order
    /// (`KOFF` -> voice parameters + `EON` -> `KON`).
    fn commit_staged_registers(&mut self) -> Result<()> {
        let mut koff_mask: u32 = 0;
        for (vi, reg) in self.staged.iter_mut().enumerate() {
            if (reg.dirty & DIRTY_KOFF) != 0 {
                koff_mask |= 1u32 << vi;
                reg.dirty &= !DIRTY_KOFF;
            }
        }
        if koff_mask != 0 {
            self.device
                .write_ps1_register(ps1::KOFF_LO, (koff_mask & 0xFFFF) as u16)?;
            self.device
                .write_ps1_register(ps1::KOFF_HI, ((koff_mask >> 16) & 0xFF) as u16)?;
        }

        let mut kon_mask: u32 = 0;
        let mut eon_changed = false;
        for (vi, reg) in self.staged.iter_mut().enumerate() {
            let dirty = reg.dirty;
            if dirty == 0 {
                continue;
            }
            let base = (vi as u32) * ps1::VOICE_STRIDE;
            if (dirty & DIRTY_KON) != 0 {
                kon_mask |= 1u32 << vi;
            }
            if (dirty & DIRTY_SSA) != 0 {
                self.device
                    .write_ps1_register(base + ps1::V_START_ADDR, (reg.ssa_bytes >> 3) as u16)?;
            }
            if (dirty & DIRTY_ADSR) != 0 {
                self.device
                    .write_ps1_register(base + ps1::V_ADSR1, reg.adsr1)?;
                self.device
                    .write_ps1_register(base + ps1::V_ADSR2, reg.adsr2)?;
            }
            if (dirty & DIRTY_PITCH) != 0 {
                self.device
                    .write_ps1_register(base + ps1::V_PITCH, reg.pitch)?;
            }
            if (dirty & DIRTY_REVERB) != 0 {
                let bit = 1u32 << vi;
                let new_mask = if reg.reverb {
                    self.eon_mask | bit
                } else {
                    self.eon_mask & !bit
                };
                if new_mask != self.eon_mask {
                    self.eon_mask = new_mask;
                    eon_changed = true;
                }
            }
            if (dirty & DIRTY_VOLUME) != 0 {
                self.device
                    .write_ps1_register(base + ps1::V_VOLL, reg.vol_l as u16)?;
                self.device
                    .write_ps1_register(base + ps1::V_VOLR, reg.vol_r as u16)?;
            }
            reg.dirty = 0;
        }

        if eon_changed {
            self.device
                .write_ps1_register(ps1::EON_LO, (self.eon_mask & 0xFFFF) as u16)?;
            self.device
                .write_ps1_register(ps1::EON_HI, ((self.eon_mask >> 16) & 0xFF) as u16)?;
        }
        if kon_mask != 0 {
            self.device
                .write_ps1_register(ps1::KON_LO, (kon_mask & 0xFFFF) as u16)?;
            self.device
                .write_ps1_register(ps1::KON_HI, ((kon_mask >> 16) & 0xFF) as u16)?;
        }
        Ok(())
    }
}

fn validate_config(config: &RenderConfig) -> Result<()> {
    if config.sample_rate == 0 {
        return Err(Error::InvalidArgument("sample_rate must be non-zero"));
    }
    if config.tick_hz == 0 {
        return Err(Error::InvalidArgument("tick_hz must be non-zero"));
    }
    if config.fade_seconds < 0.0 || !config.fade_seconds.is_finite() {
        return Err(Error::InvalidArgument(
            "fade_seconds must be non-negative and finite",
        ));
    }
    if config.tail_seconds < 0.0 || !config.tail_seconds.is_finite() {
        return Err(Error::InvalidArgument(
            "tail_seconds must be non-negative and finite",
        ));
    }
    if config.max_seconds <= 0.0 || !config.max_seconds.is_finite() {
        return Err(Error::InvalidArgument(
            "max_seconds must be positive and finite",
        ));
    }
    Ok(())
}
