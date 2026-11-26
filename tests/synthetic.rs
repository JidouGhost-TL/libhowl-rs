// SPDX-License-Identifier: MIT OR Apache-2.0

//! Comprehensive synthetic unit and integration tests for `libhowl`.

use std::collections::BTreeMap;

use libhowl::{
    compute_melodic_pitch, compute_percussion_pitch, compute_stereo_volume, compute_tempo_fp16,
    compute_wide_pitch, read_vlq, Error, HowlArchive, HowlArchiveView, HowlCommand, HowlEvent,
    HowlPlayer, InstrumentEntry, RenderConfig, ReverbPreset, Sequence, SongSet, SongSetHeader,
    Track, CENTER_PARAM, HOWL_SECTOR_SIZE, NOTE_PITCH_LUT,
};
use libspu2::registers::ps1;
use libspu2::StereoI16;

fn make_looping_adpcm(blocks: usize) -> Vec<u8> {
    let mut blk = [0u8; 16];
    blk[0] = 0x0C; // shift=12, filter=0
    blk[1] = 0x03; // loop + end
    for (i, b) in blk[2..].iter_mut().enumerate() {
        *b = if i % 2 == 0 { 0x74 } else { 0x9B };
    }
    let mut out = Vec::with_capacity(blocks * 16);
    for _ in 0..blocks {
        out.extend_from_slice(&blk);
    }
    out
}

fn push_vlq(out: &mut Vec<u8>, mut val: u32) {
    let mut tmp = [0u8; 5];
    let mut n = 0usize;
    tmp[n] = (val & 0x7F) as u8;
    n += 1;
    val >>= 7;
    while val != 0 {
        tmp[n] = ((val & 0x7F) as u8) | 0x80;
        n += 1;
        val >>= 7;
    }
    for i in (0..n).rev() {
        out.push(tmp[i]);
    }
}

/// Builds a valid, sector-aligned synthetic `HOWL` archive containing:
/// - 2 SPU ADPCM samples (ID 0 = melodic wave, ID 1 = percussion/SFX wave)
/// - 1 standard SFX entry + 1 engine SFX entry
/// - 1 sample bank at sector 1 (ADPCM data at sector 2)
/// - 1 song set at sector 3 with 1 melodic instrument, 1 percussion entry,
///   and 1 two-track sequence exercising all 11 opcodes (`0x00..=0x0A`).
fn build_synthetic_howl() -> Vec<u8> {
    let wave0 = make_looping_adpcm(8); // 128 bytes
    let wave1 = make_looping_adpcm(8); // 128 bytes

    let n_spu: u32 = 2;
    let n_sfx: u32 = 1;
    let n_engine: u32 = 1;
    let n_banks: u32 = 1;
    let n_songs: u32 = 1;
    let hdr_table_bytes = n_spu * 4 + n_sfx * 8 + n_engine * 8 + n_banks * 2 + n_songs * 2;

    let bank_sector: u16 = 1;
    let song_sector: u16 = 3;

    // Build Track 0 (melodic) exercising opcodes 0x00..=0x0A.
    let mut trk0 = Vec::new();
    trk0.push(0x00); // flags: melodic
    trk0.push(0x00); // id: 0
    push_vlq(&mut trk0, 0);
    trk0.push(0x00); // Nop
    push_vlq(&mut trk0, 0);
    trk0.extend_from_slice(&[0x02, 0x01]); // Reserved2
    push_vlq(&mut trk0, 0);
    trk0.extend_from_slice(&[0x04, 0x02]); // Reserved4
    push_vlq(&mut trk0, 0);
    trk0.extend_from_slice(&[0x09, 0x00]); // SetInstrument(0)
    push_vlq(&mut trk0, 0);
    trk0.extend_from_slice(&[0x06, 0xF0]); // SetVolume(240)
    push_vlq(&mut trk0, 0);
    trk0.extend_from_slice(&[0x07, 0x60]); // SetPan(96)
    push_vlq(&mut trk0, 0);
    trk0.extend_from_slice(&[0x08, 0x01]); // SetReverb(1)
    push_vlq(&mut trk0, 0);
    trk0.extend_from_slice(&[0x05, 60, 127]); // NoteOn(C4, 127)
    push_vlq(&mut trk0, 16);
    trk0.extend_from_slice(&[0x0A, 0x90]); // SetPitchBend(0x90)
    push_vlq(&mut trk0, 16);
    trk0.extend_from_slice(&[0x01, 60]); // NoteOff(C4)
    push_vlq(&mut trk0, 128); // Multi-byte VLQ delta (>= 128)
    trk0.push(0x03); // EndOfTrack

    // Build Track 1 (percussion).
    let mut trk1 = Vec::new();
    trk1.push(0x01); // flags: percussion
    trk1.push(0x01); // id: 1
    push_vlq(&mut trk1, 8);
    trk1.extend_from_slice(&[0x05, 0, 120]); // NoteOn(drum 0, 120)
    push_vlq(&mut trk1, 24);
    trk1.extend_from_slice(&[0x0A, 0x88]); // SetPitchBend on percussion
    push_vlq(&mut trk1, 16);
    trk1.extend_from_slice(&[0x01, 0]); // NoteOff(drum 0)
    push_vlq(&mut trk1, 16);
    trk1.push(0x03); // EndOfTrack

    // Build Sequence 0 (`CSeq`).
    let mut seq = Vec::new();
    seq.push(0x00); // flags
    seq.push(0x02); // track_count = 2
    seq.extend_from_slice(&120u16.to_le_bytes()); // bpm = 120
    seq.extend_from_slice(&48u16.to_le_bytes()); // tpqn = 48
    let trk0_off: u16 = 0;
    let trk1_off: u16 = trk0.len() as u16;
    seq.extend_from_slice(&trk0_off.to_le_bytes());
    seq.extend_from_slice(&trk1_off.to_le_bytes());
    while (seq.len() & 3) != 0 {
        seq.push(0);
    }
    seq.extend_from_slice(&trk0);
    seq.extend_from_slice(&trk1);

    // Build SongSet 0.
    let mut song_set = Vec::new();
    song_set.extend_from_slice(&0u32.to_le_bytes()); // placeholder size
    song_set.push(1); // instrument_count
    song_set.push(1); // percussion_count
    song_set.extend_from_slice(&1u16.to_le_bytes()); // sequence_count
    // Instrument 0 (12 bytes): flags, vol, reserved, base_pitch, spu_index, adsr1, adsr2
    song_set.push(0x00);
    song_set.push(0xFF);
    song_set.extend_from_slice(&0u16.to_le_bytes());
    song_set.extend_from_slice(&0x1000u16.to_le_bytes());
    song_set.extend_from_slice(&0u16.to_le_bytes());
    song_set.extend_from_slice(&0x80FFu16.to_le_bytes());
    song_set.extend_from_slice(&0x1FC2u16.to_le_bytes());
    // Percussion 0 (8 bytes): flags, vol, base_pitch, spu_index, reserved
    song_set.push(0x00);
    song_set.push(0xE0);
    song_set.extend_from_slice(&0x1000u16.to_le_bytes());
    song_set.extend_from_slice(&1u16.to_le_bytes());
    song_set.extend_from_slice(&0u16.to_le_bytes());
    // Sequence offset table (1 * u16 = 0)
    song_set.extend_from_slice(&0u16.to_le_bytes());
    while (song_set.len() & 3) != 0 {
        song_set.push(0);
    }
    song_set.extend_from_slice(&seq);
    let ss_len = song_set.len() as u32;
    song_set[0..4].copy_from_slice(&ss_len.to_le_bytes());

    let total_size = (usize::from(song_sector) + 1) * HOWL_SECTOR_SIZE;
    let mut buf = vec![0u8; total_size];

    // Sector 0: Primary HOWL header + directory tables.
    buf[0x00..0x04].copy_from_slice(b"HOWL");
    buf[0x04..0x08].copy_from_slice(&128u32.to_le_bytes());
    buf[0x10..0x14].copy_from_slice(&n_spu.to_le_bytes());
    buf[0x14..0x18].copy_from_slice(&n_sfx.to_le_bytes());
    buf[0x18..0x1C].copy_from_slice(&n_engine.to_le_bytes());
    buf[0x1C..0x20].copy_from_slice(&n_banks.to_le_bytes());
    buf[0x20..0x24].copy_from_slice(&n_songs.to_le_bytes());
    buf[0x24..0x28].copy_from_slice(&hdr_table_bytes.to_le_bytes());

    let mut pos = 0x28;
    // SPU entries (2 * 4 bytes).
    buf[pos..pos + 2].copy_from_slice(&0u16.to_le_bytes());
    buf[pos + 2..pos + 4].copy_from_slice(&((wave0.len() / 8) as u16).to_le_bytes());
    pos += 4;
    buf[pos..pos + 2].copy_from_slice(&0u16.to_le_bytes());
    buf[pos + 2..pos + 4].copy_from_slice(&((wave1.len() / 8) as u16).to_le_bytes());
    pos += 4;
    // SFX entry 0 (8 bytes).
    buf[pos] = 0x00;
    buf[pos + 1] = 0xFF;
    buf[pos + 2..pos + 4].copy_from_slice(&0x1000u16.to_le_bytes());
    buf[pos + 4..pos + 6].copy_from_slice(&1u16.to_le_bytes());
    buf[pos + 6..pos + 8].copy_from_slice(&60u16.to_le_bytes());
    pos += 8;
    // Engine SFX entry 0 (8 bytes).
    buf[pos] = 0x00;
    buf[pos + 1] = 0xC0;
    buf[pos + 2..pos + 4].copy_from_slice(&0x0C00u16.to_le_bytes());
    buf[pos + 4..pos + 6].copy_from_slice(&0u16.to_le_bytes());
    buf[pos + 6..pos + 8].copy_from_slice(&0u16.to_le_bytes());
    pos += 8;
    // Bank sector table (1 * u16).
    buf[pos..pos + 2].copy_from_slice(&bank_sector.to_le_bytes());
    pos += 2;
    // Song-set sector table (1 * u16).
    buf[pos..pos + 2].copy_from_slice(&song_sector.to_le_bytes());

    // Sector 1: Bank 0 header (2 samples: ID 0 and ID 1).
    let bank_off = usize::from(bank_sector) * HOWL_SECTOR_SIZE;
    buf[bank_off..bank_off + 2].copy_from_slice(&2u16.to_le_bytes());
    buf[bank_off + 2..bank_off + 4].copy_from_slice(&0u16.to_le_bytes());
    buf[bank_off + 4..bank_off + 6].copy_from_slice(&1u16.to_le_bytes());

    // Sector 2: Bank 0 ADPCM payload.
    let adpcm_off = (usize::from(bank_sector) + 1) * HOWL_SECTOR_SIZE;
    buf[adpcm_off..adpcm_off + wave0.len()].copy_from_slice(&wave0);
    buf[adpcm_off + wave0.len()..adpcm_off + wave0.len() + wave1.len()].copy_from_slice(&wave1);

    // Sector 3: SongSet 0.
    let song_off = usize::from(song_sector) * HOWL_SECTOR_SIZE;
    buf[song_off..song_off + song_set.len()].copy_from_slice(&song_set);

    buf
}

#[test]
fn parse_synthetic_archive_and_all_opcodes() {
    let data = build_synthetic_howl();
    assert!(HowlArchiveView::detect(&data));

    let view = HowlArchiveView::parse(&data).unwrap();
    let hdr = view.header();
    assert_eq!(hdr.version, 128);
    assert_eq!(hdr.spu_entry_count, 2);
    assert_eq!(hdr.sfx_count, 1);
    assert_eq!(hdr.engine_sfx_count, 1);
    assert_eq!(hdr.bank_count, 1);
    assert_eq!(hdr.song_set_count, 1);

    let bank = view.bank(0).unwrap();
    assert_eq!(bank.sample_count(), 2);
    assert_eq!(bank.sample_id(0), Some(0));
    assert_eq!(bank.sample_id(1), Some(1));
    assert_eq!(bank.adpcm_data().len(), 256);
    assert!(view.find_sample_adpcm(0).is_some());
    assert!(view.find_sample_adpcm(1).is_some());
    assert!(view.find_sample_adpcm(99).is_none());

    let ss = view.song_set(0).unwrap();
    assert_eq!(ss.header().instrument_count, 1);
    assert_eq!(ss.header().percussion_count, 1);
    assert_eq!(ss.header().sequence_count, 1);

    let seq = ss.sequence(0).unwrap();
    assert_eq!(seq.bpm, 120);
    assert_eq!(seq.tpqn, 48);
    assert_eq!(seq.track_count, 2);

    let t0 = seq.track(0).unwrap();
    assert!(!t0.is_percussion());
    let events: Vec<_> = t0.events().map(|r| r.unwrap()).collect();
    assert_eq!(events.len(), 11);
    assert_eq!(events[0].command, HowlCommand::Nop);
    assert_eq!(events[1].command, HowlCommand::Reserved2(1));
    assert_eq!(events[2].command, HowlCommand::Reserved4(2));
    assert_eq!(events[3].command, HowlCommand::SetInstrument(0));
    assert_eq!(events[4].command, HowlCommand::SetVolume(0xF0));
    assert_eq!(events[5].command, HowlCommand::SetPan(0x60));
    assert_eq!(events[6].command, HowlCommand::SetReverb(1));
    assert_eq!(
        events[7].command,
        HowlCommand::NoteOn {
            note: 60,
            velocity: 127
        }
    );
    assert_eq!(events[8].command, HowlCommand::SetPitchBend(0x90));
    assert_eq!(events[9].command, HowlCommand::NoteOff { note: 60 });
    assert_eq!(events[10].delta, 128);
    assert_eq!(events[10].command, HowlCommand::EndOfTrack);

    let t1 = seq.track(1).unwrap();
    assert!(t1.is_percussion());
    let owned = HowlArchive::parse(&data).unwrap();
    assert_eq!(owned.header(), hdr);
    assert_eq!(owned.view().header(), hdr);
}

#[test]
fn rejects_malformed_and_random_inputs() {
    let valid = build_synthetic_howl();

    // Empty and truncated headers.
    assert!(!HowlArchiveView::detect(&[]));
    assert_eq!(
        HowlArchiveView::parse(&valid[..0x20]),
        Err(Error::UnexpectedEof("HOWL header"))
    );

    // Bad magic.
    let mut bad = valid.clone();
    bad[0] = b'X';
    assert_eq!(HowlArchiveView::parse(&bad), Err(Error::InvalidMagic));

    // Nonzero reserved word.
    let mut bad = valid.clone();
    bad[0x08] = 1;
    assert!(matches!(
        HowlArchiveView::parse(&bad),
        Err(Error::InvalidHeader(_))
    ));

    // Mismatched header_table_bytes.
    let mut bad = valid.clone();
    bad[0x24] ^= 0xFF;
    assert!(matches!(
        HowlArchiveView::parse(&bad),
        Err(Error::InvalidHeader(_))
    ));

    // Truncated ADPCM payload.
    assert!(HowlArchiveView::parse(&valid[..HOWL_SECTOR_SIZE * 2 + 16]).is_err());

    // Unknown opcode in command decoder.
    assert!(HowlCommand::decode(&[0x0B]).is_err());
    assert!(HowlCommand::decode(&[0x05, 60]).is_err());

    // VLQ overflow / EOF.
    assert!(read_vlq(&[]).is_err());
    assert!(read_vlq(&[0x80, 0x80, 0x80, 0x80, 0x80, 0x00]).is_err());

    // Deterministic xorshift garbage.
    let mut state = 0xDEAD_BEEF_CAFE_BABEu64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for len in [0, 4, 64, 512, 4096, 16384] {
        let buf: Vec<u8> = (0..len).map(|_| (next() & 0xFF) as u8).collect();
        assert!(!HowlArchiveView::detect(&buf));
    }
    assert!(!HowlArchiveView::detect(&vec![0u8; 8192]));
    assert!(!HowlArchiveView::detect(&vec![0xFFu8; 8192]));
}

#[test]
fn pitch_pan_and_tempo_math() {
    assert_eq!(NOTE_PITCH_LUT[60], 0x1000);
    assert_eq!(compute_melodic_pitch(0x1000, 60, CENTER_PARAM), 0x1000);
    assert_eq!(compute_melodic_pitch(0x1000, 72, CENTER_PARAM), 0x2000);
    assert_eq!(compute_melodic_pitch(0x1000, 48, CENTER_PARAM), 0x0800);
    // Fine bend upward increases pitch.
    assert!(compute_melodic_pitch(0x1000, 60, CENTER_PARAM + 16) > 0x1000);
    assert!(compute_melodic_pitch(0x1000, 60, CENTER_PARAM - 16) < 0x1000);

    // Percussion +/-1 octave.
    assert_eq!(compute_percussion_pitch(0x1000, CENTER_PARAM), 0x1000);
    assert_eq!(compute_percussion_pitch(0x1000, 0), 0x0800);
    assert!(compute_percussion_pitch(0x1000, 255) > 0x1F00);

    // Wide +/-2 octave.
    assert_eq!(compute_wide_pitch(0x1000, CENTER_PARAM), 0x1000);
    assert_eq!(compute_wide_pitch(0x1000, 0), 0x0400);

    // Stereo panning.
    let (l_left, r_left) = compute_stereo_volume(0x2000, 0, true);
    assert!(l_left > 0 && r_left == 0);
    let (l_right, r_right) = compute_stereo_volume(0x2000, 255, true);
    assert!(l_right == 0 && r_right > 0);
    let (l_mono, r_mono) = compute_stereo_volume(0x2000, 0, false);
    assert_eq!(l_mono, r_mono);

    // Tempo: 120 BPM * 48 TPQN / 60 = 96 ticks/sec -> 96/60 = 1.6 ticks/frame in 16.16.
    let fp16 = compute_tempo_fp16(120, 48, 60);
    assert_eq!(fp16, (96u32 << 16) / 60);
}

#[test]
fn player_renders_nonzero_pcm_and_reset_is_deterministic() {
    let data = build_synthetic_howl();
    let view = HowlArchiveView::parse(&data).unwrap();

    let cfg = RenderConfig {
        target_loops: 0,
        tail_seconds: 0.05,
        max_seconds: 3.0,
        reverb_preset: ReverbPreset::Room,
        ..Default::default()
    };

    let mut player = HowlPlayer::from_archive(&view, 0, 0, cfg).unwrap();
    assert_eq!(player.track_count(), 2);
    assert_eq!(player.upload_summary().sample_count, 2);

    let first = player.render_all().unwrap();
    assert!(first.reached_end);
    assert!(!first.faded_out);
    assert!(first.frames_rendered > 0);
    assert!(first.samples.iter().any(|&s| s != 0));

    player.reset().unwrap();
    let second = player.render_all().unwrap();
    assert_eq!(first.samples, second.samples);
}

#[test]
fn player_loops_and_fades_out() {
    let data = build_synthetic_howl();
    let view = HowlArchiveView::parse(&data).unwrap();

    let cfg = RenderConfig {
        target_loops: 1,
        fade_seconds: 0.2,
        tail_seconds: 0.05,
        max_seconds: 5.0,
        ..Default::default()
    };

    let mut player = HowlPlayer::from_archive(&view, 0, 0, cfg).unwrap();
    let res = player.render_all().unwrap();
    assert!(res.loops_completed >= 1);
    assert!(res.faded_out);
    assert!(res.samples.iter().any(|&s| s != 0));
}

#[test]
fn player_mute_solo_f32_and_sfx() {
    let data = build_synthetic_howl();
    let view = HowlArchiveView::parse(&data).unwrap();
    let sfx = view.sfx_entry(0).unwrap();

    let cfg = RenderConfig {
        target_loops: 0,
        tail_seconds: 0.02,
        max_seconds: 2.0,
        ..Default::default()
    };

    let mut player = HowlPlayer::from_archive(&view, 0, 0, cfg).unwrap();
    player.set_mute_mask(0b11);
    let muted = player.render_all().unwrap();
    assert!(muted.samples.iter().all(|&s| s == 0));

    // Planar and interleaved f32 streaming + one-shot SFX trigger.
    player.reset().unwrap();
    player.set_mute_mask(0);
    player.set_solo_mask(0b01);
    assert!(player.trigger_sfx(sfx, 200, CENTER_PARAM, CENTER_PARAM).is_some());

    let mut left = [0.0f32; 512];
    let mut right = [0.0f32; 512];
    let n = player.render_f32_planar(&mut left, &mut right).unwrap();
    assert!(n > 0);
    assert!(left[..n].iter().any(|&x| x != 0.0));

    let snap = player.track_snapshot(0).unwrap();
    assert!(!snap.muted);
    assert!(player.track_snapshot(1).unwrap().muted);

    let mut inter = [0.0f32; 512];
    let n2 = player.render_f32_interleaved(&mut inter).unwrap();
    assert!(n2 > 0);
    assert!(player.elapsed_seconds() > 0.0);

    // StereoI16 streaming chunk.
    let mut frames = [StereoI16::default(); 256];
    let n3 = player.render_samples(&mut frames).unwrap();
    assert!(n3 > 0);
}

#[test]
fn rejects_invalid_render_config() {
    let data = build_synthetic_howl();
    let view = HowlArchiveView::parse(&data).unwrap();

    let cfg = RenderConfig {
        sample_rate: 0,
        ..Default::default()
    };
    assert!(HowlPlayer::from_archive(&view, 0, 0, cfg).is_err());

    let cfg = RenderConfig {
        tick_hz: 0,
        ..Default::default()
    };
    assert!(HowlPlayer::from_archive(&view, 0, 0, cfg).is_err());

    let cfg = RenderConfig {
        fade_seconds: -1.0,
        ..Default::default()
    };
    assert!(HowlPlayer::from_archive(&view, 0, 0, cfg).is_err());
}

fn minimal_song_set(
    bpm: u16,
    tpqn: u16,
    instruments: Vec<InstrumentEntry>,
    tracks: Vec<Track>,
) -> SongSet {
    SongSet {
        header: SongSetHeader {
            size: 64,
            instrument_count: instruments.len() as u8,
            percussion_count: 0,
            sequence_count: 1,
        },
        instruments,
        percussion: Vec::new(),
        sequences: vec![Sequence {
            flags: 0,
            bpm,
            tpqn,
            tracks,
        }],
    }
}

fn default_inst(spu_index: u16, base_pitch: u16) -> InstrumentEntry {
    InstrumentEntry {
        flags: 0,
        volume: 0xFF,
        reserved: 0,
        base_pitch,
        spu_index,
        adsr1: 0x80FF,
        adsr2: 0x1FC2,
    }
}

#[test]
fn reverb_ram_overlap_guard_and_mvol_masking() {
    // Upload a sample large enough to overlap ReverbPreset::Echo (0x18040 bytes at top of 512 KiB RAM,
    // work_start = 0x80000 - 0x18040 = 0x67FC0), while still fitting below MAX_SPU_RAM_END (0x7E000).
    let mut big_wave = make_looping_adpcm(27_000); // 432,000 bytes (0x69780), end = 0x6A790 > 0x67FC0
    big_wave.truncate(0x69780);
    let mut samples = BTreeMap::new();
    samples.insert(0u16, big_wave);

    let ss = minimal_song_set(
        120,
        48,
        vec![default_inst(0, 0x1000)],
        vec![Track {
            flags: 0,
            id: 0,
            events: vec![
                HowlEvent {
                    delta: 0,
                    command: HowlCommand::NoteOn {
                        note: 60,
                        velocity: 120,
                    },
                },
                HowlEvent {
                    delta: 16,
                    command: HowlCommand::EndOfTrack,
                },
            ],
        }],
    );

    // Echo work area overlaps uploaded sample -> reverb must be disabled automatically.
    let cfg_echo = RenderConfig {
        reverb_preset: ReverbPreset::Echo,
        master_volume: 0xBFFF, // bit 15 set: must be masked to 0x3FFF
        target_loops: 0,
        tail_seconds: 0.0,
        ..Default::default()
    };
    let player_echo = HowlPlayer::from_parts(ss.clone(), 0, &samples, cfg_echo).unwrap();
    assert!(!player_echo.reverb_active());
    assert_eq!(
        player_echo.device().read_ps1_register(ps1::MVOLL).unwrap(),
        0x3FFF
    );
    assert_eq!(
        player_echo.device().read_ps1_register(ps1::MVOLR).unwrap(),
        0x3FFF
    );

    // Room work area (0x26C0, starts at 0x7D940 > 0x6A790) fits cleanly -> reverb active.
    let cfg_room = RenderConfig {
        reverb_preset: ReverbPreset::Room,
        ..cfg_echo
    };
    let player_room = HowlPlayer::from_parts(ss, 0, &samples, cfg_room).unwrap();
    assert!(player_room.reverb_active());
}

#[test]
fn fade_out_leaves_mvol_untouched_and_supports_manual_and_zero_fade() {
    let data = build_synthetic_howl();
    let view = HowlArchiveView::parse(&data).unwrap();

    let cfg = RenderConfig {
        target_loops: 10,
        fade_seconds: 0.2,
        tail_seconds: 0.05,
        max_seconds: 5.0,
        ..Default::default()
    };
    let mut player = HowlPlayer::from_archive(&view, 0, 0, cfg).unwrap();
    assert_eq!(player.config(), cfg);
    assert_eq!(player.song_set().sequences.len(), 1);
    assert!(!player.is_fading());
    assert_eq!(player.sequence_volume(), cfg.sequence_volume);

    // Render a bit, then manually trigger fade-out.
    let mut buf = [StereoI16::default(); 735];
    player.render_samples(&mut buf).unwrap();
    player.start_fade_out();
    assert!(player.is_fading());

    let mut prev_vol = player.sequence_volume();
    while player.is_fading() {
        player.render_samples(&mut buf).unwrap();
        let cur = player.sequence_volume();
        assert!(cur <= prev_vol, "fade volume must be monotonic");
        prev_vol = cur;
        assert_eq!(player.device().read_ps1_register(ps1::MVOLL).unwrap(), 0x3FFF);
        assert_eq!(player.device().read_ps1_register(ps1::MVOLR).unwrap(), 0x3FFF);
    }
    assert_eq!(player.sequence_volume(), 0);
    assert!(player.frames_rendered() > 0);

    // Zero fade + zero tail finishes immediately after target_loops.
    let cfg_zero = RenderConfig {
        target_loops: 1,
        fade_seconds: 0.0,
        tail_seconds: 0.0,
        max_seconds: 5.0,
        ..Default::default()
    };
    let mut p0 = HowlPlayer::from_archive(&view, 0, 0, cfg_zero).unwrap();
    let res = p0.render_all().unwrap();
    assert!(res.faded_out);
    assert_eq!(p0.loops_completed(), res.loops_completed);
    assert!(p0.is_finished());
}

#[test]
fn mute_beats_solo_and_flushes_immediately() {
    let data = build_synthetic_howl();
    let view = HowlArchiveView::parse(&data).unwrap();
    let cfg = RenderConfig {
        target_loops: 0,
        tail_seconds: 0.1,
        max_seconds: 2.0,
        ..Default::default()
    };
    let mut player = HowlPlayer::from_archive(&view, 0, 0, cfg).unwrap();

    // Solo track 0 AND explicitly mute track 0 -> mute must win!
    player.set_solo_mask(0b01);
    player.set_track_muted(0, true);
    assert_eq!(player.solo_mask(), 0b01);
    assert_eq!(player.mute_mask(), 0b01);
    assert!(player.track_snapshot(0).unwrap().muted);
    assert!(player.track_snapshot(1).unwrap().muted);

    // Render a tick so track 0 triggers its note; because it is muted, voice volume is 0.
    let mut buf = [StereoI16::default(); 735];
    player.render_samples(&mut buf).unwrap();
    assert!(buf.iter().all(|s| s.left == 0 && s.right == 0));

    // Unmute track 0 -> SPU volume must flush immediately before next tick.
    player.set_track_muted(0, false);
    assert!(!player.track_snapshot(0).unwrap().muted);
    let voice = player.track_snapshot(0).unwrap().voice.unwrap();
    let base = (voice as u32) * ps1::VOICE_STRIDE;
    let voll = player.device().read_ps1_register(base + ps1::V_VOLL).unwrap();
    assert_ne!(voll, 0, "unmuting must immediately flush V_VOLL to SPU");
}

#[test]
fn f32_buffer_validation_and_zero_alloc_equivalence() {
    let data = build_synthetic_howl();
    let view = HowlArchiveView::parse(&data).unwrap();
    let cfg = RenderConfig {
        target_loops: 0,
        tail_seconds: 0.02,
        max_seconds: 1.0,
        ..Default::default()
    };
    let mut p_i16 = HowlPlayer::from_archive(&view, 0, 0, cfg).unwrap();
    let mut p_planar = HowlPlayer::from_archive(&view, 0, 0, cfg).unwrap();
    let mut p_inter = HowlPlayer::from_archive(&view, 0, 0, cfg).unwrap();

    // Mismatched planar lengths -> InvalidArgument.
    let mut l_short = [0.0f32; 16];
    let mut r_long = [0.0f32; 32];
    assert!(matches!(
        p_planar.render_f32_planar(&mut l_short, &mut r_long),
        Err(Error::InvalidArgument(_))
    ));

    // Odd interleaved length -> InvalidArgument.
    let mut odd = [0.0f32; 15];
    assert!(matches!(
        p_inter.render_f32_interleaved(&mut odd),
        Err(Error::InvalidArgument(_))
    ));

    // Compare rendered frames across all three APIs.
    let mut s_i16 = [StereoI16::default(); 1024];
    let mut left = [0.0f32; 1024];
    let mut right = [0.0f32; 1024];
    let mut inter = [0.0f32; 2048];
    let n1 = p_i16.render_samples(&mut s_i16).unwrap();
    let n2 = p_planar.render_f32_planar(&mut left, &mut right).unwrap();
    let n3 = p_inter.render_f32_interleaved(&mut inter).unwrap();
    assert_eq!(n1, n2);
    assert_eq!(n2, n3);
    for i in 0..n1 {
        assert!((left[i] - inter[i * 2]).abs() < 1e-6);
        assert!((right[i] - inter[i * 2 + 1]).abs() < 1e-6);
        assert!((left[i] - f32::from(s_i16[i].left) / 32768.0).abs() <= 1.0 / 32768.0 + 1e-5);
    }
}

#[test]
fn voice_stealing_and_sustaining_pitch_bend_across_instrument_switch() {
    let mut samples = BTreeMap::new();
    samples.insert(0u16, make_looping_adpcm(8));

    // 2 instruments with distinct base_pitch: inst 0 = 0x1000, inst 1 = 0x2000.
    // Track 0 triggers note 60 with inst 0, switches to inst 1, then bends pitch.
    // The sustaining voice must still bend relative to inst 0 (0x1000), not inst 1 (0x2000)!
    let ss_bend = minimal_song_set(
        120,
        48,
        vec![default_inst(0, 0x1000), default_inst(0, 0x2000)],
        vec![Track {
            flags: 0,
            id: 0,
            events: vec![
                HowlEvent {
                    delta: 0,
                    command: HowlCommand::SetInstrument(0),
                },
                HowlEvent {
                    delta: 0,
                    command: HowlCommand::NoteOn {
                        note: 60,
                        velocity: 120,
                    },
                },
                HowlEvent {
                    delta: 2,
                    command: HowlCommand::SetInstrument(1),
                },
                HowlEvent {
                    delta: 2,
                    command: HowlCommand::SetPitchBend(0x90),
                },
                HowlEvent {
                    delta: 16,
                    command: HowlCommand::EndOfTrack,
                },
            ],
        }],
    );
    let mut player = HowlPlayer::from_parts(
        ss_bend,
        0,
        &samples,
        RenderConfig {
            target_loops: 0,
            tail_seconds: 0.0,
            ..Default::default()
        },
    )
    .unwrap();
    let mut buf = [StereoI16::default(); 735 * 6];
    player.render_samples(&mut buf).unwrap();
    let voice = player.track_snapshot(0).unwrap().voice.unwrap();
    let actual_pitch = player
        .device()
        .read_ps1_register((voice as u32) * ps1::VOICE_STRIDE + ps1::V_PITCH)
        .unwrap();
    let expected_pitch = compute_melodic_pitch(0x1000, 60, 0x90);
    assert_eq!(actual_pitch, expected_pitch);

    // >24 simultaneous NoteOns on a single tick must not panic or steal same-tick KON slots.
    let mut events = vec![HowlEvent {
        delta: 0,
        command: HowlCommand::SetInstrument(0),
    }];
    for note in 36..66u8 {
        events.push(HowlEvent {
            delta: 0,
            command: HowlCommand::NoteOn {
                note,
                velocity: 100,
            },
        });
    }
    events.push(HowlEvent {
        delta: 8,
        command: HowlCommand::NoteOff { note: 36 },
    });
    events.push(HowlEvent {
        delta: 8,
        command: HowlCommand::NoteOn {
            note: 72,
            velocity: 100,
        },
    });
    events.push(HowlEvent {
        delta: 8,
        command: HowlCommand::EndOfTrack,
    });
    let ss_poly = minimal_song_set(
        120,
        48,
        vec![default_inst(0, 0x1000)],
        vec![Track {
            flags: 0,
            id: 0,
            events,
        }],
    );
    let mut p_poly = HowlPlayer::from_parts(
        ss_poly,
        0,
        &samples,
        RenderConfig {
            target_loops: 0,
            tail_seconds: 0.02,
            max_seconds: 2.0,
            ..Default::default()
        },
    )
    .unwrap();
    let res = p_poly.render_all().unwrap();
    assert!(res.reached_end);
    assert!(res.samples.iter().any(|&s| s != 0));
}

#[test]
fn zero_delay_loop_guard_zero_tempo_guard_and_non_standard_rates() {
    let mut samples = BTreeMap::new();
    samples.insert(0u16, make_looping_adpcm(8));

    // Zero-delay EndOfTrack loop with target_loops > 0 must halt cleanly with 0 loops_completed.
    let ss_zero_loop = minimal_song_set(
        120,
        48,
        vec![default_inst(0, 0x1000)],
        vec![Track {
            flags: 0,
            id: 0,
            events: vec![
                HowlEvent {
                    delta: 0,
                    command: HowlCommand::Nop,
                },
                HowlEvent {
                    delta: 0,
                    command: HowlCommand::EndOfTrack,
                },
            ],
        }],
    );
    let mut p_zl = HowlPlayer::from_parts(
        ss_zero_loop,
        0,
        &samples,
        RenderConfig {
            target_loops: 2,
            tail_seconds: 0.0,
            max_seconds: 2.0,
            ..Default::default()
        },
    )
    .unwrap();
    let res = p_zl.render_all().unwrap();
    assert!(res.reached_end);
    assert_eq!(res.loops_completed, 0);
    assert_eq!(res.frames_rendered, 0);

    // bpm == 0 or tpqn == 0 must be rejected.
    let ss_bad_bpm = minimal_song_set(0, 48, vec![], vec![]);
    assert!(HowlPlayer::from_parts(ss_bad_bpm, 0, &samples, RenderConfig::default()).is_err());
    let ss_bad_tpqn = minimal_song_set(120, 0, vec![], vec![]);
    assert!(HowlPlayer::from_parts(ss_bad_tpqn, 0, &samples, RenderConfig::default()).is_err());

    // Non-44.1 kHz (48000 Hz) and non-60 Hz (50 Hz, 120 Hz) timing.
    let data = build_synthetic_howl();
    let view = HowlArchiveView::parse(&data).unwrap();
    for (sample_rate, tick_hz) in [(48_000, 60), (44_100, 50), (44_100, 120)] {
        let cfg = RenderConfig {
            sample_rate,
            tick_hz,
            target_loops: 0,
            tail_seconds: 0.02,
            max_seconds: 3.0,
            ..Default::default()
        };
        let mut p = HowlPlayer::from_archive(&view, 0, 0, cfg).unwrap();
        let r = p.render_all().unwrap();
        assert!(r.reached_end, "failed for {sample_rate}Hz @ {tick_hz}Hz");
        assert!(r.samples.iter().any(|&s| s != 0));
    }
}
