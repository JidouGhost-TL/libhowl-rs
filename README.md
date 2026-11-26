# libhowl

`libhowl` is a standalone, `#![no_std]` (`alloc`-compatible), `#![forbid(unsafe_code)]`
pure-Rust PlayStation 1 **`HOWL`** audio bank and music sequencer built directly
on top of [`libspu2`](https://github.com/JidouGhost-TL/libspu2-rs).

## Binary Format Overview

A `HOWL` archive (little-endian) packs SPU-ADPCM sample banks, SFX descriptors,
and multi-track music sequence sets into a single 2048-byte sector-aligned container:

- **Primary Header (`0x28` bytes)**: `"HOWL"` magic (`+0x00`), `version` (`+0x04`),
  two zero reserved `u32` words (`+0x08..0x10`), `spu_entry_count` (`+0x10`),
  `sfx_count` (`+0x14`), `engine_sfx_count` (`+0x18`), `bank_count` (`+0x1C`),
  `song_set_count` (`+0x20`), and exact `header_table_bytes` (`+0x24`).
- **Directory Tables (starting at `+0x28`)**:
  - `spu_entry_count × 4` B SPU sample entries (`u16 spu_addr_div8`, `u16 size_div8`).
  - `sfx_count × 8` B standard SFX descriptors.
  - `engine_sfx_count × 8` B wide-pitch (`±2` octave) SFX descriptors.
  - `bank_count × 2` B `u16` sector offsets to ADPCM sample banks.
  - `song_set_count × 2` B `u16` sector offsets to music song sets.
- **Sample Banks (`sector × 0x800`)**: 1-sector header containing `u16 sample_count`
  and `sample_count × u16` SPU sample IDs, followed at `(sector + 1) × 0x800` by
  the concatenated PS1 SPU-ADPCM waveforms.
- **Song Sets (`sector × 0x800`)**: 8-byte header (`u32 size`, `u8 instrument_count`,
  `u8 percussion_count`, `u16 sequence_count`), `12`-byte melodic instrument entries
  (with raw PS1 `ADSR1`/`ADSR2`), `8`-byte percussion entries, `u16` sequence offsets
  (4-byte aligned), and per-sequence track tables.
- **Track Bytecode (`0x00..=0x0A`)**: MIDI-style 7-bit VLQ delta times preceding
  11 compact commands: `0x00` Nop, `0x01` NoteOff, `0x02`/`0x04` Reserved,
  `0x03` EndOfTrack/Loop, `0x05` NoteOn, `0x06` SetVolume, `0x07` SetPan,
  `0x08` SetReverb, `0x09` SetInstrument, `0x0A` SetPitchBend.

## Hardware-Faithful Playback

- **60 Hz sequencer clock** with 16.16 fixed-point tempo accumulator
  (`(((bpm * tpqn) / 60) << 16) / 60`).
- **24-voice PS1 SPU state machine** with hardware-ordered register staging
  (`KOFF` -> voice registers + `EON` -> `KON`).
- **12-TET pitch + fractional fine bend** (`±2` semitones melodic, `±1` octave
  percussion/SFX) and constant-power quarter-sine stereo panning.
- **10 PlayStation 1 SPU hardware reverb presets** with RAM-overlap guarding.
- **Linear voice-volume fade-out** leaving `MVOLL`/`MVOLR` untouched so release
  envelopes and reverb tails decay naturally.

## Quick Example

```rust
use alloc::collections::BTreeMap;
use libhowl::{
    HowlCommand, HowlEvent, HowlPlayer, InstrumentEntry, RenderConfig, Sequence, SongSet,
    SongSetHeader, Track,
};

extern crate alloc;

// Build a minimal looping PS1 SPU-ADPCM waveform (128 bytes).
let mut adpcm = vec![0u8; 16];
adpcm[1] = 0x03; // loop + end
for (i, b) in adpcm[2..].iter_mut().enumerate() {
    *b = if i % 2 == 0 { 0x74 } else { 0x9B };
}
let wave = adpcm.repeat(8);

let mut samples = BTreeMap::new();
samples.insert(0u16, wave);

let song_set = SongSet {
    header: SongSetHeader {
        size: 64,
        instrument_count: 1,
        percussion_count: 0,
        sequence_count: 1,
    },
    instruments: vec![InstrumentEntry {
        flags: 0,
        volume: 0xFF,
        reserved: 0,
        base_pitch: 0x1000,
        spu_index: 0,
        adsr1: 0x80FF,
        adsr2: 0x1FC2,
    }],
    percussion: Vec::new(),
    sequences: vec![Sequence {
        flags: 0,
        bpm: 120,
        tpqn: 48,
        tracks: vec![Track {
            flags: 0,
            id: 0,
            events: vec![
                HowlEvent { delta: 0, command: HowlCommand::SetInstrument(0) },
                HowlEvent { delta: 0, command: HowlCommand::NoteOn { note: 60, velocity: 127 } },
                HowlEvent { delta: 24, command: HowlCommand::NoteOff { note: 60 } },
                HowlEvent { delta: 12, command: HowlCommand::EndOfTrack },
            ],
        }],
    }],
};

let mut cfg = RenderConfig::default();
cfg.target_loops = 0;
cfg.tail_seconds = 0.05;
cfg.max_seconds = 1.0;

let mut player = HowlPlayer::from_parts(song_set, 0, &samples, cfg).unwrap();
let rendered = player.render_all().unwrap();
assert!(rendered.samples.iter().any(|&s| s != 0));
```

## License

Dual-licensed under MIT (`LICENSE-MIT`) or Apache-2.0 (`LICENSE-APACHE`), at your option.
