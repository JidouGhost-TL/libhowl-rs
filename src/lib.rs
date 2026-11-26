// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(feature = "alloc", doc = include_str!("../README.md"))]
#![cfg_attr(
    not(feature = "alloc"),
    doc = "Pure-Rust PlayStation 1 `HOWL` audio bank and music sequencer built on `libspu2`.\n\n\
           Built without the `alloc` feature: exposes zero-copy archive and sequence views \
           (`HowlArchiveView`, `SampleBankView`, `SongSetView`, `SequenceView`, `TrackView`, \
           `TrackEventIter`), pitch/pan/tempo helpers, `Error`/`Result`, and the 10 hardware \
           SPU reverb presets (`ReverbPreset`, `ReverbParams`). Enable the `alloc` or `std` \
           feature for owned containers (`HowlArchive`, `SongSet`) and `HowlPlayer`."
)]
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;

use core::fmt;

pub mod bank;
#[cfg(feature = "alloc")]
pub mod player;
pub mod reverb;
pub mod sequence;
pub mod tables;

#[cfg(feature = "alloc")]
pub use bank::HowlArchive;
pub use bank::{
    EngineSfxEntry, HowlArchiveView, HowlHeader, SampleBankView, SfxEntry, SpuSampleEntry,
    ENGINE_SFX_ENTRY_SIZE, HOWL_HEADER_SIZE, HOWL_MAGIC, HOWL_SECTOR_SIZE, SFX_ENTRY_SIZE,
    SPU_ENTRY_SIZE,
};
#[cfg(feature = "alloc")]
pub use player::{
    HowlPlayer, RenderConfig, RenderResult, SpuUploadSummary, TrackSnapshot,
    DEFAULT_MUSIC_VOLUME, DEFAULT_PERCUSSION_ADSR1, DEFAULT_PERCUSSION_ADSR2,
    DEFAULT_SEQUENCE_VOLUME, DEFAULT_SFX_VOLUME, DEFAULT_SPU_RAM_BASE, HOWL_TICK_HZ,
    MAX_SPU_RAM_END,
};
pub use reverb::{ReverbParams, ReverbPreset};
#[cfg(feature = "alloc")]
pub use sequence::{Sequence, SongSet, Track};
pub use sequence::{
    read_vlq, HowlCommand, HowlEvent, InstrumentEntry, PercussionEntry, SequenceView,
    SongSetHeader, SongSetView, TrackEventIter, TrackView, INSTRUMENT_ENTRY_SIZE,
    PERCUSSION_ENTRY_SIZE, SEQUENCE_HEADER_SIZE, SONG_SET_HEADER_SIZE, TRACK_FLAG_PERCUSSION,
};
pub use tables::{
    compute_melodic_pitch, compute_percussion_pitch, compute_stereo_volume, compute_tempo_fp16,
    compute_wide_pitch, CENTER_PARAM, FINE_PITCH_LUT, MAX_SPU_VOLUME, NOTE_PITCH_LUT, PAN_LUT,
    PITCH_SCALE_1OCT_LUT, PITCH_SCALE_2OCT_LUT,
};

/// Crate-wide result type alias for `libhowl`.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors returned by `libhowl` parsing, upload, and playback APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Error {
    /// Input buffer was shorter than required for a header, table, or operand.
    UnexpectedEof(&'static str),
    /// Input did not begin with the `"HOWL"` magic signature.
    InvalidMagic,
    /// Header field or table violated a structural invariant.
    InvalidHeader(&'static str),
    /// Offset, index, VLQ delta, or SPU RAM address was out of range.
    OutOfRange(&'static str),
    /// An invalid configuration or argument was supplied.
    InvalidArgument(&'static str),
    /// An underlying `libspu2` device error occurred.
    Spu(libspu2::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof(ctx) => write!(f, "unexpected end of data: {ctx}"),
            Self::InvalidMagic => write!(f, "invalid HOWL magic header"),
            Self::InvalidHeader(msg) => write!(f, "invalid HOWL header: {msg}"),
            Self::OutOfRange(msg) => write!(f, "out of range: {msg}"),
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Self::Spu(err) => write!(f, "SPU error: {err}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spu(err) => Some(err),
            _ => None,
        }
    }
}

impl From<libspu2::Error> for Error {
    fn from(err: libspu2::Error) -> Self {
        Self::Spu(err)
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Error>();
    assert_send_sync::<HowlHeader>();
    assert_send_sync::<SpuSampleEntry>();
    assert_send_sync::<SfxEntry>();
    assert_send_sync::<EngineSfxEntry>();
    assert_send_sync::<SampleBankView<'static>>();
    assert_send_sync::<HowlArchiveView<'static>>();
    assert_send_sync::<SongSetHeader>();
    assert_send_sync::<InstrumentEntry>();
    assert_send_sync::<PercussionEntry>();
    assert_send_sync::<HowlCommand>();
    assert_send_sync::<HowlEvent>();
    assert_send_sync::<SongSetView<'static>>();
    assert_send_sync::<SequenceView<'static>>();
    assert_send_sync::<TrackView<'static>>();
    assert_send_sync::<TrackEventIter<'static>>();
    assert_send_sync::<ReverbPreset>();
    assert_send_sync::<ReverbParams>();
    #[cfg(feature = "alloc")]
    {
        assert_send_sync::<HowlArchive>();
        assert_send_sync::<SongSet>();
        assert_send_sync::<Sequence>();
        assert_send_sync::<Track>();
        assert_send_sync::<RenderConfig>();
        assert_send_sync::<RenderResult>();
        assert_send_sync::<SpuUploadSummary>();
        assert_send_sync::<TrackSnapshot>();
        assert_send_sync::<HowlPlayer>();
    }
};
