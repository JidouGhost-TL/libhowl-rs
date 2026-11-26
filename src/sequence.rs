// SPDX-License-Identifier: MIT OR Apache-2.0

//! `HOWL` sequence-set, sequence, track, and bytecode parser.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::{Error, Result};

/// Fixed byte size of a song-set header (`size`, `instrument_count`, `percussion_count`, `sequence_count`).
pub const SONG_SET_HEADER_SIZE: usize = 8;

/// Byte stride of each melodic instrument descriptor inside a song set.
pub const INSTRUMENT_ENTRY_SIZE: usize = 12;

/// Byte stride of each percussion descriptor inside a song set.
pub const PERCUSSION_ENTRY_SIZE: usize = 8;

/// Fixed byte size of a sequence header (`flags`, `track_count`, `bpm`, `tpqn`).
pub const SEQUENCE_HEADER_SIZE: usize = 6;

/// Track flag bit indicating a percussion (drum) track rather than a melodic track.
pub const TRACK_FLAG_PERCUSSION: u8 = 0x01;

/// Maximum number of bytes accepted in a single VLQ delta-time value.
const MAX_VLQ_BYTES: usize = 5;

/// Header of a `HOWL` song set (`SongSet`), located at a sector offset listed in
/// [`HowlArchiveView::song_set_sector`](crate::HowlArchiveView::song_set_sector).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SongSetHeader {
    /// Total byte size of this song-set payload (including this 8-byte header).
    pub size: u32,
    /// Number of 12-byte melodic instrument descriptors.
    pub instrument_count: u8,
    /// Number of 8-byte percussion descriptors.
    pub percussion_count: u8,
    /// Number of sequences contained in this song set.
    pub sequence_count: u16,
}

/// 12-byte melodic instrument descriptor inside a `HOWL` song set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstrumentEntry {
    /// Instrument flags.
    pub flags: u8,
    /// Base instrument volume (`0..=255`).
    pub volume: u8,
    /// Reserved / alignment field.
    pub reserved: u16,
    /// Base PS1 SPU pitch (`0x1000` = unity 44.1 kHz at MIDI note 60).
    pub base_pitch: u16,
    /// Index into the parent `HOWL` archive's SPU sample table.
    pub spu_index: u16,
    /// Raw PS1 SPU `VP_ADSR1` register value.
    pub adsr1: u16,
    /// Raw PS1 SPU `VP_ADSR2` register value.
    pub adsr2: u16,
}

/// 8-byte percussion (drum) descriptor inside a `HOWL` song set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PercussionEntry {
    /// Percussion flags.
    pub flags: u8,
    /// Base percussion volume (`0..=255`).
    pub volume: u8,
    /// Base PS1 SPU pitch (`0x1000` = unity 44.1 kHz).
    pub base_pitch: u16,
    /// Index into the parent `HOWL` archive's SPU sample table.
    pub spu_index: u16,
    /// Reserved / alignment field.
    pub reserved: u16,
}

/// Decoded `HOWL` track bytecode command (`0x00..=0x0A`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HowlCommand {
    /// `0x00` (1 byte): No-operation.
    Nop,
    /// `0x01` (2 bytes): Release active voice(s) on this track matching `note`.
    NoteOff {
        /// Note / percussion index (`0..=255`).
        note: u8,
    },
    /// `0x02` (2 bytes): Reserved 1-operand no-operation.
    Reserved2(u8),
    /// `0x03` (1 byte): End of track (loops if the track's loop mode is active, otherwise stops song).
    EndOfTrack,
    /// `0x04` (2 bytes): Reserved 1-operand no-operation.
    Reserved4(u8),
    /// `0x05` (3 bytes): Trigger a note on this track.
    NoteOn {
        /// Note number (melodic) or percussion table index (percussion track).
        note: u8,
        /// Note velocity (`0..=255`).
        velocity: u8,
    },
    /// `0x06` (2 bytes): Set track volume (`0..=255`) and update active voices.
    SetVolume(u8),
    /// `0x07` (2 bytes): Set track stereo pan (`0..=255`, `0x80` = center) and update active voices.
    SetPan(u8),
    /// `0x08` (2 bytes): Set hardware reverb send (`0` = off, non-zero = on) on active voices.
    SetReverb(u8),
    /// `0x09` (2 bytes): Select melodic instrument index for subsequent `NoteOn` events.
    SetInstrument(u8),
    /// `0x0A` (2 bytes): Set track pitch bend (`0..=255`, `0x80` = center) and update active voices.
    SetPitchBend(u8),
}

impl HowlCommand {
    /// Returns the total encoded byte length (opcode + operands) of a `HOWL` opcode,
    /// or `None` if `opcode > 0x0A`.
    #[inline]
    #[must_use]
    pub const fn encoded_len(opcode: u8) -> Option<usize> {
        match opcode {
            0x00 | 0x03 => Some(1),
            0x01 | 0x02 | 0x04 | 0x06 | 0x07 | 0x08 | 0x09 | 0x0A => Some(2),
            0x05 => Some(3),
            _ => None,
        }
    }

    /// Decodes a single command starting at `bytes[0]`, returning `(command, bytes_consumed)`.
    pub fn decode(bytes: &[u8]) -> Result<(Self, usize)> {
        let &op = bytes
            .first()
            .ok_or(Error::UnexpectedEof("track command opcode"))?;
        let len = Self::encoded_len(op).ok_or(Error::InvalidHeader("unknown HOWL opcode"))?;
        if bytes.len() < len {
            return Err(Error::UnexpectedEof("track command operand"));
        }
        let cmd = match op {
            0x00 => Self::Nop,
            0x01 => Self::NoteOff { note: bytes[1] },
            0x02 => Self::Reserved2(bytes[1]),
            0x03 => Self::EndOfTrack,
            0x04 => Self::Reserved4(bytes[1]),
            0x05 => Self::NoteOn {
                note: bytes[1],
                velocity: bytes[2],
            },
            0x06 => Self::SetVolume(bytes[1]),
            0x07 => Self::SetPan(bytes[1]),
            0x08 => Self::SetReverb(bytes[1]),
            0x09 => Self::SetInstrument(bytes[1]),
            0x0A => Self::SetPitchBend(bytes[1]),
            _ => return Err(Error::InvalidHeader("unknown HOWL opcode")),
        };
        Ok((cmd, len))
    }
}

/// A timed `HOWL` track event (`delta` ticks before executing `command`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HowlEvent {
    /// Delta ticks to wait before executing `command`.
    pub delta: u32,
    /// Command to execute.
    pub command: HowlCommand,
}

/// Reads a big-endian MIDI-style 7-bit variable-length quantity (VLQ) from `data`.
///
/// Returns `(value, bytes_consumed)`.
pub fn read_vlq(data: &[u8]) -> Result<(u32, usize)> {
    let mut val: u32 = 0;
    for (i, &b) in data.iter().enumerate() {
        if i >= MAX_VLQ_BYTES {
            return Err(Error::OutOfRange("VLQ delta overflow"));
        }
        val = val
            .checked_mul(0x80)
            .and_then(|v| v.checked_add(u32::from(b & 0x7F)))
            .ok_or(Error::OutOfRange("VLQ delta overflow"))?;
        if (b & 0x80) == 0 {
            return Ok((val, i + 1));
        }
    }
    Err(Error::UnexpectedEof("VLQ delta"))
}

/// Zero-copy borrowed view over a `HOWL` song set (`SongSet`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SongSetView<'a> {
    data: &'a [u8],
    header: SongSetHeader,
    inst_offset: usize,
    drum_offset: usize,
    seq_table_offset: usize,
    seq_data_offset: usize,
}

impl<'a> SongSetView<'a> {
    /// Parses a `HOWL` song set from `data` (starting at the song-set header).
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        if data.len() < SONG_SET_HEADER_SIZE {
            return Err(Error::UnexpectedEof("song set header"));
        }
        let size = read_u32_le(data, 0)?;
        let instrument_count = data[4];
        let percussion_count = data[5];
        let sequence_count = read_u16_le(data, 6)?;

        let size_usize = size as usize;
        if size_usize < SONG_SET_HEADER_SIZE || size_usize > data.len() {
            return Err(Error::InvalidHeader("song set size out of bounds"));
        }
        let slice = &data[..size_usize];

        let inst_offset = SONG_SET_HEADER_SIZE;
        let inst_bytes = usize::from(instrument_count)
            .checked_mul(INSTRUMENT_ENTRY_SIZE)
            .ok_or(Error::OutOfRange("instrument table size"))?;
        let drum_offset = inst_offset
            .checked_add(inst_bytes)
            .ok_or(Error::OutOfRange("percussion table offset"))?;
        let drum_bytes = usize::from(percussion_count)
            .checked_mul(PERCUSSION_ENTRY_SIZE)
            .ok_or(Error::OutOfRange("percussion table size"))?;
        let seq_table_offset = drum_offset
            .checked_add(drum_bytes)
            .ok_or(Error::OutOfRange("sequence offset table"))?;
        let seq_table_bytes = usize::from(sequence_count)
            .checked_mul(2)
            .ok_or(Error::OutOfRange("sequence offset table size"))?;
        let seq_table_end = seq_table_offset
            .checked_add(seq_table_bytes)
            .ok_or(Error::OutOfRange("sequence offset table end"))?;
        let seq_data_offset = align4(seq_table_end)?;
        if seq_data_offset > slice.len() {
            return Err(Error::UnexpectedEof("song set tables"));
        }

        let view = Self {
            data: slice,
            header: SongSetHeader {
                size,
                instrument_count,
                percussion_count,
                sequence_count,
            },
            inst_offset,
            drum_offset,
            seq_table_offset,
            seq_data_offset,
        };

        // Validate every sequence header fits inside the song set.
        for i in 0..usize::from(sequence_count) {
            let _ = view.sequence(i)?;
        }

        Ok(view)
    }

    /// Returns the parsed [`SongSetHeader`].
    #[inline]
    #[must_use]
    pub const fn header(&self) -> SongSetHeader {
        self.header
    }

    /// Returns the raw byte slice of this song set.
    #[inline]
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.data
    }

    /// Returns melodic instrument descriptor `index`, if in range.
    #[must_use]
    pub fn instrument(&self, index: usize) -> Option<InstrumentEntry> {
        if index >= usize::from(self.header.instrument_count) {
            return None;
        }
        let off = self.inst_offset + index * INSTRUMENT_ENTRY_SIZE;
        Some(InstrumentEntry {
            flags: self.data[off],
            volume: self.data[off + 1],
            reserved: u16::from_le_bytes([self.data[off + 2], self.data[off + 3]]),
            base_pitch: u16::from_le_bytes([self.data[off + 4], self.data[off + 5]]),
            spu_index: u16::from_le_bytes([self.data[off + 6], self.data[off + 7]]),
            adsr1: u16::from_le_bytes([self.data[off + 8], self.data[off + 9]]),
            adsr2: u16::from_le_bytes([self.data[off + 10], self.data[off + 11]]),
        })
    }

    /// Returns percussion descriptor `index`, if in range.
    #[must_use]
    pub fn percussion(&self, index: usize) -> Option<PercussionEntry> {
        if index >= usize::from(self.header.percussion_count) {
            return None;
        }
        let off = self.drum_offset + index * PERCUSSION_ENTRY_SIZE;
        Some(PercussionEntry {
            flags: self.data[off],
            volume: self.data[off + 1],
            base_pitch: u16::from_le_bytes([self.data[off + 2], self.data[off + 3]]),
            spu_index: u16::from_le_bytes([self.data[off + 4], self.data[off + 5]]),
            reserved: u16::from_le_bytes([self.data[off + 6], self.data[off + 7]]),
        })
    }

    /// Returns sequence `index` inside this song set.
    pub fn sequence(&self, index: usize) -> Result<SequenceView<'a>> {
        if index >= usize::from(self.header.sequence_count) {
            return Err(Error::OutOfRange("sequence index"));
        }
        let rel = read_u16_le(self.data, self.seq_table_offset + index * 2)?;
        let start = self
            .seq_data_offset
            .checked_add(usize::from(rel))
            .ok_or(Error::OutOfRange("sequence start offset"))?;
        if start >= self.data.len() {
            return Err(Error::OutOfRange("sequence start out of bounds"));
        }
        SequenceView::parse(&self.data[start..])
    }

    /// Converts this borrowed song-set view into an owned [`SongSet`].
    #[cfg(feature = "alloc")]
    pub fn to_owned(&self) -> Result<SongSet> {
        let mut instruments = Vec::with_capacity(usize::from(self.header.instrument_count));
        for i in 0..usize::from(self.header.instrument_count) {
            if let Some(inst) = self.instrument(i) {
                instruments.push(inst);
            }
        }
        let mut percussion = Vec::with_capacity(usize::from(self.header.percussion_count));
        for i in 0..usize::from(self.header.percussion_count) {
            if let Some(drum) = self.percussion(i) {
                percussion.push(drum);
            }
        }
        let mut sequences = Vec::with_capacity(usize::from(self.header.sequence_count));
        for i in 0..usize::from(self.header.sequence_count) {
            sequences.push(self.sequence(i)?.to_owned()?);
        }
        Ok(SongSet {
            header: self.header,
            instruments,
            percussion,
            sequences,
        })
    }
}

/// Zero-copy borrowed view over a single `HOWL` sequence (`CSeq`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceView<'a> {
    data: &'a [u8],
    /// Sequence flags byte.
    pub flags: u8,
    /// Number of tracks in this sequence.
    pub track_count: u8,
    /// Tempo in beats per minute (BPM).
    pub bpm: u16,
    /// Ticks per quarter note (TPQN).
    pub tpqn: u16,
    track_data_offset: usize,
}

impl<'a> SequenceView<'a> {
    /// Parses a sequence header and validates its track offset table.
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        if data.len() < SEQUENCE_HEADER_SIZE {
            return Err(Error::UnexpectedEof("sequence header"));
        }
        let flags = data[0];
        let track_count = data[1];
        let bpm = read_u16_le(data, 2)?;
        let tpqn = read_u16_le(data, 4)?;

        let table_end = SEQUENCE_HEADER_SIZE
            .checked_add(usize::from(track_count) * 2)
            .ok_or(Error::OutOfRange("track offset table"))?;
        let track_data_offset = align4(table_end)?;
        if track_data_offset > data.len() {
            return Err(Error::UnexpectedEof("sequence track offset table"));
        }

        let seq = Self {
            data,
            flags,
            track_count,
            bpm,
            tpqn,
            track_data_offset,
        };
        for i in 0..usize::from(track_count) {
            let _ = seq.track(i)?;
        }
        Ok(seq)
    }

    /// Returns track `index` inside this sequence.
    pub fn track(&self, index: usize) -> Result<TrackView<'a>> {
        if index >= usize::from(self.track_count) {
            return Err(Error::OutOfRange("track index"));
        }
        let rel = read_u16_le(self.data, SEQUENCE_HEADER_SIZE + index * 2)?;
        let start = self
            .track_data_offset
            .checked_add(usize::from(rel))
            .ok_or(Error::OutOfRange("track start offset"))?;
        if start + 2 > self.data.len() {
            return Err(Error::UnexpectedEof("track header"));
        }
        Ok(TrackView {
            flags: self.data[start],
            id: self.data[start + 1],
            stream: &self.data[start + 2..],
        })
    }

    /// Converts this borrowed sequence view into an owned [`Sequence`].
    #[cfg(feature = "alloc")]
    pub fn to_owned(&self) -> Result<Sequence> {
        let mut tracks = Vec::with_capacity(usize::from(self.track_count));
        for i in 0..usize::from(self.track_count) {
            tracks.push(self.track(i)?.to_owned()?);
        }
        Ok(Sequence {
            flags: self.flags,
            bpm: self.bpm,
            tpqn: self.tpqn,
            tracks,
        })
    }
}

/// Zero-copy borrowed view over a single `HOWL` sequence track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrackView<'a> {
    /// Track flags byte (`bit 0` = percussion track).
    pub flags: u8,
    /// Track identifier byte.
    pub id: u8,
    /// Raw bytecode stream starting at the lead VLQ delta.
    stream: &'a [u8],
}

impl<'a> TrackView<'a> {
    /// Returns `true` if this track uses the song set's percussion table instead
    /// of the melodic instrument table.
    #[inline]
    #[must_use]
    pub const fn is_percussion(&self) -> bool {
        (self.flags & TRACK_FLAG_PERCUSSION) != 0
    }

    /// Returns the raw bytecode slice starting at the lead VLQ delta.
    #[inline]
    #[must_use]
    pub const fn raw_stream(&self) -> &'a [u8] {
        self.stream
    }

    /// Returns a lazy zero-copy iterator over all events in this track up to
    /// (and including) [`HowlCommand::EndOfTrack`].
    #[inline]
    #[must_use]
    pub const fn events(&self) -> TrackEventIter<'a> {
        TrackEventIter {
            stream: self.stream,
            pos: 0,
            finished: false,
        }
    }

    /// Decodes all events in this track into an owned [`Track`].
    #[cfg(feature = "alloc")]
    pub fn to_owned(&self) -> Result<Track> {
        let mut events = Vec::new();
        for item in self.events() {
            events.push(item?);
        }
        Ok(Track {
            flags: self.flags,
            id: self.id,
            events,
        })
    }
}

/// Zero-copy iterator yielding [`HowlEvent`]s from a [`TrackView`].
#[derive(Debug, Clone, Copy)]
pub struct TrackEventIter<'a> {
    stream: &'a [u8],
    pos: usize,
    finished: bool,
}

impl Iterator for TrackEventIter<'_> {
    type Item = Result<HowlEvent>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        if self.pos >= self.stream.len() {
            self.finished = true;
            return None;
        }
        let (delta, vlq_len) = match read_vlq(&self.stream[self.pos..]) {
            Ok(pair) => pair,
            Err(err) => {
                self.finished = true;
                return Some(Err(err));
            }
        };
        self.pos += vlq_len;
        let (command, cmd_len) = match HowlCommand::decode(&self.stream[self.pos..]) {
            Ok(pair) => pair,
            Err(err) => {
                self.finished = true;
                return Some(Err(err));
            }
        };
        self.pos += cmd_len;
        if matches!(command, HowlCommand::EndOfTrack) {
            self.finished = true;
        }
        Some(Ok(HowlEvent { delta, command }))
    }
}

/// Owned decoded `HOWL` track.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Track {
    /// Track flags (`bit 0` = percussion track).
    pub flags: u8,
    /// Track identifier byte.
    pub id: u8,
    /// Decoded sequence of timed events.
    pub events: Vec<HowlEvent>,
}

#[cfg(feature = "alloc")]
impl Track {
    /// Returns `true` if this track uses the percussion table.
    #[inline]
    #[must_use]
    pub const fn is_percussion(&self) -> bool {
        (self.flags & TRACK_FLAG_PERCUSSION) != 0
    }
}

/// Owned decoded `HOWL` sequence (`CSeq`).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sequence {
    /// Sequence flags byte.
    pub flags: u8,
    /// Tempo in beats per minute (BPM).
    pub bpm: u16,
    /// Ticks per quarter note (TPQN).
    pub tpqn: u16,
    /// Decoded tracks in this sequence.
    pub tracks: Vec<Track>,
}

/// Owned decoded `HOWL` song set (`SongSet`).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SongSet {
    /// Parsed song-set header.
    pub header: SongSetHeader,
    /// Melodic instrument descriptors (`0x0C` bytes each).
    pub instruments: Vec<InstrumentEntry>,
    /// Percussion descriptors (`0x08` bytes each).
    pub percussion: Vec<PercussionEntry>,
    /// Sequences belonging to this song set.
    pub sequences: Vec<Sequence>,
}

#[cfg(feature = "alloc")]
impl SongSet {
    /// Parses a song set directly into an owned [`SongSet`].
    pub fn parse(data: &[u8]) -> Result<Self> {
        SongSetView::parse(data)?.to_owned()
    }
}

#[inline]
fn align4(val: usize) -> Result<usize> {
    val.checked_add(3)
        .map(|v| v & !3)
        .ok_or(Error::OutOfRange("4-byte alignment overflow"))
}

#[inline]
fn read_u16_le(data: &[u8], off: usize) -> Result<u16> {
    let end = off
        .checked_add(2)
        .ok_or(Error::OutOfRange("u16 read offset"))?;
    if end > data.len() {
        return Err(Error::UnexpectedEof("u16 field"));
    }
    Ok(u16::from_le_bytes([data[off], data[off + 1]]))
}

#[inline]
fn read_u32_le(data: &[u8], off: usize) -> Result<u32> {
    let end = off
        .checked_add(4)
        .ok_or(Error::OutOfRange("u32 read offset"))?;
    if end > data.len() {
        return Err(Error::UnexpectedEof("u32 field"));
    }
    Ok(u32::from_le_bytes([
        data[off],
        data[off + 1],
        data[off + 2],
        data[off + 3],
    ]))
}
