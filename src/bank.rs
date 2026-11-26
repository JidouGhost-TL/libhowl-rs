// SPDX-License-Identifier: MIT OR Apache-2.0

//! `HOWL` audio bank container parser (header, SPU sample directory, SFX tables,
//! sector-aligned ADPCM sample banks, and song-set index).

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::sequence::SongSetView;
use crate::{Error, Result};

/// 4-byte ASCII magic signature at offset `0x00` of a `HOWL` archive.
pub const HOWL_MAGIC: [u8; 4] = *b"HOWL";

/// Fixed byte size of the `HOWL` primary header (`0x28` bytes).
pub const HOWL_HEADER_SIZE: usize = 0x28;

/// CD-ROM sector alignment unit (`0x800` = 2048 bytes) used by `HOWL` bank and song-set offsets.
pub const HOWL_SECTOR_SIZE: usize = 0x800;

/// Byte stride of each SPU sample table entry (`4` bytes).
pub const SPU_ENTRY_SIZE: usize = 4;

/// Byte stride of each standard SFX descriptor (`8` bytes).
pub const SFX_ENTRY_SIZE: usize = 8;

/// Byte stride of each continuous-pitch SFX descriptor (`8` bytes).
pub const ENGINE_SFX_ENTRY_SIZE: usize = 8;

/// Upper bound on SPU sample entries to keep detection bounded and overflow-free.
const MAX_SPU_ENTRIES: u32 = 8192;

/// Upper bound on bank or song-set counts.
const MAX_DIR_ENTRIES: u32 = 4096;

/// Parsed 40-byte (`0x28`) `HOWL` archive header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HowlHeader {
    /// Format revision number (e.g. `111`, `114`, `128`).
    pub version: u32,
    /// Number of 4-byte SPU sample directory entries at `+0x28`.
    pub spu_entry_count: u32,
    /// Number of 8-byte standard SFX entries.
    pub sfx_count: u32,
    /// Number of 8-byte wide-pitch SFX entries.
    pub engine_sfx_count: u32,
    /// Number of `u16` sector offsets in the ADPCM sample-bank directory.
    pub bank_count: u32,
    /// Number of `u16` sector offsets in the song-set directory.
    pub song_set_count: u32,
    /// Exact total byte length of all five header tables following `+0x28`.
    pub header_table_bytes: u32,
}

/// 4-byte SPU sample table entry at `+0x28 + index * 4`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpuSampleEntry {
    /// Runtime SPU RAM address divided by 8 (`0` on disk until uploaded).
    pub spu_addr_div8: u16,
    /// ADPCM sample byte length divided by 8.
    pub size_div8: u16,
}

impl SpuSampleEntry {
    /// Returns the ADPCM sample length in bytes (`size_div8 * 8`).
    #[inline]
    #[must_use]
    pub const fn byte_len(self) -> usize {
        (self.size_div8 as usize) * 8
    }
}

/// 8-byte standard sound-effect descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SfxEntry {
    /// SFX flags (`bit 2`: voice/dialogue volume group).
    pub flags: u8,
    /// Base volume (`0..=255`).
    pub volume: u8,
    /// Base PS1 SPU pitch (`0x1000` = unity 44.1 kHz).
    pub base_pitch: u16,
    /// Index into the archive's SPU sample table.
    pub spu_index: u16,
    /// Duration parameter (in driver ticks).
    pub duration: u16,
}

/// 8-byte wide-pitch (`+/-2` octave) sound-effect descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EngineSfxEntry {
    /// SFX flags.
    pub flags: u8,
    /// Base volume (`0..=255`).
    pub volume: u8,
    /// Base PS1 SPU pitch (`0x1000` = unity 44.1 kHz).
    pub base_pitch: u16,
    /// Reserved / auxiliary parameter.
    pub reserved: u16,
    /// Index into the archive's SPU sample table.
    pub spu_index: u16,
}

/// Zero-copy borrowed view over one sector-aligned ADPCM sample bank inside a `HOWL` archive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleBankView<'a> {
    sector: u16,
    sample_count: u16,
    id_bytes: &'a [u8],
    adpcm_data: &'a [u8],
}

impl<'a> SampleBankView<'a> {
    /// Returns the starting 2048-byte sector index of this bank.
    #[inline]
    #[must_use]
    pub const fn sector(&self) -> u16 {
        self.sector
    }

    /// Returns the number of ADPCM samples packed in this bank.
    #[inline]
    #[must_use]
    pub const fn sample_count(&self) -> usize {
        self.sample_count as usize
    }

    /// Returns the SPU sample ID at slot `index` within this bank, if in range.
    #[must_use]
    pub fn sample_id(&self, index: usize) -> Option<u16> {
        let off = index.checked_mul(2)?;
        if off + 2 > self.id_bytes.len() {
            return None;
        }
        Some(u16::from_le_bytes([
            self.id_bytes[off],
            self.id_bytes[off + 1],
        ]))
    }

    /// Returns the concatenated PS1 SPU-ADPCM payload slice of this bank
    /// (located at `(sector + 1) * 0x800`).
    #[inline]
    #[must_use]
    pub const fn adpcm_data(&self) -> &'a [u8] {
        self.adpcm_data
    }

    /// Finds the ADPCM byte slice for a given `spu_index` inside this bank, if present.
    #[must_use]
    pub fn find_sample(&self, archive: &HowlArchiveView<'a>, spu_index: u16) -> Option<&'a [u8]> {
        let mut cursor = 0usize;
        for slot in 0..self.sample_count() {
            let id = self.sample_id(slot)?;
            let len = archive.spu_entry(usize::from(id))?.byte_len();
            let end = cursor.checked_add(len)?;
            if end > self.adpcm_data.len() {
                return None;
            }
            if id == spu_index {
                return Some(&self.adpcm_data[cursor..end]);
            }
            cursor = end;
        }
        None
    }
}

/// Zero-copy borrowed view over an entire `HOWL` archive (`.HWL`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HowlArchiveView<'a> {
    data: &'a [u8],
    header: HowlHeader,
    spu_table_off: usize,
    sfx_table_off: usize,
    engine_sfx_table_off: usize,
    bank_table_off: usize,
    song_table_off: usize,
}

impl<'a> HowlArchiveView<'a> {
    /// Fast, panic-free structural format detector.
    ///
    /// Returns `true` only when `data` passes full header, table-size,
    /// sample-bank, and song-set structural validation.
    #[must_use]
    pub fn detect(data: &[u8]) -> bool {
        HowlArchiveView::parse(data).is_ok()
    }

    /// Parses and structurally validates a `HOWL` archive from `data`.
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        if data.len() < HOWL_HEADER_SIZE {
            return Err(Error::UnexpectedEof("HOWL header"));
        }
        if data[0..4] != HOWL_MAGIC {
            return Err(Error::InvalidMagic);
        }

        let version = read_u32_le(data, 0x04)?;
        let reserved0 = read_u32_le(data, 0x08)?;
        let reserved1 = read_u32_le(data, 0x0C)?;
        let spu_entry_count = read_u32_le(data, 0x10)?;
        let sfx_count = read_u32_le(data, 0x14)?;
        let engine_sfx_count = read_u32_le(data, 0x18)?;
        let bank_count = read_u32_le(data, 0x1C)?;
        let song_set_count = read_u32_le(data, 0x20)?;
        let header_table_bytes = read_u32_le(data, 0x24)?;

        if version == 0 || version > 0xFFFF {
            return Err(Error::InvalidHeader("HOWL version out of range"));
        }
        if reserved0 != 0 || reserved1 != 0 {
            return Err(Error::InvalidHeader("HOWL reserved words must be zero"));
        }
        if spu_entry_count == 0 || spu_entry_count > MAX_SPU_ENTRIES {
            return Err(Error::InvalidHeader("SPU entry count out of range"));
        }
        if bank_count > MAX_DIR_ENTRIES || song_set_count > MAX_DIR_ENTRIES {
            return Err(Error::InvalidHeader("directory entry count out of range"));
        }
        if bank_count == 0 && song_set_count == 0 {
            return Err(Error::InvalidHeader("archive has no banks or song sets"));
        }

        let spu_bytes = (spu_entry_count as usize)
            .checked_mul(SPU_ENTRY_SIZE)
            .ok_or(Error::OutOfRange("SPU table size"))?;
        let sfx_bytes = (sfx_count as usize)
            .checked_mul(SFX_ENTRY_SIZE)
            .ok_or(Error::OutOfRange("SFX table size"))?;
        let engine_bytes = (engine_sfx_count as usize)
            .checked_mul(ENGINE_SFX_ENTRY_SIZE)
            .ok_or(Error::OutOfRange("engine SFX table size"))?;
        let bank_bytes = (bank_count as usize)
            .checked_mul(2)
            .ok_or(Error::OutOfRange("bank sector table size"))?;
        let song_bytes = (song_set_count as usize)
            .checked_mul(2)
            .ok_or(Error::OutOfRange("song sector table size"))?;

        let spu_table_off = HOWL_HEADER_SIZE;
        let sfx_table_off = spu_table_off
            .checked_add(spu_bytes)
            .ok_or(Error::OutOfRange("SFX table offset"))?;
        let engine_sfx_table_off = sfx_table_off
            .checked_add(sfx_bytes)
            .ok_or(Error::OutOfRange("engine SFX table offset"))?;
        let bank_table_off = engine_sfx_table_off
            .checked_add(engine_bytes)
            .ok_or(Error::OutOfRange("bank table offset"))?;
        let song_table_off = bank_table_off
            .checked_add(bank_bytes)
            .ok_or(Error::OutOfRange("song table offset"))?;
        let tables_end = song_table_off
            .checked_add(song_bytes)
            .ok_or(Error::OutOfRange("header tables end"))?;

        let computed_table_bytes = tables_end - HOWL_HEADER_SIZE;
        if computed_table_bytes != header_table_bytes as usize {
            return Err(Error::InvalidHeader("header table byte count mismatch"));
        }
        if tables_end > data.len() {
            return Err(Error::UnexpectedEof("HOWL header tables"));
        }

        let view = Self {
            data,
            header: HowlHeader {
                version,
                spu_entry_count,
                sfx_count,
                engine_sfx_count,
                bank_count,
                song_set_count,
                header_table_bytes,
            },
            spu_table_off,
            sfx_table_off,
            engine_sfx_table_off,
            bank_table_off,
            song_table_off,
        };

        // Validate every bank and song set structurally.
        for bi in 0..bank_count as usize {
            let _ = view.bank(bi)?;
        }
        for si in 0..song_set_count as usize {
            let _ = view.song_set(si)?;
        }

        Ok(view)
    }

    /// Returns the parsed [`HowlHeader`].
    #[inline]
    #[must_use]
    pub const fn header(&self) -> HowlHeader {
        self.header
    }

    /// Returns the full raw archive byte slice.
    #[inline]
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.data
    }

    /// Returns SPU sample directory entry `index`, if in range.
    #[must_use]
    pub fn spu_entry(&self, index: usize) -> Option<SpuSampleEntry> {
        if index >= self.header.spu_entry_count as usize {
            return None;
        }
        let off = self.spu_table_off + index * SPU_ENTRY_SIZE;
        Some(SpuSampleEntry {
            spu_addr_div8: u16::from_le_bytes([self.data[off], self.data[off + 1]]),
            size_div8: u16::from_le_bytes([self.data[off + 2], self.data[off + 3]]),
        })
    }

    /// Returns standard SFX descriptor `index`, if in range.
    #[must_use]
    pub fn sfx_entry(&self, index: usize) -> Option<SfxEntry> {
        if index >= self.header.sfx_count as usize {
            return None;
        }
        let off = self.sfx_table_off + index * SFX_ENTRY_SIZE;
        Some(SfxEntry {
            flags: self.data[off],
            volume: self.data[off + 1],
            base_pitch: u16::from_le_bytes([self.data[off + 2], self.data[off + 3]]),
            spu_index: u16::from_le_bytes([self.data[off + 4], self.data[off + 5]]),
            duration: u16::from_le_bytes([self.data[off + 6], self.data[off + 7]]),
        })
    }

    /// Returns wide-pitch SFX descriptor `index`, if in range.
    #[must_use]
    pub fn engine_sfx_entry(&self, index: usize) -> Option<EngineSfxEntry> {
        if index >= self.header.engine_sfx_count as usize {
            return None;
        }
        let off = self.engine_sfx_table_off + index * ENGINE_SFX_ENTRY_SIZE;
        Some(EngineSfxEntry {
            flags: self.data[off],
            volume: self.data[off + 1],
            base_pitch: u16::from_le_bytes([self.data[off + 2], self.data[off + 3]]),
            reserved: u16::from_le_bytes([self.data[off + 4], self.data[off + 5]]),
            spu_index: u16::from_le_bytes([self.data[off + 6], self.data[off + 7]]),
        })
    }

    /// Returns the starting 2048-byte sector of sample bank `index`, if in range.
    #[must_use]
    pub fn bank_sector(&self, index: usize) -> Option<u16> {
        if index >= self.header.bank_count as usize {
            return None;
        }
        let off = self.bank_table_off + index * 2;
        Some(u16::from_le_bytes([self.data[off], self.data[off + 1]]))
    }

    /// Returns the starting 2048-byte sector of song set `index`, if in range.
    #[must_use]
    pub fn song_set_sector(&self, index: usize) -> Option<u16> {
        if index >= self.header.song_set_count as usize {
            return None;
        }
        let off = self.song_table_off + index * 2;
        Some(u16::from_le_bytes([self.data[off], self.data[off + 1]]))
    }

    /// Returns a validated [`SampleBankView`] for bank `index`.
    pub fn bank(&self, index: usize) -> Result<SampleBankView<'a>> {
        let sector = self
            .bank_sector(index)
            .ok_or(Error::OutOfRange("bank index"))?;
        let base = usize::from(sector)
            .checked_mul(HOWL_SECTOR_SIZE)
            .ok_or(Error::OutOfRange("bank byte offset"))?;
        let adpcm_start = base
            .checked_add(HOWL_SECTOR_SIZE)
            .ok_or(Error::OutOfRange("bank ADPCM start"))?;
        if base + 2 > self.data.len() {
            return Err(Error::UnexpectedEof("bank header"));
        }
        let sample_count = u16::from_le_bytes([self.data[base], self.data[base + 1]]);
        let id_bytes_len = usize::from(sample_count) * 2;
        if 2 + id_bytes_len > HOWL_SECTOR_SIZE || base + 2 + id_bytes_len > self.data.len() {
            return Err(Error::InvalidHeader("bank sample ID table exceeds sector"));
        }
        let id_bytes = &self.data[base + 2..base + 2 + id_bytes_len];

        let mut total_adpcm = 0usize;
        for slot in 0..usize::from(sample_count) {
            let sid = u16::from_le_bytes([id_bytes[slot * 2], id_bytes[slot * 2 + 1]]);
            let entry = self
                .spu_entry(usize::from(sid))
                .ok_or(Error::OutOfRange("bank sample ID out of range"))?;
            total_adpcm = total_adpcm
                .checked_add(entry.byte_len())
                .ok_or(Error::OutOfRange("bank ADPCM payload size"))?;
        }
        let adpcm_end = adpcm_start
            .checked_add(total_adpcm)
            .ok_or(Error::OutOfRange("bank ADPCM payload end"))?;
        if adpcm_end > self.data.len() {
            return Err(Error::UnexpectedEof("bank ADPCM payload"));
        }

        Ok(SampleBankView {
            sector,
            sample_count,
            id_bytes,
            adpcm_data: &self.data[adpcm_start..adpcm_end],
        })
    }

    /// Returns a validated [`SongSetView`] for song set `index`.
    pub fn song_set(&self, index: usize) -> Result<SongSetView<'a>> {
        let sector = self
            .song_set_sector(index)
            .ok_or(Error::OutOfRange("song set index"))?;
        let base = usize::from(sector)
            .checked_mul(HOWL_SECTOR_SIZE)
            .ok_or(Error::OutOfRange("song set byte offset"))?;
        if base >= self.data.len() {
            return Err(Error::UnexpectedEof("song set offset"));
        }
        SongSetView::parse(&self.data[base..])
    }

    /// Searches all sample banks in this archive for the ADPCM payload of `spu_index`.
    #[must_use]
    pub fn find_sample_adpcm(&self, spu_index: u16) -> Option<&'a [u8]> {
        for bi in 0..self.header.bank_count as usize {
            if let Ok(bank) = self.bank(bi) {
                if let Some(slice) = bank.find_sample(self, spu_index) {
                    return Some(slice);
                }
            }
        }
        None
    }

    /// Copies the raw archive into an owned [`HowlArchive`].
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn to_owned(&self) -> HowlArchive {
        HowlArchive {
            raw: self.data.to_vec(),
            header: self.header,
        }
    }
}

/// Owned, validated `HOWL` archive buffer.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HowlArchive {
    raw: Vec<u8>,
    header: HowlHeader,
}

#[cfg(feature = "alloc")]
impl HowlArchive {
    /// Parses and validates a `HOWL` archive, copying its bytes into an owned buffer.
    pub fn parse(data: &[u8]) -> Result<Self> {
        let view = HowlArchiveView::parse(data)?;
        Ok(view.to_owned())
    }

    /// Wraps an owned byte vector after validating it as a `HOWL` archive.
    pub fn from_vec(raw: Vec<u8>) -> Result<Self> {
        let header = HowlArchiveView::parse(&raw)?.header();
        Ok(Self { raw, header })
    }

    /// Returns the parsed [`HowlHeader`].
    #[inline]
    #[must_use]
    pub const fn header(&self) -> HowlHeader {
        self.header
    }

    /// Borrows a validated [`HowlArchiveView`] over this owned archive.
    #[must_use]
    pub fn view(&self) -> HowlArchiveView<'_> {
        // Safe: `raw` was already validated at construction time.
        match HowlArchiveView::parse(&self.raw) {
            Ok(v) => v,
            Err(_) => unreachable!("validated HowlArchive invariant"),
        }
    }

    /// Returns the underlying raw bytes of the archive.
    #[inline]
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.raw
    }
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
