// SPDX-License-Identifier: MIT OR Apache-2.0

//! Audio rendering throughput benchmark for `libhowl`.

use std::hint::black_box;
use std::time::Instant;

use libhowl::{HowlArchiveView, HowlPlayer, RenderConfig, ReverbPreset, HOWL_SECTOR_SIZE};
use libspu2::StereoI16;

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

fn build_bench_howl() -> Vec<u8> {
    let mut blk = [0u8; 16];
    blk[0] = 0x0C;
    blk[1] = 0x03; // loop + end
    for (i, b) in blk[2..].iter_mut().enumerate() {
        *b = if i % 2 == 0 { 0x74 } else { 0x9B };
    }
    let wave = blk.repeat(8); // 128 bytes

    // Build 8 polyphonic melodic tracks playing looping chord arpeggios.
    let base_notes = [48u8, 52, 55, 60, 64, 67, 72, 76];
    let mut tracks: Vec<Vec<u8>> = Vec::with_capacity(8);
    for (ti, &base_note) in base_notes.iter().enumerate() {
        let mut trk = Vec::new();
        trk.push(0x00); // melodic
        trk.push(ti as u8);
        push_vlq(&mut trk, 0);
        trk.extend_from_slice(&[0x09, 0x00]); // SetInstrument(0)
        push_vlq(&mut trk, 0);
        trk.extend_from_slice(&[0x06, 0xE0]); // SetVolume
        push_vlq(&mut trk, 0);
        trk.extend_from_slice(&[0x07, (32 + ti * 28) as u8]); // SetPan
        push_vlq(&mut trk, 0);
        trk.extend_from_slice(&[0x08, 0x01]); // SetReverb(1)
        for step in 0..4u8 {
            let note = base_note + step * 2;
            push_vlq(&mut trk, 0);
            trk.extend_from_slice(&[0x05, note, 110]);
            push_vlq(&mut trk, 12);
            trk.extend_from_slice(&[0x01, note]);
        }
        push_vlq(&mut trk, 0);
        trk.push(0x03); // EndOfTrack
        tracks.push(trk);
    }

    let mut seq = Vec::new();
    seq.push(0x00);
    seq.push(tracks.len() as u8);
    seq.extend_from_slice(&120u16.to_le_bytes()); // bpm
    seq.extend_from_slice(&48u16.to_le_bytes()); // tpqn
    let mut rel: u16 = 0;
    for trk in &tracks {
        seq.extend_from_slice(&rel.to_le_bytes());
        rel += trk.len() as u16;
    }
    while (seq.len() & 3) != 0 {
        seq.push(0);
    }
    for trk in &tracks {
        seq.extend_from_slice(trk);
    }

    let mut song_set = Vec::new();
    song_set.extend_from_slice(&0u32.to_le_bytes());
    song_set.push(1); // 1 instrument
    song_set.push(0); // 0 percussion
    song_set.extend_from_slice(&1u16.to_le_bytes());
    // Instrument 0 (12 bytes)
    song_set.push(0x00);
    song_set.push(0xFF);
    song_set.extend_from_slice(&0u16.to_le_bytes());
    song_set.extend_from_slice(&0x1000u16.to_le_bytes());
    song_set.extend_from_slice(&0u16.to_le_bytes());
    song_set.extend_from_slice(&0x80FFu16.to_le_bytes());
    song_set.extend_from_slice(&0x1FC2u16.to_le_bytes());
    song_set.extend_from_slice(&0u16.to_le_bytes()); // seq 0 offset
    while (song_set.len() & 3) != 0 {
        song_set.push(0);
    }
    song_set.extend_from_slice(&seq);
    let ss_len = song_set.len() as u32;
    song_set[0..4].copy_from_slice(&ss_len.to_le_bytes());

    let bank_sector: u16 = 1;
    let song_sector: u16 = 3;
    let total_size = (usize::from(song_sector) + 1) * HOWL_SECTOR_SIZE;
    let mut buf = vec![0u8; total_size];

    let hdr_table_bytes: u32 = 4 + 2 + 2; // 1 spu + 1 bank + 1 song_set
    buf[0x00..0x04].copy_from_slice(b"HOWL");
    buf[0x04..0x08].copy_from_slice(&128u32.to_le_bytes());
    buf[0x10..0x14].copy_from_slice(&1u32.to_le_bytes()); // spu_entry_count
    buf[0x1C..0x20].copy_from_slice(&1u32.to_le_bytes()); // bank_count
    buf[0x20..0x24].copy_from_slice(&1u32.to_le_bytes()); // song_set_count
    buf[0x24..0x28].copy_from_slice(&hdr_table_bytes.to_le_bytes());

    buf[0x28..0x2A].copy_from_slice(&0u16.to_le_bytes());
    buf[0x2A..0x2C].copy_from_slice(&((wave.len() / 8) as u16).to_le_bytes());
    buf[0x2C..0x2E].copy_from_slice(&bank_sector.to_le_bytes());
    buf[0x2E..0x30].copy_from_slice(&song_sector.to_le_bytes());

    let bank_off = usize::from(bank_sector) * HOWL_SECTOR_SIZE;
    buf[bank_off..bank_off + 2].copy_from_slice(&1u16.to_le_bytes());
    buf[bank_off + 2..bank_off + 4].copy_from_slice(&0u16.to_le_bytes());

    let adpcm_off = (usize::from(bank_sector) + 1) * HOWL_SECTOR_SIZE;
    buf[adpcm_off..adpcm_off + wave.len()].copy_from_slice(&wave);

    let song_off = usize::from(song_sector) * HOWL_SECTOR_SIZE;
    buf[song_off..song_off + song_set.len()].copy_from_slice(&song_set);

    buf
}

fn main() {
    const TOTAL_FRAMES: usize = 44_100 * 10; // 10 seconds @ 44.1 kHz
    const CHUNK: usize = 1024;

    let raw = build_bench_howl();
    let view = HowlArchiveView::parse(&raw).unwrap();
    let cfg = RenderConfig {
        target_loops: 1000,
        max_seconds: 600.0,
        reverb_preset: ReverbPreset::Hall,
        ..RenderConfig::bgm_default()
    };

    // 1. StereoI16 streaming
    let mut player = HowlPlayer::from_archive(&view, 0, 0, cfg).unwrap();
    let mut i16_buf = vec![StereoI16::default(); CHUNK];
    let t0 = Instant::now();
    let mut done = 0usize;
    while done < TOTAL_FRAMES {
        let n = (TOTAL_FRAMES - done).min(CHUNK);
        let got = player.render_samples(&mut i16_buf[..n]).unwrap();
        black_box(&i16_buf[..got]);
        if got == 0 {
            player.reset().unwrap();
        }
        done += got;
    }
    let dt_i16 = t0.elapsed().as_secs_f64();
    let rt_i16 = (TOTAL_FRAMES as f64 / 44_100.0) / dt_i16;
    let mfps_i16 = (TOTAL_FRAMES as f64 / dt_i16) / 1e6;

    // 2. Planar f32 (zero-allocation)
    player.reset().unwrap();
    let mut left = vec![0.0f32; CHUNK];
    let mut right = vec![0.0f32; CHUNK];
    let t1 = Instant::now();
    let mut done = 0usize;
    while done < TOTAL_FRAMES {
        let n = (TOTAL_FRAMES - done).min(CHUNK);
        let got = player
            .render_f32_planar(&mut left[..n], &mut right[..n])
            .unwrap();
        black_box(&left[..got]);
        black_box(&right[..got]);
        if got == 0 {
            player.reset().unwrap();
        }
        done += got;
    }
    let dt_planar = t1.elapsed().as_secs_f64();
    let rt_planar = (TOTAL_FRAMES as f64 / 44_100.0) / dt_planar;
    let mfps_planar = (TOTAL_FRAMES as f64 / dt_planar) / 1e6;

    // 3. Interleaved f32 (zero-allocation)
    player.reset().unwrap();
    let mut inter = vec![0.0f32; CHUNK * 2];
    let t2 = Instant::now();
    let mut done = 0usize;
    while done < TOTAL_FRAMES {
        let n = (TOTAL_FRAMES - done).min(CHUNK);
        let got = player.render_f32_interleaved(&mut inter[..n * 2]).unwrap();
        black_box(&inter[..got * 2]);
        if got == 0 {
            player.reset().unwrap();
        }
        done += got;
    }
    let dt_inter = t2.elapsed().as_secs_f64();
    let rt_inter = (TOTAL_FRAMES as f64 / 44_100.0) / dt_inter;
    let mfps_inter = (TOTAL_FRAMES as f64 / dt_inter) / 1e6;

    println!(
        "bench_howl (8-track HOWL + Hall reverb, 10s @ 44.1kHz = {TOTAL_FRAMES} frames):\n\
         - render_samples (i16):   {:.2} ms ({:.2} Mf/s, {:.1}x RT)\n\
         - render_f32_planar:      {:.2} ms ({:.2} Mf/s, {:.1}x RT)\n\
         - render_f32_interleaved: {:.2} ms ({:.2} Mf/s, {:.1}x RT)",
        dt_i16 * 1000.0,
        mfps_i16,
        rt_i16,
        dt_planar * 1000.0,
        mfps_planar,
        rt_planar,
        dt_inter * 1000.0,
        mfps_inter,
        rt_inter,
    );
}
