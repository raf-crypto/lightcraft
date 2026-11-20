//! Film dust detection on real scans, for review by eye:
//! `cargo run --release -p lightcraft-engine --example film_dust_eval -- OUT_DIR scan.CR3 …`
//!
//! Imports each scan (its XMP sidecar's edits applied; nothing is ever written next to it), runs
//! the detection `spot.findDust` runs on a film scan (`lightcraft_pipeline::film_dust`) on the
//! uncropped photo, and writes to OUT_DIR, per scan: `<name>-frame.png` (the frame, about 2400 px,
//! every detection circled: red specks, blue hairs) and `<name>-sheet.png` (the strongest
//! detections, each a 96 px crop at full size, enlarged 2×). Prints the counts and the time.
//! `SENSITIVITY=0..100` (50), `POLARITY=bright|dark` (bright when the scan is a converted negative).
use std::time::Instant;

use lightcraft_codecs::encode::{EncodeImage, EncodeMeta, encode_png};
use lightcraft_engine::Session;
use lightcraft_engine::catalog::PhotoId;
use lightcraft_pipeline::film_dust::{Found, Params, Polarity, detect};
use lightcraft_raster::Rgba8;

const SHEET: usize = 8;
const CROP: usize = 96;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(args.len() >= 2, "usage: film_dust_eval OUT_DIR FILE...");
    let out = std::path::Path::new(&args[0]);
    std::fs::create_dir_all(out).expect("output folder");
    let sensitivity: f32 = std::env::var("SENSITIVITY").ok().and_then(|v| v.parse().ok()).unwrap_or(50.0);
    let mut s = Session::new().with_fs();
    s.execute("library.xmpPreferences", &serde_json::json!({"autoWrite": false})).expect("auto-write off");
    for path in &args[1..] {
        let path = std::fs::canonicalize(path).expect("input file").to_string_lossy().to_string();
        let r = s.execute("library.import", &serde_json::json!({"paths": [path], "mode": "add"})).expect("import");
        let Some(id) = r["imported"][0].as_u64().map(PhotoId) else {
            println!("{path}: not imported ({r})");
            continue;
        };
        let negative = s.develop_of(id).is_some_and(|d| lightcraft_pipeline::negative::enabled(&d));
        let polarity = match std::env::var("POLARITY").as_deref() {
            Ok("dark") => Polarity::Dark,
            Ok("bright") => Polarity::Bright,
            _ if negative => Polarity::Bright,
            _ => Polarity::Dark,
        };
        let p = s.catalog.photo(id).expect("photo");
        let edge = (p.width.max(p.height) as usize).clamp(1, 8192);
        let t = Instant::now();
        let img = s.render_job(id, edge, edge, false, false).expect("render job").run().rendered.expect("render").image;
        let rendered = t.elapsed();
        let t = Instant::now();
        let region = s.develop_of(id).map(|d| lightcraft_pipeline::film_dust::crop_region(&d, img.width as f64, img.height as f64));
        let found = detect(&img, &Params { sensitivity, polarity, region, ..Params::default() });
        let detected = t.elapsed();
        let name = std::path::Path::new(&path).file_stem().map_or("scan".into(), |n| n.to_string_lossy().to_string());
        let hairs = found.iter().filter(|f| f.hair).count();
        println!(
            "{name}: {}×{} {:?}{}, {} found ({} hairs), render {:.1} s, detect {:.2} s",
            img.width,
            img.height,
            polarity,
            if negative { " (converted negative)" } else { "" },
            found.len(),
            hairs,
            rendered.as_secs_f64(),
            detected.as_secs_f64()
        );
        save(&frame(&img, &found), &out.join(format!("{name}-frame.png")));
        save(&sheet(&img, &found), &out.join(format!("{name}-sheet.png")));
    }
}

fn save(img: &Rgba8, path: &std::path::Path) {
    let png = encode_png(&EncodeImage::rgba8(img), &EncodeMeta::default()).expect("png");
    std::fs::write(path, png).expect("write png");
}

/// `img` shrunk to about 2400 px with every detection circled.
fn frame(img: &Rgba8, found: &[Found]) -> Rgba8 {
    let k = (img.width.max(img.height) as f64 / 2400.0).max(1.0);
    let (w, h) = ((img.width as f64 / k) as usize, (img.height as f64 / k) as usize);
    let mut o = Rgba8 { width: w, height: h, data: (0..w * h).map(|i| img.data[((i / w) as f64 * k) as usize * img.width + ((i % w) as f64 * k) as usize]).collect() };
    for f in found {
        for &(x, y) in &f.points {
            let r = (f.radius * img.width.max(img.height) as f64 / k).max(4.0) + 2.0;
            circle(&mut o, x * w as f64, y * h as f64, r, if f.hair { [40, 120, 255, 255] } else { [255, 40, 40, 255] });
        }
    }
    o
}

/// The strongest detections as full-size crops, enlarged 2×, in a grid.
fn sheet(img: &Rgba8, found: &[Found]) -> Rgba8 {
    let n = found.len().clamp(1, SHEET * 6);
    let rows = n.div_ceil(SHEET);
    let cell = CROP * 2 + 4;
    let (w, h) = (SHEET * cell, rows * cell);
    let mut o = Rgba8 { width: w, height: h, data: vec![[30, 30, 30, 255]; w * h] };
    for (k, f) in found.iter().take(n).enumerate() {
        let (cx, cy) = (f.points[0].0 * img.width as f64, f.points[0].1 * img.height as f64);
        let (ox, oy) = ((k % SHEET) * cell + 2, (k / SHEET) * cell + 2);
        for y in 0..CROP * 2 {
            for x in 0..CROP * 2 {
                let sx = (cx - CROP as f64 / 2.0 + x as f64 / 2.0).clamp(0.0, img.width as f64 - 1.0) as usize;
                let sy = (cy - CROP as f64 / 2.0 + y as f64 / 2.0).clamp(0.0, img.height as f64 - 1.0) as usize;
                o.data[(oy + y) * w + ox + x] = img.data[sy * img.width + sx];
            }
        }
        let r = f.radius * img.width.max(img.height) as f64 * 2.0 + 3.0;
        circle(&mut o, (ox + CROP) as f64, (oy + CROP) as f64, r.min(CROP as f64), if f.hair { [40, 120, 255, 255] } else { [255, 40, 40, 255] });
    }
    o
}

fn circle(o: &mut Rgba8, cx: f64, cy: f64, r: f64, c: [u8; 4]) {
    for t in 0..((r * 7.0) as usize).max(24) {
        let a = t as f64 / ((r * 7.0) as usize).max(24) as f64 * std::f64::consts::TAU;
        let (x, y) = (cx + r * a.cos(), cy + r * a.sin());
        if x >= 0.0 && y >= 0.0 && (x as usize) < o.width && (y as usize) < o.height {
            o.data[y as usize * o.width + x as usize] = c;
        }
    }
}
