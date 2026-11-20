//! Dust and hairs on film scans: specks of dust, fibres and hairs that sat on the film when it was
//! scanned. They block light, so on a converted negative they are brighter than their surroundings
//! (on a slide, or a photo, darker); they lie on the film, so they are as sharp as its grain, and
//! narrow. Unlike sensor dust ([`crate::dust`]) they are small (a few pixels of a 45 MP scan), and
//! hairs are long.
//!
//! Classical: a white top-hat (the image minus its opening by a square about 2.5 ‰ of the long edge
//! wide) keeps what is narrower than that square; each tile's own spread of it (film grain, texture)
//! sets how far a pixel must stand above it, so grain and busy detail don't trigger. Compact blobs
//! are specks; long thin ones are hairs (and scratches), returned as stroke points for a brushed spot.

use lightcraft_raster::morph::{max_filter, min_filter};
use lightcraft_raster::blur::gaussian;
use lightcraft_raster::{Plane, Rgba8};
use rayon::prelude::*;

/// Which way the defects stand out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Polarity {
    /// Brighter than their surroundings: a converted negative.
    Bright,
    /// Darker: a slide, or a scan not converted.
    Dark,
}

#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// 0..100: higher finds fainter defects (and more false ones).
    pub sensitivity: f32,
    pub polarity: Polarity,
    /// Also return hairs and scratches (long thin defects).
    pub hairs: bool,
    /// At most this many, the strongest.
    pub max: usize,
    /// Only within this part of the image (x0, y0, x1, y1, 0..1): the photo's crop, which leaves
    /// out the film's rebate and the holder, whose sharp edges would otherwise fill the list.
    pub region: Option<(f64, f64, f64, f64)>,
}

impl Default for Params {
    fn default() -> Self {
        Params { sensitivity: 50.0, polarity: Polarity::Bright, hairs: true, max: 300, region: None }
    }
}

/// A found defect: its points (0..1 of the image's width / height; one for a speck, several
/// along a hair), a radius (fraction of the long edge) that covers it, and how far it stands above
/// its tile's threshold (≥ 1).
#[derive(Clone, Debug, PartialEq)]
pub struct Found {
    pub points: Vec<(f64, f64)>,
    pub radius: f64,
    pub strength: f32,
    pub hair: bool,
}

/// The part of the uncropped frame (`w × h`) that `s`'s crop keeps, for [`Params::region`]: its
/// rectangle a little inside (a straightened crop's corners reach past its upright box).
pub fn crop_region(s: &lightcraft_develop::DevelopSettings, w: f64, h: f64) -> (f64, f64, f64, f64) {
    let g = s.crop.geometry.bounded(w, h);
    let r = g.rect;
    let m = g.angle.to_radians().sin().abs() * 0.5 + 0.005;
    (r.x0 + m, r.y0 + m, r.x1 - m, r.y1 - m)
}

/// Tiles of this many pixels on a side each have their own threshold.
const TILE: usize = 128;

/// Dust specks and hairs in a rendered (display-encoded) image of the whole frame.
pub fn detect(img: &Rgba8, p: &Params) -> Vec<Found> {
    let (w, h) = (img.width, img.height);
    if w < 32 || h < 32 || img.data.len() != w * h {
        return Vec::new();
    }
    let long = w.max(h);
    let dark = p.polarity == Polarity::Dark;
    let lum = Plane::from_fn(w, h, |x, y| {
        let c = img.data[y * w + x];
        let v = (0.2126 * c[0] as f32 + 0.7152 * c[1] as f32 + 0.0722 * c[2] as f32) / 255.0;
        if dark { 1.0 - v } else { v }
    });
    // (at least 10 px: a speck's soft edge reaches about 2.5 σ, and dust of a few pixels' σ must fit)
    let r = ((long as f32 * 0.0025).round() as usize).max(10);
    let opened = max_filter(&min_filter(&lum, r), r);
    let top: Vec<f32> = lum.data.iter().zip(&opened.data).map(|(a, b)| (a - b).max(0.0)).collect();
    let s = if p.sensitivity.is_finite() { (p.sensitivity / 100.0).clamp(0.0, 1.0) } else { 0.5 };
    // how many robust spreads of its tile's top-hat a pixel must stand above, and at least how far
    let k = 12.0 - 8.0 * s;
    let floor = 0.06 - 0.04 * s;
    let (tw, th) = (w.div_ceil(TILE), h.div_ceil(TILE));
    let thr: Vec<f32> = (0..tw * th)
        .into_par_iter()
        .map(|t| {
            let (x0, y0) = ((t % tw) * TILE, (t / tw) * TILE);
            let mut v: Vec<f32> = (y0..(y0 + TILE).min(h)).flat_map(|y| top[y * w + x0..y * w + (x0 + TILE).min(w)].iter().copied()).collect();
            let med = median(&mut v);
            for x in &mut v {
                *x = (*x - med).abs();
            }
            let mad = median(&mut v);
            (med + k * 1.4826 * mad).max(floor)
        })
        .collect();
    // Dust lies on a fairly even background: where the background itself jumps (an edge within the
    // square, e.g. a window frame, or the film's rebate), the bright side's ragged edge stands out
    // of the top-hat too, so the threshold rises with the background's local range.
    let range: Vec<f32> = {
        let (hi, lo) = (max_filter(&opened, r), min_filter(&opened, r));
        hi.data.iter().zip(&lo.data).map(|(a, b)| a - b).collect()
    };
    let (rx0, ry0, rx1, ry1) = match p.region {
        Some((x0, y0, x1, y1)) if x1 > x0 && y1 > y0 => (
            (x0.clamp(0.0, 1.0) * w as f64) as usize,
            (y0.clamp(0.0, 1.0) * h as f64) as usize,
            (x1.clamp(0.0, 1.0) * w as f64) as usize,
            (y1.clamp(0.0, 1.0) * h as f64) as usize,
        ),
        _ => (0, 0, w, h),
    };
    // The tile's threshold misses texture smaller than a tile (a head of hair beside smooth skin):
    // the neighbourhood's own mean and spread of the top-hat (clipped at the tile's threshold, so
    // a defect doesn't raise its own) set a second one.
    let tile_thr = |i: usize| thr[(i / w / TILE) * tw + (i % w) / TILE];
    let local: Vec<f32> = {
        let clipped = Plane { width: w, height: h, data: (0..w * h).map(|i| top[i].min(tile_thr(i))).collect() };
        let sq = Plane { width: w, height: h, data: clipped.data.iter().map(|v| v * v).collect() };
        let sigma = 2.0 * r as f32;
        let (mean, mean_sq) = (gaussian(&clipped, sigma), gaussian(&sq, sigma));
        // (the plain spread of the skewed top-hat runs larger than the tile's robust one: 0.75 × k)
        mean.data.iter().zip(&mean_sq.data).map(|(m, m2)| m + 0.75 * k * (m2 - m * m).max(0.0).sqrt()).collect()
    };
    let thr_at = |i: usize| tile_thr(i).max(local[i]) + 0.5 * range[i];
    let on: Vec<bool> = (0..w * h)
        .into_par_iter()
        .map(|i| {
            let (x, y) = (i % w, i / w);
            x >= rx0 && x < rx1 && y >= ry0 && y < ry1 && top[i] > thr_at(i)
        })
        .collect();
    let min_px = (5.0 - 3.0 * s).round() as usize;
    let cap = 64 * r * r; // a component larger than this is a structure, not a defect
    let mut seen = vec![false; w * h];
    let mut out = Vec::new();
    let mut pixels: Vec<usize> = Vec::new();
    for start in 0..w * h {
        if seen[start] || !on[start] {
            continue;
        }
        pixels.clear();
        let mut stack = vec![start];
        seen[start] = true;
        let mut peak = 0f32;
        while let Some(i) = stack.pop() {
            if pixels.len() <= cap {
                pixels.push(i);
            }
            peak = peak.max(top[i] / thr_at(i));
            let (x, y) = (i % w, i / w);
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if !seen[j] && on[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        let n = pixels.len();
        if n < min_px || n > cap || !sharp(&lum, &top, &pixels) {
            continue;
        }
        let Some(f) = describe(&pixels, w, r, p.hairs) else { continue };
        out.push(Found {
            points: f.0.into_iter().map(|(x, y)| ((x + 0.5) / w as f64, (y + 0.5) / h as f64)).collect(),
            radius: f.1 / long as f64,
            strength: peak,
            hair: f.2,
        });
    }
    out.sort_by(|a, b| b.strength.total_cmp(&a.strength));
    out.truncate(p.max);
    out
}

/// A component's points (pixels), covering radius (pixels) and whether it is a hair; `None` when
/// it is neither a compact speck nor (with `hairs`) a long thin hair.
fn describe(pixels: &[usize], w: usize, r: usize, hairs: bool) -> Option<(Vec<(f64, f64)>, f64, bool)> {
    let n = pixels.len() as f64;
    let (mut sx, mut sy) = (0.0, 0.0);
    let (mut x0, mut x1, mut y0, mut y1) = (usize::MAX, 0, usize::MAX, 0);
    for &i in pixels {
        let (x, y) = (i % w, i / w);
        sx += x as f64;
        sy += y as f64;
        (x0, x1, y0, y1) = (x0.min(x), x1.max(x), y0.min(y), y1.max(y));
    }
    let (cx, cy) = (sx / n, sy / n);
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for &i in pixels {
        let (dx, dy) = ((i % w) as f64 - cx, (i / w) as f64 - cy);
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    let (sxx, syy, sxy) = (sxx / n, syy / n, sxy / n);
    // the covariance's eigenvalues: spread along and across the component
    let mid = (sxx + syy) / 2.0;
    let d = (((sxx - syy) / 2.0).powi(2) + sxy * sxy).sqrt();
    let (along, across) = (mid + d, (mid - d).max(1.0 / 12.0));
    let length = (12.0 * along).sqrt(); // a uniform line of length L has variance L²/12
    let elongation = (along / across).sqrt();
    let (bw, bh) = ((x1 - x0 + 1) as f64, (y1 - y0 + 1) as f64);
    if elongation >= 4.0 && length >= 4.0 * r as f64 {
        if !hairs {
            return None;
        }
        // stroke points: the centroids of the hair's pixels in cells about twice its width
        let width = (n / length).max(1.0);
        let cell = (2.0 * width).max(3.0);
        let mut cells: std::collections::HashMap<(i64, i64), (f64, f64, f64)> = std::collections::HashMap::new();
        for &i in pixels {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            let e = cells.entry(((x / cell) as i64, (y / cell) as i64)).or_insert((0.0, 0.0, 0.0));
            *e = (e.0 + x, e.1 + y, e.2 + 1.0);
        }
        let mut points: Vec<(f64, f64)> = cells.into_values().map(|(x, y, k)| (x / k, y / k)).collect();
        points.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        return Some((points, (width / 2.0 + 1.5) * 1.3, true));
    }
    // a speck: compact, and filling a fair part of its box
    if elongation > 3.0 || n < 0.3 * bw * bh {
        return None;
    }
    Some((vec![(cx, cy)], (bw.max(bh) / 2.0 + 1.5) * 1.3, false))
}

/// Whether a component is as sharp as something lying on the film: its steepest step (a central
/// difference of the image) is a fair part of its height above the background. Dust and hairs
/// have edges a pixel or two wide; a bright, out-of-focus part of the scene rises over many.
fn sharp(lum: &Plane, top: &[f32], pixels: &[usize]) -> bool {
    let (w, h) = (lum.width, lum.height);
    let (mut height, mut step) = (0f32, 0f32);
    for &i in pixels {
        height = height.max(top[i]);
        let (x, y) = (i % w, i / w);
        if x > 0 && x + 1 < w && y > 0 && y + 1 < h {
            let d = &lum.data;
            let gx = (d[i + 1] - d[i - 1]).abs() / 2.0;
            let gy = (d[i + w] - d[i - w]).abs() / 2.0;
            step = step.max(gx.max(gy));
        }
    }
    height > 0.0 && step >= 0.25 * height
}

/// The median of `v` (reordered); 0 when empty.
fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let m = v.len() / 2;
    *v.select_nth_unstable_by(m, |a, b| a.total_cmp(b)).1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grey scan with film grain (a deterministic noise of `grain` spread), as display values.
    fn grainy(w: usize, h: usize, grain: f32) -> Vec<f32> {
        let mut state = 0x2545_f491_u32;
        let mut noise = || {
            // sum of four uniforms: close enough to a Gaussian for grain
            let mut s = 0.0;
            for _ in 0..4 {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                s += state as f32 / u32::MAX as f32 - 0.5;
            }
            s * grain * 1.7
        };
        (0..w * h).map(|i| 0.45 + 0.1 * (i / w) as f32 / h as f32 + noise()).collect()
    }

    fn to_rgba(v: &[f32], w: usize, h: usize) -> Rgba8 {
        let data = v.iter().map(|x| (x.clamp(0.0, 1.0) * 255.0).round() as u8).map(|b| [b, b, b, 255]).collect();
        Rgba8 { width: w, height: h, data }
    }

    /// A particle of radius `r` lying on the film: a hard-edged disc, softened by the lens over
    /// about 0.7 px. (`r` ≥ 20: a soft out-of-focus blob of that size instead, as the scene's.)
    fn speck(v: &mut [f32], w: usize, (cx, cy): (f32, f32), r: f32, amp: f32) {
        let soft = if r >= 20.0 { r } else { 0.7 };
        for y in 0..v.len() / w {
            for x in 0..w {
                let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
                let t = ((r + soft - d) / (2.0 * soft)).clamp(0.0, 1.0);
                v[y * w + x] += amp * t * t * (3.0 - 2.0 * t);
            }
        }
    }

    const SPECKS: [(f32, f32, f32); 6] = [(150.0, 120.0, 1.5), (420.0, 300.0, 2.5), (700.0, 520.0, 3.5), (980.0, 200.0, 2.0), (300.0, 650.0, 5.0), (1100.0, 700.0, 1.2)];

    /// Specks of 1–5 px and a curved hair on grainy grey: every speck and the hair found, and
    /// at most a couple of grain clumps besides.
    #[test]
    fn finds_specks_and_a_hair_on_grain() {
        let (w, h) = (1200usize, 800usize);
        let mut v = grainy(w, h, 0.03);
        for (cx, cy, r) in SPECKS {
            speck(&mut v, w, (cx, cy), r, 0.35);
        }
        // a hair: a curve 2 px wide from (200, 400) to (500, 460)
        for t in 0..1200 {
            let x = 200.0 + t as f32 * 0.25;
            let y = 400.0 + 30.0 * (t as f32 / 1200.0 * std::f32::consts::PI).sin() + t as f32 * 0.05;
            for (dx, dy) in [(0, 0), (0, 1), (1, 0)] {
                v[(y as usize + dy) * w + x as usize + dx] += 0.3;
            }
        }
        let found = detect(&to_rgba(&v, w, h), &Params::default());
        for (cx, cy, _) in SPECKS {
            assert!(
                found.iter().any(|f| !f.hair && (f.points[0].0 * w as f64 - cx as f64).abs() < 3.0 && (f.points[0].1 * h as f64 - cy as f64).abs() < 3.0),
                "speck at {cx},{cy} missed: {found:?}"
            );
        }
        let hair = found.iter().find(|f| f.hair).expect("the hair");
        let xs: Vec<f64> = hair.points.iter().map(|p| p.0 * w as f64).collect();
        assert!(xs.iter().cloned().fold(f64::MAX, f64::min) < 215.0 && xs.iter().cloned().fold(0.0, f64::max) > 485.0, "{xs:?}");
        assert!(found.len() <= SPECKS.len() + 1 + 2, "{} found", found.len());
    }

    #[test]
    fn grain_alone_and_wide_bright_areas_are_not_dust() {
        let (w, h) = (1200usize, 800usize);
        let mut v = grainy(w, h, 0.04);
        speck(&mut v, w, (600.0, 400.0), 60.0, 0.4); // a lamp, not dust
        let found = detect(&to_rgba(&v, w, h), &Params::default());
        assert!(found.len() <= 1, "{found:?}");
    }

    #[test]
    fn dark_specks_need_the_dark_polarity() {
        let (w, h) = (800usize, 600usize);
        let mut v = grainy(w, h, 0.03);
        speck(&mut v, w, (300.0, 300.0), 2.5, -0.3);
        let img = to_rgba(&v, w, h);
        assert!(detect(&img, &Params::default()).iter().all(|f| (f.points[0].0 * w as f64 - 300.0).abs() > 5.0));
        let dark = detect(&img, &Params { polarity: Polarity::Dark, ..Params::default() });
        assert!(dark.iter().any(|f| (f.points[0].0 * w as f64 - 300.0).abs() < 3.0), "{dark:?}");
    }
}
