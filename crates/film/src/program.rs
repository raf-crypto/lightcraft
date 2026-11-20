//! A [`Converter`] with its [`Tables`] as plain numbers, for a GPU kernel that runs
//! [`Converter::convert_with`] on a linear host's pixels (the GPU renderer's Convert Negative).
//!
//! One `f32` array: a header of the conversion's numbers at the `HEAD_*` offsets, then the levels,
//! output and fine tables, three channels each. The kernel follows `convert_with` step by step,
//! except for the output stage's exact path: below [`EXACT_BELOW`] it reads the fine table, which
//! samples that path [`FINE_SIZE`] times (so the tone curve needs no second implementation).
//! A host generates its kernel's constants from the ones here, so the two layouts can't drift.

use crate::{Converter, EXACT_BELOW, Encoding, LEVELS_CAP, TABLE_SIZE, Tables};

/// Entries per channel of the fine output table, over `0..EXACT_BELOW`.
pub const FINE_SIZE: usize = 256;

pub const HEAD_FACTOR: usize = 0;
pub const HEAD_EXPONENT: usize = 3;
pub const HEAD_SCALE: usize = 6;
/// Where each channel's levels table reaches its black point (a square-root index).
pub const HEAD_BLACK: usize = 7;
pub const HEAD_GAINS: usize = 10;
/// Row-major 3 × 3.
pub const HEAD_MATRIX: usize = 13;
pub const HEAD_MIXES: usize = 22;
pub const HEAD_PRESAT: usize = 23;
pub const HEAD_WEIGHTED: usize = 24;
pub const HEAD_MIDS: usize = 25;
pub const HEAD_SHADOW_TINT: usize = 28;
pub const HEAD_COLOUR_IDENTITY: usize = 31;
pub const HEAD_MONOCHROME: usize = 32;
pub const HEAD_WHITE_GAIN: usize = 33;
pub const HEAD_BLACK_POINT: usize = 34;
/// The output tables end in linear light (no toning follows); otherwise sRGB-encoded.
pub const HEAD_FOLDED: usize = 35;
pub const HEAD_HIGHS: usize = 36;
pub const HEAD_HIGHS_REACH: usize = 39;
pub const HEAD_LOWS: usize = 40;
pub const HEAD_LOWS_REACH: usize = 43;
pub const HEAD_SATURATION: usize = 44;
pub const HEAD_TONING_IDENTITY: usize = 45;
/// Header length (the tables start here).
pub const HEAD_LEN: usize = 48;

/// Channel `ch`'s levels table starts at `LEVELS + ch * TABLE_SIZE`.
pub const LEVELS: usize = HEAD_LEN;
pub const OUTPUT: usize = LEVELS + 3 * TABLE_SIZE;
pub const FINE: usize = OUTPUT + 3 * TABLE_SIZE;
/// The whole program's length.
pub const LEN: usize = FINE + 3 * FINE_SIZE;

/// Inputs below this take the exact (levels) or fine (output) path, as in `convert_with`.
pub const PROGRAM_EXACT_BELOW: f32 = EXACT_BELOW;
/// Where the levelled positive stops.
pub const PROGRAM_LEVELS_CAP: f32 = LEVELS_CAP;

impl Converter {
    /// This conversion and `t` (filled by [`Converter::tables`] for it) as one array in the layout
    /// above, for a GPU kernel. `None` for an sRGB host: the GPU renderer's pixels are linear.
    pub fn program(&self, t: &Tables) -> Option<Vec<f32>> {
        if self.enc != Encoding::Linear {
            return None;
        }
        let flag = |b: bool| if b { 1.0 } else { 0.0 };
        let mut p = vec![0.0f32; LEN];
        let c = &self.colour;
        let tn = &self.toning;
        for ch in 0..3 {
            p[HEAD_FACTOR + ch] = self.factor[ch];
            p[HEAD_EXPONENT + ch] = self.exponent[ch];
            p[HEAD_BLACK + ch] = t.black[ch];
            p[HEAD_GAINS + ch] = c.gains[ch];
            p[HEAD_MIDS + ch] = c.mids[ch];
            p[HEAD_SHADOW_TINT + ch] = c.shadow_tint[ch];
            p[HEAD_HIGHS + ch] = tn.highs[ch];
            p[HEAD_LOWS + ch] = tn.lows[ch];
            for k in 0..3 {
                p[HEAD_MATRIX + ch * 3 + k] = c.matrix[ch][k];
            }
            p[LEVELS + ch * TABLE_SIZE..LEVELS + (ch + 1) * TABLE_SIZE].copy_from_slice(&t.levels[ch]);
            p[OUTPUT + ch * TABLE_SIZE..OUTPUT + (ch + 1) * TABLE_SIZE].copy_from_slice(&t.output[ch]);
            for i in 0..FINE_SIZE {
                let v = i as f32 / (FINE_SIZE - 1) as f32 * EXACT_BELOW;
                p[FINE + ch * FINE_SIZE + i] = self.output_entry(v, ch, t.folded);
            }
        }
        p[HEAD_SCALE] = self.scale;
        p[HEAD_MIXES] = flag(c.mixes);
        p[HEAD_PRESAT] = c.presat;
        p[HEAD_WEIGHTED] = flag(c.weighted);
        p[HEAD_COLOUR_IDENTITY] = flag(c.is_identity);
        p[HEAD_MONOCHROME] = flag(self.monochrome);
        p[HEAD_WHITE_GAIN] = self.white_gain;
        p[HEAD_BLACK_POINT] = self.black_point;
        p[HEAD_FOLDED] = flag(t.folded);
        p[HEAD_HIGHS_REACH] = tn.highs_reach;
        p[HEAD_LOWS_REACH] = tn.lows_reach;
        p[HEAD_SATURATION] = tn.saturation;
        p[HEAD_TONING_IDENTITY] = flag(tn.is_identity);
        Some(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Analysis, Params};

    #[test]
    fn the_layout_is_packed_and_filled() {
        assert!(HEAD_TONING_IDENTITY < HEAD_LEN);
        assert_eq!(LEN, HEAD_LEN + 6 * TABLE_SIZE + 3 * FINE_SIZE);
        let a = Analysis::manual([0.6, 0.35, 0.15]);
        let c = Converter::new(&a, &Params::default(), Encoding::Linear, None);
        let mut t = std::boxed::Box::new(Tables::blank());
        c.tables(&mut t);
        let p = c.program(&t).expect("a linear host");
        assert_eq!(p.len(), LEN);
        assert!(p.iter().all(|v| v.is_finite()));
        assert_eq!(p[OUTPUT + TABLE_SIZE - 1], t.output[0][TABLE_SIZE - 1]);
        assert!(Converter::new(&a, &Params::default(), Encoding::Srgb, None).program(&t).is_none(), "an sRGB host");
    }
}
