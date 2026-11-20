// Convert Negative: `lightcraft_pipeline::negative::Conversion::pixel`, i.e. Rec.2020 -> linear sRGB,
// `lightcraft_film::Converter::convert_with` on a linear host, then back, step by step. The conversion
// comes as the array `Converter::program` lays out (`prog`; the NEG_* constants are generated from
// `lightcraft_film::program`); the output stage's exact path below NEG_EXACT_BELOW reads its fine
// table. P[0] = pixels, P[1..10] Rec.2020 -> linear sRGB and P[10..19] back (row-major).

// `film` crate helpers, kept in step with their CPU twins.

// Neither NaN nor infinite (a bit test: fast-math may fold `v == v`).
fn neg_finite(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u;
}

// `color::luminance` (Rec.709 weights on linear sRGB).
fn neg_lum(c: vec3<f32>) -> f32 {
    return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
}

// `look::smoothstep`.
fn neg_smooth(x: f32) -> f32 {
    let t = clamp(select(0.0, x, neg_finite(x)), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// `color::srgb_decode`.
fn neg_decode(v: f32) -> f32 {
    if (!neg_finite(v)) {
        return 0.0;
    }
    let x = clamp(v, 0.0, 1.0);
    return select(pow((x + 0.055) / 1.055, 2.4), x / 12.92, x <= 0.04045);
}

// `lookup`: table `size` long at `base`, at `x` in 0..1, linearly interpolated.
fn neg_lookup(base: u32, size: u32, x: f32) -> f32 {
    let t = clamp(x, 0.0, 1.0) * f32(size - 1u);
    let i = min(u32(t), size - 1u);
    let a = prog[base + i];
    if (i + 1u >= size) {
        return a;
    }
    return a + (prog[base + i + 1u] - a) * (t - f32(i));
}

fn neg_h(i: u32) -> f32 {
    return prog[i];
}

fn neg_h3(i: u32) -> vec3<f32> {
    return vec3<f32>(prog[i], prog[i + 1u], prog[i + 2u]);
}

// `Converter::levelled` (the exact path): the levelled positive of a linear scan value.
fn neg_levelled(n: f32, ch: u32) -> f32 {
    var e = neg_h(NEG_FACTOR + ch) * pow(max(n, NEG_MIN_T), -neg_h(NEG_EXPONENT + ch));
    // (an overflow is the most exposure, `film::exposure_encoded_raw`)
    e = select(3.4028235e38, max(e, 1.0), neg_finite(e));
    let v = (e - 1.0) * neg_h(NEG_SCALE);
    return clamp(select(NEG_CAP, v, neg_finite(v)), 0.0, NEG_CAP);
}

// `look::Colour::apply`, monochrome and `Converter::points`: `Converter::graded`.
fn neg_graded(p0: vec3<f32>) -> vec3<f32> {
    var p = p0;
    if (neg_h(NEG_COLOUR_IDENTITY) == 0.0) {
        p *= neg_h3(NEG_GAINS);
        if (neg_h(NEG_MIXES) != 0.0) {
            let m = NEG_MATRIX;
            p = vec3<f32>(
                neg_h(m) * p.x + neg_h(m + 1u) * p.y + neg_h(m + 2u) * p.z,
                neg_h(m + 3u) * p.x + neg_h(m + 4u) * p.y + neg_h(m + 5u) * p.z,
                neg_h(m + 6u) * p.x + neg_h(m + 7u) * p.y + neg_h(m + 8u) * p.z,
            );
        }
        let y = max(neg_lum(p), 0.0);
        let presat = neg_h(NEG_PRESAT);
        if (neg_h(NEG_WEIGHTED) == 0.0) {
            p = max(vec3<f32>(y) + (p - vec3<f32>(y)) * presat, vec3<f32>(0.0));
        } else {
            let yl = sqrt(y);
            let yc = clamp(yl, 0.0, 1.0);
            let w_mid = 4.0 * yc * (1.0 - yc);
            let w_low = neg_smooth((0.35 - yl) / 0.35);
            var x = vec3<f32>(y) + (p - vec3<f32>(y)) * presat;
            x *= vec3<f32>(1.0) + (neg_h3(NEG_MIDS) - vec3<f32>(1.0)) * w_mid;
            x *= vec3<f32>(1.0) + (neg_h3(NEG_SHADOW_TINT) - vec3<f32>(1.0)) * w_low;
            p = max(x, vec3<f32>(0.0));
        }
    }
    if (neg_h(NEG_MONOCHROME) != 0.0) {
        p = vec3<f32>(max(neg_lum(p), 0.0));
    }
    let bp = neg_h(NEG_BLACK_POINT);
    var x = p * neg_h(NEG_WHITE_GAIN);
    if (bp > 0.0) {
        x = vec3<f32>(bp) + x * (1.0 - bp);
    } else if (bp < 0.0) {
        x = (x + vec3<f32>(bp)) / (1.0 + bp);
    }
    return clamp(x, vec3<f32>(0.0), vec3<f32>(1.0));
}

// `look::Toning::apply`.
fn neg_toned(c: vec3<f32>) -> vec3<f32> {
    if (neg_h(NEG_TONING_IDENTITY) != 0.0) {
        return c;
    }
    let y = (c.x + c.y + c.z) / 3.0;
    let rh = clamp(neg_h(NEG_HIGHS_REACH), 0.05, 1.0);
    let rl = clamp(neg_h(NEG_LOWS_REACH), 0.05, 1.0);
    let w_hi = neg_smooth((y - (1.0 - rh)) / rh);
    let w_lo = neg_smooth((rl - y) / rl);
    let v = vec3<f32>(y) + (c - vec3<f32>(y)) * neg_h(NEG_SATURATION);
    return clamp(v + neg_h3(NEG_HIGHS) * w_hi + neg_h3(NEG_LOWS) * w_lo, vec3<f32>(0.0), vec3<f32>(1.0));
}

// `Converter::convert_with` on a linear host's scan sample (0..1 per channel).
fn neg_convert(neg: vec3<f32>) -> vec3<f32> {
    var p = vec3<f32>(0.0);
    for (var ch = 0u; ch < 3u; ch++) {
        let n = neg[ch];
        let u = sqrt(max(n, 0.0));
        if (u >= neg_h(NEG_BLACK + ch)) {
            p[ch] = 0.0;
        } else if (u < NEG_EXACT_BELOW) {
            p[ch] = neg_levelled(n, ch);
        } else {
            p[ch] = max(neg_lookup(NEG_LEVELS + ch * NEG_TABLE, NEG_TABLE, u), 0.0);
        }
    }
    p = neg_graded(p);
    var out = vec3<f32>(0.0);
    for (var ch = 0u; ch < 3u; ch++) {
        let v = p[ch];
        if (v < NEG_EXACT_BELOW) {
            out[ch] = neg_lookup(NEG_FINE + ch * NEG_FINE_SIZE, NEG_FINE_SIZE, v / NEG_EXACT_BELOW);
        } else {
            out[ch] = neg_lookup(NEG_OUTPUT + ch * NEG_TABLE, NEG_TABLE, v);
        }
    }
    if (neg_h(NEG_FOLDED) != 0.0) {
        return out;
    }
    let t = neg_toned(out);
    return vec3<f32>(neg_decode(t.x), neg_decode(t.y), neg_decode(t.z));
}

// A scan sample inside 0..1 (`negative::unit`: not a number is 0).
fn neg_unit(v: f32) -> f32 {
    return select(0.0, clamp(v, 0.0, 1.0), neg_finite(v));
}

fn neg_mat(o: u32, c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        pf(o) * c.x + pf(o + 1u) * c.y + pf(o + 2u) * c.z,
        pf(o + 3u) * c.x + pf(o + 4u) * c.y + pf(o + 5u) * c.z,
        pf(o + 6u) * c.x + pf(o + 7u) * c.y + pf(o + 8u) * c.z,
    );
}

@compute @workgroup_size(256)
fn negative_k(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    let px = vec3<f32>(src[3u * i], src[3u * i + 1u], src[3u * i + 2u]);
    let s = neg_mat(1u, px);
    let pos = neg_convert(vec3<f32>(neg_unit(s.x), neg_unit(s.y), neg_unit(s.z)));
    let out = neg_mat(10u, pos);
    dst[3u * i] = out.x;
    dst[3u * i + 1u] = out.y;
    dst[3u * i + 2u] = out.z;
}
