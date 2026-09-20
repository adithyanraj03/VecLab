//! Deterministic pseudo-random numbers: SplitMix64 + Box-Muller Gaussian sampling.
//!
//! No `rand` crate. SplitMix64 (V8, 2018) gives a fast, well-mixed u64 stream;
//! a Box-Muller transform on top of it gives standard-normal samples. Given the
//! same seed, every stream is bit-identical on every machine.

use std::f64::consts::PI;

/// A SplitMix64 state, advancing by the golden-ratio increment each step.
#[derive(Debug, Clone, Copy)]
pub struct Rng {
    state: u64,
}

const INC: u64 = 0x9E37_79B9_7F4A_7C15;
const M1: u64 = 0xBF58_476D_1CE4_E5B9;
const M2: u64 = 0x94D0_49BB_1331_11EB;

impl Rng {
    /// Create a stream seeded with `seed`.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Current state (exposed for tests; the stream is fully determined by it).
    pub fn state(&self) -> u64 {
        self.state
    }

    /// Next uniform u64, full 64 bits of SplitMix64 output.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(INC);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(M1);
        z = (z ^ (z >> 27)).wrapping_mul(M2);
        z ^ (z >> 31)
    }

    /// Next uniform f64 in [2^-53, 1): 53 bits of resolution, never zero.
    pub fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        bits as f64 / 9_007_199_254_740_992.0 + 1.0 / 9_007_199_254_740_992.0
    }

    /// Next f32 in [0, 1): 24 bits of resolution.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16_777_216.0
    }

    /// Next standard-normal sample (Box-Muller, cosine branch).
    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        r * (2.0 * PI * u2).cos()
    }

    /// Next standard-normal sample, clamped to [-8, 8] and narrowed to f32.
    ///
    /// The clamp keeps corpus generation numerically safe: at |z| > 8 the
    /// Gaussian density is below 1e-15, so clamping is undetectable in
    /// distribution terms and bounds every coordinate we ever emit.
    pub fn next_gaussian_f32(&mut self) -> f32 {
        self.next_gaussian().clamp(-8.0, 8.0) as f32
    }

    /// Uniform integer in [0, n). Panics if n == 0.
    pub fn next_usize_below(&mut self, n: usize) -> usize {
        assert!(n > 0, "next_usize_below(0)");
        // Rejection sampling on the low 32 bits keeps the bias well under 1e-9
        // for any n that fits in a u32, which is every corpus size we target.
        loop {
            let x = (self.next_u64() & 0xFFFF_FFFF) as usize;
            let t = (usize::MAX - n + 1) % n; // = !n % n; values < t are biased
            if x >= t {
                return x % n;
            }
        }
    }
}
