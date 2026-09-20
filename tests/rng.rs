//! SplitMix64 known-answer tests (reference values computed independently in
//! Python) + statistical and determinism properties.

use veclab::rng::Rng;

/// Reference values for seed 0 (first six u64), computed with an independent
/// Python SplitMix64 implementation.
const SEED0_FIRST6: [u64; 6] = [
    0xe220a8397b1dcdaf,
    0x6e789e6aa1b965f4,
    0x06c45d188009454f,
    0xf88bb8a8724c81ec,
    0x1b39896a51a8749b,
    0x53cb9f0c747ea2ea,
];

const SEED0_STATE_AFTER_6: u64 = 0xb54cda58fbbee87e;

const SEED7_FIRST4: [u64; 4] = [
    0x63cbe1e459320dd7,
    0x044c3cd7f43c661c,
    0xe6984080bab12a02,
    0x953aeb70673e29cb,
];

#[test]
fn kat_seed0_first_six() {
    let mut r = Rng::new(0);
    for &expect in &SEED0_FIRST6 {
        assert_eq!(r.next_u64(), expect);
    }
}

#[test]
fn kat_seed0_state_advances_deterministically() {
    let mut r = Rng::new(0);
    for _ in 0..6 {
        r.next_u64();
    }
    assert_eq!(r.state(), SEED0_STATE_AFTER_6);
}

#[test]
fn kat_seed7_first_four() {
    let mut r = Rng::new(7);
    for &expect in &SEED7_FIRST4 {
        assert_eq!(r.next_u64(), expect);
    }
}

#[test]
fn two_streams_same_seed_bit_identical() {
    let mut a = Rng::new(12345);
    let mut b = Rng::new(12345);
    for _ in 0..1024 {
        assert_eq!(a.next_u64(), b.next_u64());
        assert_eq!(a.next_f64(), b.next_f64());
        assert_eq!(a.next_gaussian(), b.next_gaussian());
    }
}

#[test]
fn different_seeds_differ() {
    let mut a = Rng::new(1);
    let mut b = Rng::new(2);
    // Overwhelmingly likely the streams differ immediately.
    assert_ne!(a.next_u64(), b.next_u64());
}

#[test]
fn state_changes_each_draw() {
    let mut r = Rng::new(0);
    let s0 = r.state();
    r.next_u64();
    assert_ne!(r.state(), s0);
}

#[test]
fn f64_uniform_in_half_open_unit() {
    let mut r = Rng::new(99);
    for _ in 0..20_000 {
        let u = r.next_f64();
        assert!((0.0..1.0).contains(&u), "u={u} out of [0,1)");
    }
}

#[test]
fn f32_uniform_in_half_open_unit() {
    let mut r = Rng::new(42);
    for _ in 0..20_000 {
        let u = r.next_f32();
        assert!((0.0..1.0).contains(&u), "u={u} out of [0,1)");
    }
}

#[test]
fn gaussian_mean_near_zero() {
    let mut r = Rng::new(7);
    const N: u32 = 100_000;
    let mut s = 0.0f64;
    for _ in 0..N {
        s += r.next_gaussian();
    }
    let mean = s / N as f64;
    assert!(mean.abs() < 0.02, "mean={mean} not near 0");
}

#[test]
fn gaussian_std_near_one() {
    let mut r = Rng::new(7);
    const N: u32 = 100_000;
    let mut s = 0.0f64;
    let mut s2 = 0.0f64;
    for _ in 0..N {
        let x = r.next_gaussian();
        s += x;
        s2 += x * x;
    }
    let mean = s / N as f64;
    let var = s2 / N as f64 - mean * mean;
    let std = var.sqrt();
    assert!((std - 1.0).abs() < 0.02, "std={std} not near 1");
}

#[test]
fn gaussian_f32_clamped_to_eight() {
    let mut r = Rng::new(7);
    for _ in 0..200_000 {
        let x = r.next_gaussian_f32();
        assert!((-8.0..=8.0).contains(&x));
        assert!(x.is_finite());
    }
}

#[test]
fn usize_below_in_range() {
    let mut r = Rng::new(5);
    for _ in 0..50_000 {
        let x = r.next_usize_below(17);
        assert!(x < 17);
    }
}

#[test]
fn usize_below_one_is_zero() {
    let mut r = Rng::new(5);
    for _ in 0..100 {
        assert_eq!(r.next_usize_below(1), 0);
    }
}

#[test]
fn usize_below_covers_all_residues() {
    let mut r = Rng::new(11);
    let mut seen = vec![false; 12];
    for _ in 0..100_000 {
        let x = r.next_usize_below(12);
        assert!(x < 12);
        seen[x] = true;
    }
    assert!(seen.iter().all(|&b| b), "not all residues seen: {seen:?}");
}

#[test]
fn usize_below_roughly_uniform() {
    let mut r = Rng::new(3);
    const N: usize = 12;
    let mut counts = vec![0usize; N];
    for _ in 0..120_000 {
        counts[r.next_usize_below(N)] += 1;
    }
    let expected = 120_000 / N;
    for &c in &counts {
        // Each bucket within 15% of expectation.
        assert!((c as f64 - expected as f64).abs() < expected as f64 * 0.15, "counts={counts:?}");
    }
}
