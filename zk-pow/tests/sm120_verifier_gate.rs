//! The handoff's 1,000-case canonical verifier gate.
//!
//! It is ignored in the default suite because it is intentionally a long release check.

use rand::{SeedableRng, rngs::StdRng};
use zk_pow::{
    api::{
        proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, SeedDerivation},
        verify::verify_plain_proof,
    },
    ffi::mine::try_mine_one,
};

fn fixture_config() -> (IncompleteBlockHeader, MiningConfiguration) {
    let header = IncompleteBlockHeader {
        version: 0,
        prev_block: [0x51; 32],
        merkle_root: *b"sm120-reference-vector-fixture!!",
        timestamp: 0x6666_6666,
        nbits: 0x207f_ffff,
    };
    let config = MiningConfiguration {
        common_dim: 2048,
        rank: 128,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
        cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 32, 33, 40, 41]).unwrap(),
        moe: None,
    };
    (header, config)
}

#[test]
#[ignore = "release gate: run explicitly with --ignored"]
fn canonical_verifier_accepts_1000_of_1000_rank128_vectors() {
    let (header, config) = fixture_config();
    for index in 0..1000u64 {
        let mut rng = StdRng::seed_from_u64(0x5045_4152_4c53_4d31 ^ index);
        let proof = loop {
            if let Some(proof) = try_mine_one(
                &mut rng,
                128,
                128,
                2048,
                header,
                config.clone(),
                None,
                false,
                SeedDerivation::Legacy,
            )
            .unwrap()
            {
                break proof;
            }
        };
        verify_plain_proof(&header, &proof, None, SeedDerivation::Legacy)
            .unwrap_or_else(|error| panic!("canonical verifier rejected vector {index}: {error}"));
    }
}
