//! Deliberate corruption gate for rank-128 plain proofs.

use rand::{SeedableRng, rngs::StdRng};
use zk_pow::{
    api::{
        proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, SeedDerivation},
        verify::verify_plain_proof,
    },
    ffi::mine::try_mine_one,
};

fn fixture() -> (IncompleteBlockHeader, MiningConfiguration) {
    (
        IncompleteBlockHeader {
            version: 0,
            prev_block: [0x51; 32],
            merkle_root: *b"sm120-reference-vector-fixture!!",
            timestamp: 0x6666_6666,
            nbits: 0x207f_ffff,
        },
        MiningConfiguration {
            common_dim: 2048,
            rank: 128,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 32, 33, 40, 41]).unwrap(),
            moe: None,
        },
    )
}

#[test]
fn canonical_verifier_rejects_corrupted_rank128_fixtures() {
    let (header, config) = fixture();
    for index in 0..5u64 {
        let mut rng = StdRng::seed_from_u64(0xC0DE_0000u64 ^ index);
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
        verify_plain_proof(&header, &proof, None, SeedDerivation::Legacy).unwrap();

        let mut root_a = proof.clone();
        root_a.a.proof.root[0] ^= 1;
        assert!(verify_plain_proof(&header, &root_a, None, SeedDerivation::Legacy).is_err());

        let mut root_b = proof.clone();
        root_b.bt.proof.root[0] ^= 1;
        assert!(verify_plain_proof(&header, &root_b, None, SeedDerivation::Legacy).is_err());

        let mut leaf = proof.clone();
        leaf.a.proof.leaf_data[0][0] ^= 1;
        assert!(verify_plain_proof(&header, &leaf, None, SeedDerivation::Legacy).is_err());

        let mut index_fixture = proof.clone();
        index_fixture.a.row_indices[0] = 128;
        assert!(verify_plain_proof(&header, &index_fixture, None, SeedDerivation::Legacy).is_err());
    }
}
