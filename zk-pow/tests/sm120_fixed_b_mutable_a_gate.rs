//! Canonical V3 gate for the fixed-B / mutable-A nonce construction.
//!
//! This proves the protocol property needed by the standalone miner: B may
//! remain byte-identical while a reserved, unopened A row changes the A root,
//! salted A binding and A-side noise seed. The opened jackpot strips remain
//! ordinary in-range matrix data and the canonical verifier must accept every
//! resulting proof.

use pearl_blake3::{blake3_digest, MerkleTree};
use zk_pow::{
    api::{
        proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, SeedDerivation},
        verify::verify_plain_proof,
    },
    ffi::plain_proof::{MatrixMerkleProof, PlainProof},
};

const M: usize = 128;
const N: usize = 128;
const K: usize = 2048;
const NONCE_ROW: usize = M - 1;

const A_ROWS: [usize; 4] = [0, 8, 64, 72];
const B_COLS: [usize; 8] = [0, 1, 8, 9, 32, 33, 40, 41];

fn fixture() -> (IncompleteBlockHeader, MiningConfiguration) {
    let header = IncompleteBlockHeader {
        version: 0,
        prev_block: [0x31; 32],
        merkle_root: *b"sm120-fixed-b-mutable-a-v3-gate!",
        timestamp: 0x6a6a_6a6a,
        // Deliberately easy local release-gate target. Validity is still checked
        // by the real V3 parser, commitment/noise derivation and jackpot path.
        nbits: 0x207f_ffff,
    };
    let config = MiningConfiguration {
        common_dim: K as u32,
        rank: 128,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
        cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 32, 33, 40, 41]).unwrap(),
        moe: None,
    };
    (header, config)
}

fn job_key(header: &IncompleteBlockHeader, config: &MiningConfiguration) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(128);
    bytes.extend_from_slice(&header.to_bytes());
    bytes.extend_from_slice(&config.to_bytes());
    blake3_digest(&bytes, None)
}

fn encode_nonce_row(a: &mut [u8], nonce: u64) {
    let start = NONCE_ROW * K;
    a[start..start + K].fill(0);

    // Encode the nonce as 64 ordinary matrix values in {0,1}. The row is never
    // among A_ROWS, so it changes the commitment without entering the opened
    // jackpot strips.
    for bit in 0..64 {
        a[start + bit] = ((nonce >> bit) & 1) as u8;
    }
}

fn proof_for(a_bytes: &[u8], b_bytes: &[u8], key: [u8; 32]) -> PlainProof {
    let a_tree = MerkleTree::new(a_bytes, key);
    let b_tree = MerkleTree::new(b_bytes, key);

    let a_leaves = MerkleTree::compute_leaf_indices_from_rows(&A_ROWS, (M, K));
    let b_leaves = MerkleTree::compute_leaf_indices_from_rows(&B_COLS, (N, K));

    PlainProof {
        m: M,
        n: N,
        k: K,
        noise_rank: 128,
        a: MatrixMerkleProof {
            proof: a_tree.get_multileaf_proof(&a_leaves),
            row_indices: A_ROWS.to_vec(),
        },
        bt: MatrixMerkleProof {
            proof: b_tree.get_multileaf_proof(&b_leaves),
            row_indices: B_COLS.to_vec(),
        },
        moe: None,
    }
}

#[test]
#[ignore = "release gate: run explicitly with --ignored"]
fn v3_fixed_b_mutable_a_accepts_1000_of_1000() {
    let (header, config) = fixture();
    let key = job_key(&header, &config);

    let mut a_bytes = vec![0u8; M * K];
    let b_bytes = vec![0u8; N * K];

    let fixed_b_tree = MerkleTree::new(&b_bytes, key);
    let fixed_b_root = fixed_b_tree.root();
    let mut previous_a_root = None;

    for nonce in 0..1000u64 {
        encode_nonce_row(&mut a_bytes, nonce);
        let proof = proof_for(&a_bytes, &b_bytes, key);

        assert_eq!(proof.bt.proof.root, fixed_b_root, "B root changed at nonce {nonce}");
        assert!(!A_ROWS.contains(&NONCE_ROW), "reserved nonce row entered jackpot strips");

        if let Some(previous) = previous_a_root {
            assert_ne!(proof.a.proof.root, previous, "A root did not change at nonce {nonce}");
        }
        previous_a_root = Some(proof.a.proof.root);

        verify_plain_proof(&header, &proof, None, SeedDerivation::Salted)
            .unwrap_or_else(|error| panic!("V3 fixed-B proof {nonce} rejected: {error}"));
    }
}
