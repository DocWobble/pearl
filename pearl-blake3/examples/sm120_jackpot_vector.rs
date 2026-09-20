use pearl_blake3::blake3_digest;

fn main() {
    let key: [u8; 32] = core::array::from_fn(|i| i as u8);
    let message: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(11));
    let digest = blake3_digest(&message, Some(key));
    for byte in digest { print!("{byte:02x}"); }
    println!();
}
