
pub type Bitboard = u64;

pub const EMPTY: Bitboard = 0;
pub const FULL: Bitboard = 0xFFFF_FFFF_FFFF_FFFF;

pub const RANK_1: Bitboard = 0x0000_0000_0000_00FF;
pub const RANK_2: Bitboard = 0x0000_0000_0000_FF00;
pub const RANK_3: Bitboard = 0x0000_0000_00FF_0000;
pub const RANK_4: Bitboard = 0x0000_0000_FF00_0000;
pub const RANK_5: Bitboard = 0x0000_00FF_0000_0000;
pub const RANK_6: Bitboard = 0x0000_FF00_0000_0000;
pub const RANK_7: Bitboard = 0x00FF_0000_0000_0000;
pub const RANK_8: Bitboard = 0xFF00_0000_0000_0000;

pub const FILE_A: Bitboard = 0x0101_0101_0101_0101;
pub const FILE_B: Bitboard = 0x0202_0202_0202_0202;
pub const FILE_C: Bitboard = 0x0404_0404_0404_0404;
pub const FILE_D: Bitboard = 0x0808_0808_0808_0808;
pub const FILE_E: Bitboard = 0x1010_1010_1010_1010;
pub const FILE_F: Bitboard = 0x2020_2020_2020_2020;
pub const FILE_G: Bitboard = 0x4040_4040_4040_4040;
pub const FILE_H: Bitboard = 0x8080_8080_8080_8080;

pub const CENTER: Bitboard = 0x0000_0018_3C00_0000;
pub const EXTENDED_CENTER: Bitboard = 0x0000_3C3C_3C3C_0000;

pub const SQUARE_TO_FILE: [u8; 64] = [
    0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
    0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
    0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
    0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
];

pub const SQUARE_TO_RANK: [u8; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
];

#[inline(always)]
pub const fn square_mask(square: u8) -> Bitboard {
    1u64 << (square & 63)
}

#[inline(always)]
pub fn popcount(bb: Bitboard) -> u8 {
    bb.count_ones() as u8
}

#[inline(always)]
pub fn pop_lsb(bb: &mut Bitboard) -> u8 {
    let original = *bb;
    *bb &= *bb - 1;
    original.trailing_zeros() as u8
}

#[inline(always)]
pub const fn has_bit(bb: Bitboard, square: u8) -> bool {
    (bb & (1u64 << (square & 63))) != 0
}

#[inline(always)]
pub fn set_bit(bb: &mut Bitboard, square: u8) {
    *bb |= 1u64 << (square & 63);
}

#[inline(always)]
pub fn clear_bit(bb: &mut Bitboard, square: u8) {
    *bb &= !(1u64 << (square & 63));
}

#[inline(always)]
pub fn toggle_bit(bb: &mut Bitboard, square: u8) {
    *bb ^= 1u64 << (square & 63);
}

#[inline(always)]
pub const fn lsb(bb: Bitboard) -> Bitboard {
    bb & bb.wrapping_neg()
}

#[inline(always)]
pub const fn reset_lsb(bb: Bitboard) -> Bitboard {
    bb & (bb - 1)
}

#[inline(always)]
pub fn msb(bb: Bitboard) -> Bitboard {
    if bb == 0 {
        0
    } else {
        1u64 << (63 - bb.leading_zeros())
    }
}

#[inline(always)]
pub const fn shift_north(bb: Bitboard) -> Bitboard {
    bb << 8
}

#[inline(always)]
pub const fn shift_south(bb: Bitboard) -> Bitboard {
    bb >> 8
}

#[inline(always)]
pub const fn shift_east(bb: Bitboard) -> Bitboard {
    (bb & !FILE_H) << 1
}

#[inline(always)]
pub const fn shift_west(bb: Bitboard) -> Bitboard {
    (bb & !FILE_A) >> 1
}

#[inline(always)]
pub const fn shift_northeast(bb: Bitboard) -> Bitboard {
    (bb & !FILE_H) << 9
}

#[inline(always)]
pub const fn shift_northwest(bb: Bitboard) -> Bitboard {
    (bb & !FILE_A) << 7
}

#[inline(always)]
pub const fn shift_southeast(bb: Bitboard) -> Bitboard {
    (bb & !FILE_H) >> 7
}

#[inline(always)]
pub const fn shift_southwest(bb: Bitboard) -> Bitboard {
    (bb & !FILE_A) >> 9
}

#[inline(always)]
pub const fn get_rank(square: u8) -> u8 {
    SQUARE_TO_RANK[(square & 63) as usize]
}

#[inline(always)]
pub const fn get_file(square: u8) -> u8 {
    SQUARE_TO_FILE[(square & 63) as usize]
}

#[inline(always)]
pub const fn rank_mask(rank: u8) -> Bitboard {
    RANK_1 << ((rank & 7) * 8)
}

#[inline(always)]
pub const fn file_mask(file: u8) -> Bitboard {
    FILE_A << (file & 7)
}

#[inline(always)]
pub fn square_distance(sq1: u8, sq2: u8) -> u8 {
    let rank_diff = (get_rank(sq1) as i8 - get_rank(sq2) as i8).abs() as u8;
    let file_diff = (get_file(sq1) as i8 - get_file(sq2) as i8).abs() as u8;
    rank_diff.max(file_diff)
}

#[inline(always)]
pub fn square_to_algebraic(square: u8) -> String {
    const FILE_CHARS: [char; 8] = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
    let sq = square & 63;
    let file = SQUARE_TO_FILE[sq as usize];
    let rank = SQUARE_TO_RANK[sq as usize];
    
    let mut result = String::with_capacity(2);
    result.push(FILE_CHARS[file as usize]);
    result.push((b'1' + rank) as char);
    result
}

#[inline(always)]
pub fn algebraic_to_square(algebraic: &str) -> Option<u8> {
    let bytes = algebraic.as_bytes();
    if bytes.len() < 2 {
        return None;
    }
    
    let file = bytes[0].wrapping_sub(b'a');
    let rank = bytes[1].wrapping_sub(b'1');
    
    if file < 8 && rank < 8 {
        Some(rank * 8 + file)
    } else {
        None
    }
}

pub struct BitboardIterator {
    bb: Bitboard,
}

impl BitboardIterator {
    #[inline(always)]
    pub const fn new(bb: Bitboard) -> Self {
        Self { bb }
    }
}

impl Iterator for BitboardIterator {
    type Item = u8;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.bb == 0 {
            None
        } else {
            let square = pop_lsb(&mut self.bb);
            Some(square)
        }
    }
}

#[inline(always)]
pub const fn iter_bits(bb: Bitboard) -> BitboardIterator {
    BitboardIterator::new(bb)
}

pub fn print(bb: Bitboard) {
    println!("   +-----------------+");
    for rank in (0..8).rev() {
        print!(" {} |", rank + 1);
        for file in 0..8 {
            let square_idx = rank * 8 + file;
            print!(" {}", if has_bit(bb, square_idx) { "●" } else { "·" });
        }
        println!(" |");
    }
    println!("   +-----------------+");
    println!("     a b c d e f g h");
    println!("   Bitboard: 0x{:016X}", bb);
    println!("   Popcount: {}", popcount(bb));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_mask() {
        assert_eq!(square_mask(0), 1);
        assert_eq!(square_mask(1), 2);
        assert_eq!(square_mask(63), 0x8000000000000000);
    }

    #[test]
    fn test_bit_operations() {
        let mut bb = EMPTY;
        set_bit(&mut bb, 0);
        assert!(has_bit(bb, 0));
        assert_eq!(popcount(bb), 1);
        
        clear_bit(&mut bb, 0);
        assert!(!has_bit(bb, 0));
        assert_eq!(popcount(bb), 0);
    }

    #[test]
    fn test_pop_lsb() {
        let mut bb = 0b1010;
        let index = pop_lsb(&mut bb);
        assert_eq!(index, 1);
        assert_eq!(bb, 0b1000);
    }

    #[test]
    fn test_algebraic_conversion() {
        assert_eq!(algebraic_to_square("a1"), Some(0));
        assert_eq!(algebraic_to_square("h8"), Some(63));
        assert_eq!(square_to_algebraic(0), "a1");
        assert_eq!(square_to_algebraic(63), "h8");
    }

    #[test]
    fn test_shifts() {
        let bb = square_mask(28);
        assert_eq!(shift_north(bb), square_mask(36));
        assert_eq!(shift_south(bb), square_mask(20));
        assert_eq!(shift_east(bb), square_mask(29));
        assert_eq!(shift_west(bb), square_mask(27));
    }

    #[test]
    fn test_lsb_msb() {
        let bb = 0b1010;
        assert_eq!(lsb(bb), 0b10);
        assert_eq!(msb(bb), 0b1000);
        assert_eq!(reset_lsb(bb), 0b1000);
    }
}