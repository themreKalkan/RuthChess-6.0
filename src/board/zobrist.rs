
use super::bitboard::Bitboard;

#[repr(C, align(64))]
pub struct ZobristKeys {
    pub pieces: [[u64; 64]; 12],
    pub en_passant: [u64; 8],
    pub castling: [u64; 16],
    pub side_to_move: u64,
}

pub const CASTLE_WK: u8 = 1;
pub const CASTLE_WQ: u8 = 2;
pub const CASTLE_BK: u8 = 4;
pub const CASTLE_BQ: u8 = 8;

pub static mut ZOBRIST: ZobristKeys = ZobristKeys {
    pieces: [[0; 64]; 12],
    en_passant: [0; 8],
    castling: [0; 16],
    side_to_move: 0,
};

pub const WHITE_PAWN: usize = 0;
pub const WHITE_KNIGHT: usize = 1;
pub const WHITE_BISHOP: usize = 2;
pub const WHITE_ROOK: usize = 3;
pub const WHITE_QUEEN: usize = 4;
pub const WHITE_KING: usize = 5;
pub const BLACK_PAWN: usize = 6;
pub const BLACK_KNIGHT: usize = 7;
pub const BLACK_BISHOP: usize = 8;
pub const BLACK_ROOK: usize = 9;
pub const BLACK_QUEEN: usize = 10;
pub const BLACK_KING: usize = 11;

struct FastRng {
    state: u64,
}

impl FastRng {
    #[inline(always)]
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline(always)]
    fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
    
    #[inline(always)]
    fn next_batch(&mut self, count: usize) -> impl Iterator<Item = u64> + '_ {
        (0..count).map(|_| self.next())
    }
}

pub fn init_zobrist() {
    let mut rng = FastRng::new(0x1234567890ABCDEF);
    
    unsafe {
        for piece_idx in 0..12 {
            for square_idx in 0..64 {
                ZOBRIST.pieces[piece_idx][square_idx] = rng.next();
            }
        }
        
        for (file_idx, value) in rng.next_batch(8).enumerate() {
            ZOBRIST.en_passant[file_idx] = value;
        }
        
        for (rights_idx, value) in rng.next_batch(16).enumerate() {
            ZOBRIST.castling[rights_idx] = value;
        }
        
        ZOBRIST.side_to_move = rng.next();
    }
}

#[inline(always)]
pub const fn piece_to_zobrist_index(piece: u8, is_white: bool) -> usize {
    let base = match piece {
        1 => 0,
        2 => 1,
        3 => 2,
        4 => 3,
        5 => 4,
        6 => 5,
        _ => 0,
    };
    
    base + if is_white { 0 } else { 6 }
}

#[inline(always)]
pub fn toggle_piece_hash(hash: &mut u64, piece_index: usize, square: u8) {
    unsafe {
        *hash ^= ZOBRIST.pieces[piece_index][square as usize];
    }
}

#[inline(always)]
pub fn toggle_en_passant_hash(hash: &mut u64, file: u8) {
    unsafe {
        *hash ^= ZOBRIST.en_passant[file as usize];
    }
}

#[inline(always)]
pub fn toggle_castling_hash(hash: &mut u64, old_rights: u8, new_rights: u8) {
    unsafe {
        *hash ^= ZOBRIST.castling[old_rights as usize];
        *hash ^= ZOBRIST.castling[new_rights as usize];
    }
}

#[inline(always)]
pub fn toggle_side_to_move_hash(hash: &mut u64) {
    unsafe {
        *hash ^= ZOBRIST.side_to_move;
    }
}

#[inline(always)]
pub fn get_piece_hash(piece_index: usize, square: u8) -> u64 {
    unsafe { ZOBRIST.pieces[piece_index][square as usize] }
}

#[inline(always)]
pub fn get_en_passant_hash(file: u8) -> u64 {
    unsafe { ZOBRIST.en_passant[file as usize] }
}

#[inline(always)]
pub fn get_castling_hash(rights: u8) -> u64 {
    unsafe { ZOBRIST.castling[rights as usize] }
}

#[inline(always)]
pub fn get_side_to_move_hash() -> u64 {
    unsafe { ZOBRIST.side_to_move }
}