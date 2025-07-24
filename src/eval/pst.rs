
use crate::board::position::{Position, PieceType, Color};
use crate::eval::evaluate::Score;
use std::sync::Once;

#[repr(align(64))]
pub struct PSTTable {
    pub mg: [i32; 64],
    pub eg: [i32; 64],
}

impl PSTTable {
    const fn new() -> Self {
        Self {
            mg: [0; 64],
            eg: [0; 64],
        }
    }
    
    #[inline(always)]
    pub fn get_score(&self, square: u8) -> Score {
        Score::new(self.mg[square as usize], self.eg[square as usize])
    }
}

pub static mut PST_TABLES: [PSTTable; 7] = [
    PSTTable::new(),
    PSTTable::new(),
    PSTTable::new(),
    PSTTable::new(),
    PSTTable::new(),
    PSTTable::new(),
    PSTTable::new(),
];

static INIT: Once = Once::new();

pub fn init_pst_tables() {
    INIT.call_once(|| unsafe {
        init_pawn_table();
        init_knight_table();
        init_bishop_table();
        init_rook_table();
        init_queen_table();
        init_king_table();
    });
}


const PAWN_MG: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0
];

const PAWN_EG: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
     80,  80,  80,  80,  80,  80,  80,  80,
     50,  50,  50,  50,  50,  50,  50,  50,
     30,  30,  30,  30,  30,  30,  30,  30,
     20,  20,  20,  20,  20,  20,  20,  20,
     10,  10,  10,  10,  10,  10,  10,  10,
     10,  10,  10,  10,  10,  10,  10,  10,
      0,   0,   0,   0,   0,   0,   0,   0
];

const KNIGHT_MG: [i32; 64] = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50
];

const KNIGHT_EG: [i32; 64] = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -20, -30, -30, -20, -40, -50
];

const BISHOP_MG: [i32; 64] = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
];

const BISHOP_EG: [i32; 64] = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
];

const ROOK_MG: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0
];

const ROOK_EG: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0
];

const QUEEN_MG: [i32; 64] = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20
];

const QUEEN_EG: [i32; 64] = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20
];

const KING_MG: [i32; 64] = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20
];

const KING_EG: [i32; 64] = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50
];

unsafe fn init_pawn_table() {
    let table_ptr = std::ptr::addr_of_mut!(PST_TABLES);
    for sq in 0..64 {
        (*table_ptr)[PieceType::Pawn as usize].mg[sq] = PAWN_MG[sq];
        (*table_ptr)[PieceType::Pawn as usize].eg[sq] = PAWN_EG[sq];
    }
}

unsafe fn init_knight_table() {
    let table_ptr = std::ptr::addr_of_mut!(PST_TABLES);
    for sq in 0..64 {
        (*table_ptr)[PieceType::Knight as usize].mg[sq] = KNIGHT_MG[sq];
        (*table_ptr)[PieceType::Knight as usize].eg[sq] = KNIGHT_EG[sq];
    }
}

unsafe fn init_bishop_table() {
    let table_ptr = std::ptr::addr_of_mut!(PST_TABLES);
    for sq in 0..64 {
        (*table_ptr)[PieceType::Bishop as usize].mg[sq] = BISHOP_MG[sq];
        (*table_ptr)[PieceType::Bishop as usize].eg[sq] = BISHOP_EG[sq];
    }
}

unsafe fn init_rook_table() {
    let table_ptr = std::ptr::addr_of_mut!(PST_TABLES);
    for sq in 0..64 {
        (*table_ptr)[PieceType::Rook as usize].mg[sq] = ROOK_MG[sq];
        (*table_ptr)[PieceType::Rook as usize].eg[sq] = ROOK_EG[sq];
    }
}

unsafe fn init_queen_table() {
    let table_ptr = std::ptr::addr_of_mut!(PST_TABLES);
    for sq in 0..64 {
        (*table_ptr)[PieceType::Queen as usize].mg[sq] = QUEEN_MG[sq];
        (*table_ptr)[PieceType::Queen as usize].eg[sq] = QUEEN_EG[sq];
    }
}

unsafe fn init_king_table() {
    let table_ptr = std::ptr::addr_of_mut!(PST_TABLES);
    for sq in 0..64 {
        (*table_ptr)[PieceType::King as usize].mg[sq] = KING_MG[sq];
        (*table_ptr)[PieceType::King as usize].eg[sq] = KING_EG[sq];
    }
}

#[inline(always)]
pub fn get_pst_score(piece_type: PieceType, square: u8, color: Color) -> Score {
    unsafe {
        let table_ptr = std::ptr::addr_of!(PST_TABLES);
        let table = &(*table_ptr)[piece_type as usize];
        let sq = match color {
            Color::White => square,
            Color::Black => square ^ 56,
        };
        table.get_score(sq)
    }
}

#[inline(always)]
pub fn evaluate_pst(pos: &Position) -> Score {
    let mut score = Score::zero();
    
    for piece_type in 1..=6 {
        let piece_type = PieceType::from(piece_type);
        
        let mut white_pieces = pos.pieces_colored(piece_type, Color::White);
        while white_pieces != 0 {
            let square = white_pieces.trailing_zeros() as u8;
            white_pieces &= white_pieces - 1;
            score = score.add(get_pst_score(piece_type, square, Color::White));
        }
        
        let mut black_pieces = pos.pieces_colored(piece_type, Color::Black);
        while black_pieces != 0 {
            let square = black_pieces.trailing_zeros() as u8;
            black_pieces &= black_pieces - 1;
            score = score.sub(get_pst_score(piece_type, square, Color::Black));
        }
    }
    
    score
}

#[inline(always)]
pub fn get_pst_move_delta(piece_type: PieceType, from: u8, to: u8, color: Color) -> Score {
    let from_score = get_pst_score(piece_type, from, color);
    let to_score = get_pst_score(piece_type, to, color);
    to_score.sub(from_score)
}

#[inline(always)]
pub const fn flip_square(square: u8) -> u8 {
    square ^ 56
}

#[inline(always)]
pub const fn rank_to_square(rank: u8, file: u8) -> u8 {
    rank * 8 + file
}

#[inline(always)]
pub fn relative_rank(square: u8, color: Color) -> u8 {
    let rank = square / 8;
    match color {
        Color::White => rank,
        Color::Black => 7 - rank,
    }
}

#[inline(always)]
pub const fn file_of(square: u8) -> u8 {
    square & 7
}

#[inline(always)]
pub const fn rank_of(square: u8) -> u8 {
    square >> 3
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::position::Position;
    
    #[test]
    fn test_pst_initialization() {
        init_pst_tables();
        unsafe {
            assert_ne!(PST_TABLES[PieceType::Pawn as usize].mg[16], 0);
            assert!(PST_TABLES[PieceType::King as usize].mg[6] > PST_TABLES[PieceType::King as usize].mg[4]);
        }
    }
    
    #[test]
    fn test_square_flipping() {
        assert_eq!(flip_square(0), 56);
        assert_eq!(flip_square(7), 63);
        assert_eq!(flip_square(56), 0);
        assert_eq!(flip_square(63), 7);
    }
    
    #[test]
    fn test_pst_evaluation() {
        init_pst_tables();
        let pos = Position::startpos();
        let score = evaluate_pst(&pos);
        
        assert!(score.mg.abs() < 100);
        assert!(score.eg.abs() < 100);
    }
    
    #[test]
    fn test_relative_rank() {
        assert_eq!(relative_rank(0, Color::White), 0);
        assert_eq!(relative_rank(0, Color::Black), 7);
        assert_eq!(relative_rank(56, Color::White), 7);
        assert_eq!(relative_rank(56, Color::Black), 0);
    }
}