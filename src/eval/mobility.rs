
use crate::board::position::{Position, PieceType, Color};
use crate::board::bitboard::{Bitboard, EMPTY};
use crate::eval::evaluate::Score;
use crate::movegen::magic::{
    get_knight_attacks, get_bishop_attacks, get_rook_attacks, 
    get_queen_attacks, get_king_attacks, get_pawn_attacks
};
use std::sync::Once;
use crate::eval::king_safety::flip_board;


const KNIGHT_MOBILITY: [Score; 9] = [
    Score::new(-25, -25),
    Score::new(-15, -15),
    Score::new(-8, -8),
    Score::new(0, 0),
    Score::new(8, 8),
    Score::new(15, 15),
    Score::new(20, 20),
    Score::new(25, 25),
    Score::new(28, 28),
];

const BISHOP_MOBILITY: [Score; 14] = [
    Score::new(-25, -25), Score::new(-15, -15), Score::new(-10, -10), Score::new(-5, -5),
    Score::new(0, 0),     Score::new(5, 5),     Score::new(10, 10),   Score::new(15, 15),
    Score::new(18, 18),   Score::new(20, 20),   Score::new(22, 22),   Score::new(24, 24),
    Score::new(25, 25),   Score::new(26, 26),
];

const ROOK_MOBILITY: [Score; 15] = [
    Score::new(-20, -30), Score::new(-12, -20), Score::new(-8, -15),  Score::new(-4, -10),
    Score::new(-2, -5),   Score::new(0, 0),     Score::new(2, 5),     Score::new(4, 8),
    Score::new(6, 10),    Score::new(8, 12),    Score::new(10, 14),   Score::new(12, 16),
    Score::new(14, 18),   Score::new(16, 20),   Score::new(18, 22),
];

const QUEEN_MOBILITY: [Score; 28] = [
    Score::new(-20, -30), Score::new(-15, -25), Score::new(-10, -20), Score::new(-8, -15),
    Score::new(-6, -12),  Score::new(-4, -8),   Score::new(-2, -4),   Score::new(0, 0),
    Score::new(2, 2),     Score::new(3, 4),     Score::new(4, 6),     Score::new(5, 8),
    Score::new(6, 10),    Score::new(7, 12),    Score::new(8, 14),    Score::new(9, 16),
    Score::new(10, 18),   Score::new(11, 20),   Score::new(12, 22),   Score::new(13, 24),
    Score::new(14, 26),   Score::new(15, 28),   Score::new(16, 30),   Score::new(17, 32),
    Score::new(18, 34),   Score::new(19, 36),   Score::new(20, 38),   Score::new(21, 40),
];

#[derive(Debug, Clone, Copy)]
struct MobilityEntry {
    hash: u64,
    white_mobility: Score,
    black_mobility: Score,
}

impl Default for MobilityEntry {
    fn default() -> Self {
        Self {
            hash: 0,
            white_mobility: Score::zero(),
            black_mobility: Score::zero(),
        }
    }
}

const MOBILITY_TABLE_SIZE: usize = 65536;
static mut MOBILITY_TABLE: [MobilityEntry; MOBILITY_TABLE_SIZE] = [MobilityEntry {
    hash: 0,
    white_mobility: Score { mg: 0, eg: 0 },
    black_mobility: Score { mg: 0, eg: 0 },
}; MOBILITY_TABLE_SIZE];

static MOBILITY_INIT: Once = Once::new();

pub fn init_mobility_tables() {
    MOBILITY_INIT.call_once(|| {
        unsafe {
            let table_ptr = std::ptr::addr_of_mut!(MOBILITY_TABLE);
            for i in 0..MOBILITY_TABLE_SIZE {
                (*table_ptr)[i] = MobilityEntry::default();
            }
        }
    });
}

#[inline(always)]
fn calculate_mobility_hash(pos: &Position) -> u64 {
    let mut hash = 0u64;
    
    hash ^= pos.all_pieces();
    
    hash ^= pos.pieces_colored(PieceType::Knight, Color::White).wrapping_mul(0x9E3779B97F4A7C15);
    hash ^= pos.pieces_colored(PieceType::Knight, Color::Black).wrapping_mul(0x9E3779B97F4A7C16);
    hash ^= pos.pieces_colored(PieceType::Bishop, Color::White).wrapping_mul(0x9E3779B97F4A7C17);
    hash ^= pos.pieces_colored(PieceType::Bishop, Color::Black).wrapping_mul(0x9E3779B97F4A7C18);
    
    hash
}

pub fn evaluate_mobility(pos: &Position) -> Score {
    let mobility_hash = calculate_mobility_hash(pos);
    let index = (mobility_hash as usize) & (MOBILITY_TABLE_SIZE - 1);
    
    unsafe {
        let table_ptr = std::ptr::addr_of_mut!(MOBILITY_TABLE);
        let entry = &mut (*table_ptr)[index];
        
        if entry.hash == mobility_hash {
            return entry.white_mobility.sub(entry.black_mobility);
        }
        
        let white_mobility = calculate_color_mobility(pos, Color::White);
        let black_mobility = calculate_color_mobility(pos, Color::Black);
        
        entry.hash = mobility_hash;
        entry.white_mobility = white_mobility;
        entry.black_mobility = black_mobility;
        
        white_mobility.sub(black_mobility)
    }
}

#[inline(always)]
fn calculate_color_mobility(pos: &Position, color: Color) -> Score {
    let mut total_mobility = Score::zero();
    let our_pieces = pos.pieces(color);
    let enemy_pieces = pos.pieces(color.opposite());
    let all_pieces = pos.all_pieces();
    
    total_mobility = total_mobility.add(calculate_knight_mobility(pos, color, our_pieces));
    total_mobility = total_mobility.add(calculate_bishop_mobility(pos, color, our_pieces, all_pieces));
    total_mobility = total_mobility.add(calculate_rook_mobility(pos, color, our_pieces, all_pieces));
    total_mobility = total_mobility.add(calculate_queen_mobility(pos, color, our_pieces, all_pieces));
    
    total_mobility = total_mobility.add(evaluate_piece_coordination(pos, color));
    total_mobility = total_mobility.add(evaluate_trapped_pieces(pos, color));
    
    total_mobility
}

#[inline(always)]
fn calculate_knight_mobility(pos: &Position, color: Color, our_pieces: Bitboard) -> Score {
    let mut mobility_score = Score::zero();
    let mut knights = pos.pieces_colored(PieceType::Knight, color);
    
    while knights != 0 {
        let square = knights.trailing_zeros() as u8;
        knights &= knights - 1;
        
        let attacks = get_knight_attacks(square);
        let moves = (attacks & !our_pieces).count_ones() as usize;
        
        if moves < KNIGHT_MOBILITY.len() {
            mobility_score = mobility_score.add(KNIGHT_MOBILITY[moves]);
        } else {
            mobility_score = mobility_score.add(KNIGHT_MOBILITY[KNIGHT_MOBILITY.len() - 1]);
        }
    }
    
    mobility_score
}

#[inline(always)]
fn calculate_bishop_mobility(pos: &Position, color: Color, our_pieces: Bitboard, all_pieces: Bitboard) -> Score {
    let mut mobility_score = Score::zero();
    let mut bishops = pos.pieces_colored(PieceType::Bishop, color);
    
    while bishops != 0 {
        let square = bishops.trailing_zeros() as u8;
        bishops &= bishops - 1;
        
        let attacks = get_bishop_attacks(square, all_pieces);
        let moves = (attacks & !our_pieces).count_ones() as usize;
        
        if moves < BISHOP_MOBILITY.len() {
            mobility_score = mobility_score.add(BISHOP_MOBILITY[moves]);
        } else {
            mobility_score = mobility_score.add(BISHOP_MOBILITY[BISHOP_MOBILITY.len() - 1]);
        }
    }
    
    mobility_score
}

#[inline(always)]
fn calculate_rook_mobility(pos: &Position, color: Color, our_pieces: Bitboard, all_pieces: Bitboard) -> Score {
    let mut mobility_score = Score::zero();
    let mut rooks = pos.pieces_colored(PieceType::Rook, color);
    
    while rooks != 0 {
        let square = rooks.trailing_zeros() as u8;
        rooks &= rooks - 1;
        
        let attacks = get_rook_attacks(square, all_pieces);
        let moves = (attacks & !our_pieces).count_ones() as usize;
        
        if moves < ROOK_MOBILITY.len() {
            mobility_score = mobility_score.add(ROOK_MOBILITY[moves]);
        } else {
            mobility_score = mobility_score.add(ROOK_MOBILITY[ROOK_MOBILITY.len() - 1]);
        }
    }
    
    mobility_score
}

#[inline(always)]
fn calculate_queen_mobility(pos: &Position, color: Color, our_pieces: Bitboard, all_pieces: Bitboard) -> Score {
    let mut mobility_score = Score::zero();
    let mut queens = pos.pieces_colored(PieceType::Queen, color);
    
    while queens != 0 {
        let square = queens.trailing_zeros() as u8;
        queens &= queens - 1;
        
        let attacks = get_queen_attacks(square, all_pieces);
        let moves = (attacks & !our_pieces).count_ones() as usize;
        
        if moves < QUEEN_MOBILITY.len() {
            mobility_score = mobility_score.add(QUEEN_MOBILITY[moves]);
        } else {
            mobility_score = mobility_score.add(QUEEN_MOBILITY[QUEEN_MOBILITY.len() - 1]);
        }
    }
    
    mobility_score
}

fn evaluate_piece_coordination(pos: &Position, color: Color) -> Score {
    let mut score = Score::zero();
    
    let queens = pos.pieces_colored(PieceType::Queen, color);
    let rooks = pos.pieces_colored(PieceType::Rook, color);
    
    if queens != 0 && rooks != 0 {
        score = score.add(Score::new(15, 10));
    }
    
    let bishops = pos.pieces_colored(PieceType::Bishop, color);
    if bishops.count_ones() >= 2 {
        let mut light_squares = 0;
        let mut dark_squares = 0;
        let mut bishops_bb = bishops;
        
        while bishops_bb != 0 {
            let square = bishops_bb.trailing_zeros() as u8;
            bishops_bb &= bishops_bb - 1;
            
            if ((square / 8 + square % 8) % 2) == 0 {
                dark_squares += 1;
            } else {
                light_squares += 1;
            }
        }
        
        if light_squares > 0 && dark_squares > 0 {
            score = score.add(Score::new(25, 35));
        }
    }
    
    let knights = pos.pieces_colored(PieceType::Knight, color);
    let enemy_pawns = pos.pieces_colored(PieceType::Pawn, color.opposite());
    
    let mut knights_bb = knights;
    while knights_bb != 0 {
        let square = knights_bb.trailing_zeros() as u8;
        knights_bb &= knights_bb - 1;
        
        if is_knight_outpost(square, color, enemy_pawns) {
            score = score.add(Score::new(20, 15));
        }
    }
    
    score
}

fn is_knight_outpost(square: u8, color: Color, enemy_pawns: Bitboard) -> bool {
    let file = square % 8;
    let rank = square / 8;
    
    let enemy_territory = match color {
        Color::White => rank >= 4,
        Color::Black => rank <= 3,
    };
    
    if !enemy_territory {
        return false;
    }
    
    let attack_files = [file.saturating_sub(1), file + 1];
    
    for &attack_file in &attack_files {
        if attack_file >= 8 {
            continue;
        }
        
        let file_pawns = enemy_pawns & (0x0101010101010101u64 << attack_file);
        if file_pawns != 0 {
            let closest_pawn_rank = match color {
                Color::White => file_pawns.trailing_zeros() / 8,
                Color::Black => 7 - (file_pawns.leading_zeros() / 8),
            };
            
            if (closest_pawn_rank as i32 - rank as i32).abs() <= 1 {
                return false;
            }
        }
    }
    
    true
}

fn evaluate_trapped_pieces(pos: &Position, color: Color) -> Score {
    let mut penalty = Score::zero();
    
    let bishops = pos.pieces_colored(PieceType::Bishop, color);
    let all_pieces = pos.all_pieces();
    
    let mut bishops_bb = bishops;
    while bishops_bb != 0 {
        let square = bishops_bb.trailing_zeros() as u8;
        bishops_bb &= bishops_bb - 1;
        
        let mobility = (get_bishop_attacks(square, all_pieces) & !pos.pieces(color)).count_ones();
        if mobility <= 2 {
            penalty = penalty.add(Score::new(-50, -30));
        }
    }
    
    let rooks = pos.pieces_colored(PieceType::Rook, color);
    let mut rooks_bb = rooks;
    
    while rooks_bb != 0 {
        let square = rooks_bb.trailing_zeros() as u8;
        rooks_bb &= rooks_bb - 1;
        
        let mobility = (get_rook_attacks(square, all_pieces) & !pos.pieces(color)).count_ones();
        if mobility <= 3 {
            penalty = penalty.add(Score::new(-40, -25));
        }
    }
    
    penalty
}

pub fn calculate_average_mobility(pos: &Position) -> (f32, f32) {
    let white_pieces = [
        pos.pieces_colored(PieceType::Knight, Color::White),
        pos.pieces_colored(PieceType::Bishop, Color::White),
        pos.pieces_colored(PieceType::Rook, Color::White),
        pos.pieces_colored(PieceType::Queen, Color::White),
    ];
    
    let black_pieces = [
        pos.pieces_colored(PieceType::Knight, Color::Black),
        pos.pieces_colored(PieceType::Bishop, Color::Black),
        pos.pieces_colored(PieceType::Rook, Color::Black),
        pos.pieces_colored(PieceType::Queen, Color::Black),
    ];
    
    let white_mobility = calculate_total_mobility_count(pos, Color::White);
    let black_mobility = calculate_total_mobility_count(pos, Color::Black);
    
    let white_piece_count = white_pieces.iter().map(|bb| bb.count_ones()).sum::<u32>() as f32;
    let black_piece_count = black_pieces.iter().map(|bb| bb.count_ones()).sum::<u32>() as f32;
    
    let white_avg = if white_piece_count > 0.0 { white_mobility as f32 / white_piece_count } else { 0.0 };
    let black_avg = if black_piece_count > 0.0 { black_mobility as f32 / black_piece_count } else { 0.0 };
    
    (white_avg, black_avg)
}

fn calculate_total_mobility_count(pos: &Position, color: Color) -> u32 {
    let our_pieces = pos.pieces(color);
    let all_pieces = pos.all_pieces();
    let mut total = 0;
    
    let mut knights = pos.pieces_colored(PieceType::Knight, color);
    while knights != 0 {
        let square = knights.trailing_zeros() as u8;
        knights &= knights - 1;
        total += (get_knight_attacks(square) & !our_pieces).count_ones();
    }
    
    let mut bishops = pos.pieces_colored(PieceType::Bishop, color);
    while bishops != 0 {
        let square = bishops.trailing_zeros() as u8;
        bishops &= bishops - 1;
        total += (get_bishop_attacks(square, all_pieces) & !our_pieces).count_ones();
    }
    
    let mut rooks = pos.pieces_colored(PieceType::Rook, color);
    while rooks != 0 {
        let square = rooks.trailing_zeros() as u8;
        rooks &= rooks - 1;
        total += (get_rook_attacks(square, all_pieces) & !our_pieces).count_ones();
    }
    
    let mut queens = pos.pieces_colored(PieceType::Queen, color);
    while queens != 0 {
        let square = queens.trailing_zeros() as u8;
        queens &= queens - 1;
        total += (get_queen_attacks(square, all_pieces) & !our_pieces).count_ones();
    }
    
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::position::Position;
    
    #[test]
    fn test_mobility_initialization() {
        init_mobility_tables();
    }
    
    #[test]
    fn test_mobility_evaluation() {
        init_mobility_tables();
        let pos = Position::startpos();
        let score = evaluate_mobility(&pos);
        
        assert!(score.mg.abs() < 100);
        assert!(score.eg.abs() < 100);
    }
    
    #[test]
    fn test_knight_outpost() {
        let enemy_pawns = 0u64;
        assert!(is_knight_outpost(36, Color::White, enemy_pawns));
    }
    
    #[test]
    fn test_average_mobility() {
        init_mobility_tables();
        let pos = Position::startpos();
        let (white_avg, black_avg) = calculate_average_mobility(&pos);
        
        assert!((white_avg - black_avg).abs() < 1.0);
        assert!(white_avg > 0.0 && black_avg > 0.0);
    }
}