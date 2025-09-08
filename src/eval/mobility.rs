use crate::board::position::{Position, PieceType, Color};
use crate::board::bitboard::{Bitboard, EMPTY};
use crate::eval::evaluate::Score;
use crate::movegen::magic::{
    get_knight_attacks, get_bishop_attacks, get_rook_attacks, 
    get_queen_attacks, get_king_attacks, get_pawn_attacks
};
use std::sync::Once;


const KNIGHT_MOBILITY: [Score; 9] = [
    Score::new(-62, -81),
    Score::new(-53, -56),
    Score::new(-12, -30),
    Score::new( -4, -14),
    Score::new(  3,   5),
    Score::new( 13,  19),
    Score::new( 22,  23),
    Score::new( 28,  27),
    Score::new( 33,  33),
];

const BISHOP_MOBILITY: [Score; 14] = [
    Score::new(-48, -59),
    Score::new(-20, -23),
    Score::new( 16,  -3),
    Score::new( 26,  13),
    Score::new( 38,  24),
    Score::new( 51,  42),
    Score::new( 55,  54),
    Score::new( 63,  57),
    Score::new( 63,  65),
    Score::new( 68,  73),
    Score::new( 81,  78),
    Score::new( 81,  86),
    Score::new( 91,  88),
    Score::new( 98,  97),
];

const ROOK_MOBILITY: [Score; 15] = [
    Score::new(-60, -79),
    Score::new(-24, -55),
    Score::new( -7, -32),
    Score::new( -2, -16),
    Score::new(  3,  -4),
    Score::new(  4,   5),
    Score::new( 14,  12),
    Score::new( 18,  18),
    Score::new( 23,  28),
    Score::new( 26,  35),
    Score::new( 27,  43),
    Score::new( 30,  48),
    Score::new( 32,  56),
    Score::new( 33,  60),
    Score::new( 37,  63),
];

const QUEEN_MOBILITY: [Score; 28] = [
    Score::new(-39, -36),
    Score::new(-21, -15),
    Score::new(  3,   8),
    Score::new(  3,  18),
    Score::new( 14,  34),
    Score::new( 22,  44),
    Score::new( 28,  48),
    Score::new( 41,  53),
    Score::new( 43,  57),
    Score::new( 48,  62),
    Score::new( 56,  68),
    Score::new( 60,  74),
    Score::new( 60,  77),
    Score::new( 66,  84),
    Score::new( 67,  94),
    Score::new( 70,  96),
    Score::new( 71, 103),
    Score::new( 73, 107),
    Score::new( 79, 108),
    Score::new( 88, 108),
    Score::new( 88, 108),
    Score::new( 99, 108),
    Score::new(102, 113),
    Score::new(102, 114),
    Score::new(106, 114),
    Score::new(109, 116),
    Score::new(113, 116),
    Score::new(116, 116),
];

const ROOK_ON_FILE: [Score; 4] = [
    Score::new( 20,  7),
    Score::new( 45, 20),
    Score::new( 45, 20),
    Score::new( 45, 20),
];

const ROOK_ON_SEVENTH: Score = Score::new(42, 31);
const ROOK_ON_OPEN_FILE: Score = Score::new(43, 21);
const ROOK_ON_SEMI_OPEN_FILE: Score = Score::new(19, 10);
const ROOK_ON_QUEEN_FILE: Score = Score::new(6, 8);

const BISHOP_PAIR: Score = Score::new(48, 56);
const BISHOP_ON_LONG_DIAGONAL: Score = Score::new(22, 11);
const BISHOP_PAWNS_ON_COLOR: Score = Score::new(-3, -5);

const KNIGHT_OUTPOST: Score = Score::new(30, 21);
const KNIGHT_ON_HOLE: Score = Score::new(17, 10);

const QUEEN_EARLY_DEVELOPMENT: Score = Score::new(-28, -10);

#[derive(Debug, Clone, Copy)]
struct MobilityArea {
    white: Bitboard,
    black: Bitboard,
}

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

fn calculate_mobility_area(pos: &Position) -> MobilityArea {
    let white_pawns = pos.pieces_colored(PieceType::Pawn, Color::White);
    let black_pawns = pos.pieces_colored(PieceType::Pawn, Color::Black);
    
    let white_pawn_attacks = pawn_attacks_bb(black_pawns, Color::Black);
    let black_pawn_attacks = pawn_attacks_bb(white_pawns, Color::White);
    
    let white_blocked_pawns = white_pawns & 0x0000_0000_0000_FF00;
    let white_area = !white_blocked_pawns & !white_pawn_attacks;
    
    let black_blocked_pawns = black_pawns & 0x00FF_0000_0000_0000;
    let black_area = !black_blocked_pawns & !black_pawn_attacks;
    
    MobilityArea {
        white: white_area,
        black: black_area,
    }
}

#[inline(always)]
fn calculate_mobility_hash(pos: &Position) -> u64 {
    let mut hash = 0u64;
    
    hash ^= pos.all_pieces();
    
    for piece_type in 1..6 {
        let pt = PieceType::from(piece_type);
        hash ^= pos.pieces_colored(pt, Color::White).wrapping_mul(0x9E3779B97F4A7C15 + piece_type as u64);
        hash ^= pos.pieces_colored(pt, Color::Black).wrapping_mul(0x9E3779B97F4A7C16 + piece_type as u64);
    }
    
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
        
        let mobility_area = calculate_mobility_area(pos);
        
        let white_mobility = calculate_color_mobility(pos, Color::White, mobility_area.white);
        let black_mobility = calculate_color_mobility(pos, Color::Black, mobility_area.black);
        
        entry.hash = mobility_hash;
        entry.white_mobility = white_mobility;
        entry.black_mobility = black_mobility;
        
        white_mobility.sub(black_mobility)
    }
}

pub fn evaluate_mobility_for_color(pos: &Position, color: Color) -> Score {
    let mobility_area = calculate_mobility_area(pos);
    let area = if color == Color::White { mobility_area.white } else { mobility_area.black };
    calculate_color_mobility(pos, color, area)
}

#[inline(always)]
fn calculate_color_mobility(pos: &Position, color: Color, mobility_area: Bitboard) -> Score {
    let mut total_mobility = Score::zero();
    let our_pieces = pos.pieces(color);
    let enemy_pieces = pos.pieces(color.opposite());
    let all_pieces = pos.all_pieces();
    
    total_mobility = total_mobility.add(calculate_knight_mobility(pos, color, our_pieces, mobility_area));
    
    total_mobility = total_mobility.add(calculate_bishop_mobility(pos, color, our_pieces, all_pieces, mobility_area));
    
    total_mobility = total_mobility.add(calculate_rook_mobility(pos, color, our_pieces, all_pieces, mobility_area));
    
    total_mobility = total_mobility.add(calculate_queen_mobility(pos, color, our_pieces, all_pieces, mobility_area));
    
    total_mobility = total_mobility.add(evaluate_piece_coordination(pos, color));
    
    total_mobility = total_mobility.add(evaluate_special_patterns(pos, color));
    
    total_mobility
}

#[inline(always)]
fn calculate_knight_mobility(pos: &Position, color: Color, our_pieces: Bitboard, mobility_area: Bitboard) -> Score {
    let mut mobility_score = Score::zero();
    let mut knights = pos.pieces_colored(PieceType::Knight, color);
    
    while knights != 0 {
        let square = knights.trailing_zeros() as u8;
        knights &= knights - 1;
        
        let attacks = get_knight_attacks(square);
        
        let mobility = (attacks & !our_pieces & mobility_area).count_ones() as usize;
        
        if mobility < KNIGHT_MOBILITY.len() {
            mobility_score = mobility_score.add(KNIGHT_MOBILITY[mobility]);
        } else {
            mobility_score = mobility_score.add(KNIGHT_MOBILITY[KNIGHT_MOBILITY.len() - 1]);
        }
        
        if is_outpost(pos, square, color) {
            mobility_score = mobility_score.add(KNIGHT_OUTPOST);
            
            if is_supported_by_pawn(pos, square, color) {
                mobility_score = mobility_score.add(Score::new(9, 8));
            }
        }
    }
    
    mobility_score
}

#[inline(always)]
fn calculate_bishop_mobility(pos: &Position, color: Color, our_pieces: Bitboard, all_pieces: Bitboard, mobility_area: Bitboard) -> Score {
    let mut mobility_score = Score::zero();
    let mut bishops = pos.pieces_colored(PieceType::Bishop, color);
    let our_pawns = pos.pieces_colored(PieceType::Pawn, color);
    
    if bishops.count_ones() >= 2 {
        mobility_score = mobility_score.add(BISHOP_PAIR);
    }
    
    while bishops != 0 {
        let square = bishops.trailing_zeros() as u8;
        bishops &= bishops - 1;
        
        let attacks = get_bishop_attacks(square, all_pieces);
        
        let mobility = (attacks & !our_pieces & mobility_area).count_ones() as usize;
        
        if mobility < BISHOP_MOBILITY.len() {
            mobility_score = mobility_score.add(BISHOP_MOBILITY[mobility]);
        } else {
            mobility_score = mobility_score.add(BISHOP_MOBILITY[BISHOP_MOBILITY.len() - 1]);
        }
        
        if is_on_long_diagonal(square) {
            mobility_score = mobility_score.add(BISHOP_ON_LONG_DIAGONAL);
        }
        
        let square_color = ((square / 8 + square % 8) % 2) as u64;
        let pawns_on_color = if square_color == 0 {
            (our_pawns & 0xAA55AA55AA55AA55u64).count_ones()
        } else {
            (our_pawns & 0x55AA55AA55AA55AAu64).count_ones()
        };
        
        mobility_score = mobility_score.add(Score::new(
            BISHOP_PAWNS_ON_COLOR.mg * pawns_on_color as i32,
            BISHOP_PAWNS_ON_COLOR.eg * pawns_on_color as i32
        ));
    }
    
    mobility_score
}

#[inline(always)]
fn calculate_rook_mobility(pos: &Position, color: Color, our_pieces: Bitboard, all_pieces: Bitboard, mobility_area: Bitboard) -> Score {
    let mut mobility_score = Score::zero();
    let mut rooks = pos.pieces_colored(PieceType::Rook, color);
    let our_pawns = pos.pieces_colored(PieceType::Pawn, color);
    let enemy_pawns = pos.pieces_colored(PieceType::Pawn, color.opposite());
    
    while rooks != 0 {
        let square = rooks.trailing_zeros() as u8;
        rooks &= rooks - 1;
        
        let attacks = get_rook_attacks(square, all_pieces);
        
        let mobility = (attacks & !our_pieces & mobility_area).count_ones() as usize;
        
        if mobility < ROOK_MOBILITY.len() {
            mobility_score = mobility_score.add(ROOK_MOBILITY[mobility]);
        } else {
            mobility_score = mobility_score.add(ROOK_MOBILITY[ROOK_MOBILITY.len() - 1]);
        }
        
        let file = square % 8;
        let file_mask = 0x0101010101010101u64 << file;
        
        if (our_pawns & file_mask) == 0 {
            if (enemy_pawns & file_mask) == 0 {
                mobility_score = mobility_score.add(ROOK_ON_OPEN_FILE);
            } else {
                mobility_score = mobility_score.add(ROOK_ON_SEMI_OPEN_FILE);
            }
        }
        
        let rank = square / 8;
        let seventh_rank = if color == Color::White { 6 } else { 1 };
        
        if rank == seventh_rank {
            let enemy_king = pos.king_square(color.opposite());
            let enemy_king_rank = enemy_king / 8;
            let back_rank = if color == Color::White { 7 } else { 0 };
            
            if enemy_king_rank == back_rank || 
               (enemy_pawns & (0xFFu64 << (seventh_rank * 8))) != 0 {
                mobility_score = mobility_score.add(ROOK_ON_SEVENTH);
            }
        }
        
        if file == 3 {
            mobility_score = mobility_score.add(ROOK_ON_QUEEN_FILE);
        }
    }
    
    if rooks.count_ones() >= 2 {
        let rook1 = rooks.trailing_zeros() as u8;
        let rook2 = (rooks & (rooks - 1)).trailing_zeros() as u8;
        
        let rook1_attacks = get_rook_attacks(rook1, all_pieces);
        let rook2_attacks = get_rook_attacks(rook2, all_pieces);
        
        if (rook1_attacks & (1u64 << rook2)) != 0 || (rook2_attacks & (1u64 << rook1)) != 0 {
            mobility_score = mobility_score.add(Score::new(18, 12));
        }
    }
    
    mobility_score
}

#[inline(always)]
fn calculate_queen_mobility(pos: &Position, color: Color, our_pieces: Bitboard, all_pieces: Bitboard, mobility_area: Bitboard) -> Score {
    let mut mobility_score = Score::zero();
    let mut queens = pos.pieces_colored(PieceType::Queen, color);
    
    let phase = crate::eval::material::calculate_phase(pos);
    if phase > 200 && queens != 0 {
        let queen_sq = queens.trailing_zeros() as u8;
        let starting_square = if color == Color::White { 3 } else { 59 };
        
        if queen_sq != starting_square {
            let rank = queen_sq / 8;
            let development_rank = if color == Color::White { rank } else { 7 - rank };
            
            if development_rank >= 3 {
                mobility_score = mobility_score.add(QUEEN_EARLY_DEVELOPMENT);
            }
        }
    }
    
    while queens != 0 {
        let square = queens.trailing_zeros() as u8;
        queens &= queens - 1;
        
        let attacks = get_queen_attacks(square, all_pieces);
        
        let mobility = (attacks & !our_pieces & mobility_area).count_ones() as usize;
        
        if mobility < QUEEN_MOBILITY.len() {
            mobility_score = mobility_score.add(QUEEN_MOBILITY[mobility]);
        } else {
            mobility_score = mobility_score.add(QUEEN_MOBILITY[QUEEN_MOBILITY.len() - 1]);
        }
    }
    
    mobility_score
}

fn evaluate_piece_coordination(pos: &Position, color: Color) -> Score {
    let mut score = Score::zero();
    
    let bishops = pos.pieces_colored(PieceType::Bishop, color);
    let knights = pos.pieces_colored(PieceType::Knight, color);
    
    if bishops.count_ones() > 0 && knights.count_ones() > 0 {
        score = score.add(Score::new(9, 6));
    }
    
    let queens = pos.pieces_colored(PieceType::Queen, color);
    let rooks = pos.pieces_colored(PieceType::Rook, color);
    
    if queens != 0 && rooks != 0 {
        let all_pieces = pos.all_pieces();
        
        let mut queens_bb = queens;
        while queens_bb != 0 {
            let queen_sq = queens_bb.trailing_zeros() as u8;
            queens_bb &= queens_bb - 1;
            
            let queen_attacks = get_queen_attacks(queen_sq, all_pieces);
            
            if (queen_attacks & rooks) != 0 {
                score = score.add(Score::new(12, 8));
            }
        }
    }
    
    score
}

fn evaluate_special_patterns(pos: &Position, color: Color) -> Score {
    let mut score = Score::zero();
    
    let bishops = pos.pieces_colored(PieceType::Bishop, color);
    let all_pieces = pos.all_pieces();
    
    let mut bishops_bb = bishops;
    while bishops_bb != 0 {
        let square = bishops_bb.trailing_zeros() as u8;
        bishops_bb &= bishops_bb - 1;
        
        let mobility = (get_bishop_attacks(square, all_pieces) & !pos.pieces(color)).count_ones();
        
        if mobility <= 2 {
            score = score.add(Score::new(-50, -30));
            
            if (color == Color::White && (square == 0 || square == 7)) ||
               (color == Color::Black && (square == 56 || square == 63)) {
                score = score.add(Score::new(-25, -15));
            }
        }
    }
    
    let rooks = pos.pieces_colored(PieceType::Rook, color);
    let king_sq = pos.king_square(color);
    
    let mut rooks_bb = rooks;
    while rooks_bb != 0 {
        let square = rooks_bb.trailing_zeros() as u8;
        rooks_bb &= rooks_bb - 1;
        
        let mobility = (get_rook_attacks(square, all_pieces) & !pos.pieces(color)).count_ones();
        
        if mobility <= 3 {
            score = score.add(Score::new(-40, -25));
            
            let file_diff = (square % 8) as i8 - (king_sq % 8) as i8;
            if file_diff.abs() <= 1 {
                score = score.add(Score::new(-20, -10));
            }
        }
    }
    
    score
}


fn pawn_attacks_bb(pawns: Bitboard, color: Color) -> Bitboard {
    match color {
        Color::White => {
            let left = (pawns & !0x0101010101010101u64) << 7;
            let right = (pawns & !0x8080808080808080u64) << 9;
            left | right
        },
        Color::Black => {
            let left = (pawns & !0x8080808080808080u64) >> 7;
            let right = (pawns & !0x0101010101010101u64) >> 9;
            left | right
        }
    }
}

fn is_outpost(pos: &Position, square: u8, color: Color) -> bool {
    let file = square % 8;
    let rank = square / 8;
    
    let in_enemy_territory = match color {
        Color::White => rank >= 4,
        Color::Black => rank <= 3,
    };
    
    if !in_enemy_territory {
        return false;
    }
    
    let enemy_pawns = pos.pieces_colored(PieceType::Pawn, color.opposite());
    
    let mut attack_mask = 0u64;
    
    match color {
        Color::White => {
            for r in 0..rank {
                if file > 0 {
                    attack_mask |= 1u64 << (r * 8 + file - 1);
                }
                if file < 7 {
                    attack_mask |= 1u64 << (r * 8 + file + 1);
                }
            }
        },
        Color::Black => {
            for r in (rank + 1)..8 {
                if file > 0 {
                    attack_mask |= 1u64 << (r * 8 + file - 1);
                }
                if file < 7 {
                    attack_mask |= 1u64 << (r * 8 + file + 1);
                }
            }
        }
    }
    
    (enemy_pawns & attack_mask) == 0
}

fn is_supported_by_pawn(pos: &Position, square: u8, color: Color) -> bool {
    let file = square % 8;
    let rank = square / 8;
    
    let our_pawns = pos.pieces_colored(PieceType::Pawn, color);
    
    let support_rank = match color {
        Color::White => if rank > 0 { rank - 1 } else { return false; },
        Color::Black => if rank < 7 { rank + 1 } else { return false; },
    };
    
    let support_squares = if file > 0 { 1u64 << (support_rank * 8 + file - 1) } else { 0 }
                        | if file < 7 { 1u64 << (support_rank * 8 + file + 1) } else { 0 };
    
    (our_pawns & support_squares) != 0
}

fn is_on_long_diagonal(square: u8) -> bool {
    let file = square % 8;
    let rank = square / 8;
    
    if file == rank {
        return true;
    }
    
    if file + rank == 7 {
        return true;
    }
    
    false
}

pub fn calculate_average_mobility(pos: &Position) -> (f32, f32) {
    let mobility_area = calculate_mobility_area(pos);
    
    let white_mobility = calculate_total_mobility_count(pos, Color::White, mobility_area.white);
    let black_mobility = calculate_total_mobility_count(pos, Color::Black, mobility_area.black);
    
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
    
    let white_piece_count = white_pieces.iter().map(|bb| bb.count_ones()).sum::<u32>() as f32;
    let black_piece_count = black_pieces.iter().map(|bb| bb.count_ones()).sum::<u32>() as f32;
    
    let white_avg = if white_piece_count > 0.0 { white_mobility as f32 / white_piece_count } else { 0.0 };
    let black_avg = if black_piece_count > 0.0 { black_mobility as f32 / black_piece_count } else { 0.0 };
    
    (white_avg, black_avg)
}

fn calculate_total_mobility_count(pos: &Position, color: Color, mobility_area: Bitboard) -> u32 {
    let our_pieces = pos.pieces(color);
    let all_pieces = pos.all_pieces();
    let mut total = 0;
    
    let mut knights = pos.pieces_colored(PieceType::Knight, color);
    while knights != 0 {
        let square = knights.trailing_zeros() as u8;
        knights &= knights - 1;
        total += (get_knight_attacks(square) & !our_pieces & mobility_area).count_ones();
    }
    
    let mut bishops = pos.pieces_colored(PieceType::Bishop, color);
    while bishops != 0 {
        let square = bishops.trailing_zeros() as u8;
        bishops &= bishops - 1;
        total += (get_bishop_attacks(square, all_pieces) & !our_pieces & mobility_area).count_ones();
    }
    
    let mut rooks = pos.pieces_colored(PieceType::Rook, color);
    while rooks != 0 {
        let square = rooks.trailing_zeros() as u8;
        rooks &= rooks - 1;
        total += (get_rook_attacks(square, all_pieces) & !our_pieces & mobility_area).count_ones();
    }
    
    let mut queens = pos.pieces_colored(PieceType::Queen, color);
    while queens != 0 {
        let square = queens.trailing_zeros() as u8;
        queens &= queens - 1;
        total += (get_queen_attacks(square, all_pieces) & !our_pieces & mobility_area).count_ones();
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
        let pos = Position::from_fen("8/8/8/3n4/8/8/2P1P3/8 b - - 0 1").unwrap();
        assert!(is_outpost(&pos, 27, Color::Black));
    }
    
    #[test]
    fn test_average_mobility() {
        init_mobility_tables();
        let pos = Position::startpos();
        let (white_avg, black_avg) = calculate_average_mobility(&pos);
        
        assert!((white_avg - black_avg).abs() < 2.0);
        assert!(white_avg > 0.0 && black_avg > 0.0);
    }
    
    #[test]
    fn test_bishop_pair() {
        let pos = Position::from_fen("8/8/8/8/8/8/1BB5/8 w - - 0 1").unwrap();
        let mobility = evaluate_mobility(&pos);
        
        assert!(mobility.mg > 0);
        assert!(mobility.eg > 0);
    }
}
