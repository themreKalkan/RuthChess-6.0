
use crate::board::position::{Position, PieceType, Color};
use crate::eval::evaluate::Score;
use crate::board::bitboard::Bitboard;


const KNIGHT_BISHOP_IMBALANCE: [[i32; 9]; 9] = [
    [  0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   1,   3,   3,   3,   3,   3,   3,   3],
    [  0,   1,   4,   4,   4,   4,   4,   4,   4], 
    [  0,   1,   4,   6,   6,   6,   6,   6,   6],
    [  0,   1,   4,   6,   8,   8,   8,   8,   8],
    [  0,   1,   4,   6,   8,  10,  10,  10,  10],
    [  0,   1,   4,   6,   8,  10,  12,  12,  12],
    [  0,   1,   4,   6,   8,  10,  12,  14,  14],
    [  0,   1,   4,   6,   8,  10,  12,  14,  16],
];

const ROOK_MINOR_IMBALANCE: [[i32; 9]; 9] = [
    [  0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,  -1,  -2,  -2,  -2,  -2,  -2,  -2,  -2],
    [  0,  -1,  -3,  -3,  -3,  -3,  -3,  -3,  -3],
    [  0,  -1,  -3,  -5,  -5,  -5,  -5,  -5,  -5],
    [  0,  -1,  -3,  -5,  -7,  -7,  -7,  -7,  -7],
    [  0,  -1,  -3,  -5,  -7,  -9,  -9,  -9,  -9],
    [  0,  -1,  -3,  -5,  -7,  -9, -11, -11, -11],
    [  0,  -1,  -3,  -5,  -7,  -9, -11, -13, -13],
    [  0,  -1,  -3,  -5,  -7,  -9, -11, -13, -15],
];

const QUEEN_IMBALANCE: [[i32; 9]; 9] = [
    [  0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   2,   4,   6,   8,  10,  12,  14,  16],
    [  0,   2,   6,  10,  14,  18,  22,  26,  30],
    [  0,   2,   6,  12,  18,  24,  30,  36,  42],
    [  0,   2,   6,  12,  20,  28,  36,  44,  52],
    [  0,   2,   6,  12,  20,  30,  40,  50,  60],
    [  0,   2,   6,  12,  20,  30,  42,  54,  66],
    [  0,   2,   6,  12,  20,  30,  42,  56,  70],
    [  0,   2,   6,  12,  20,  30,  42,  56,  72],
];

const PAWN_IMBALANCE: [i32; 9] = [0, 3, 3, 1, -3, -3, -8, -8, -8];

pub fn evaluate_imbalance(pos: &Position) -> Score {
    let white_counts = get_piece_counts(pos, Color::White);
    let black_counts = get_piece_counts(pos, Color::Black);
    
    let white_imbalance = calculate_color_imbalance(&white_counts, &black_counts);
    let black_imbalance = calculate_color_imbalance(&black_counts, &white_counts);
    
    let total_imbalance = white_imbalance - black_imbalance;
    
    Score::new(total_imbalance * 16, total_imbalance * 8)
}

fn get_piece_counts(pos: &Position, color: Color) -> [u32; 6] {
    [
        pos.piece_count(color, PieceType::Pawn),
        pos.piece_count(color, PieceType::Knight),
        pos.piece_count(color, PieceType::Bishop),
        pos.piece_count(color, PieceType::Rook),
        pos.piece_count(color, PieceType::Queen),
        pos.piece_count(color, PieceType::King),
    ]
}

fn calculate_color_imbalance(us: &[u32; 6], them: &[u32; 6]) -> i32 {
    let mut bonus = 0;
    
    bonus += PAWN_IMBALANCE[us[0].min(8) as usize] * us[0] as i32;
    
    let our_knights = us[1].min(8) as usize;
    let their_bishops = them[2].min(8) as usize;
    bonus += KNIGHT_BISHOP_IMBALANCE[our_knights][their_bishops] * us[1] as i32;
    
    let our_bishops = us[2].min(8) as usize;
    let their_knights = them[1].min(8) as usize;
    bonus -= KNIGHT_BISHOP_IMBALANCE[their_knights][our_bishops] * us[2] as i32;
    
    let our_rooks = us[3].min(8) as usize;
    let their_minors = (them[1] + them[2]).min(8) as usize;
    bonus += ROOK_MINOR_IMBALANCE[our_rooks][their_minors] * us[3] as i32;
    
    let our_queens = us[4].min(8) as usize;
    let their_pieces = (them[1] + them[2] + them[3]).min(8) as usize;
    bonus += QUEEN_IMBALANCE[our_queens][their_pieces] * us[4] as i32;
    
    bonus += evaluate_specific_imbalances(us, them);
    
    bonus
}

fn evaluate_specific_imbalances(us: &[u32; 6], them: &[u32; 6]) -> i32 {
    let mut bonus = 0;
    
    if us[2] >= 2 && them[1] >= 1 && them[2] >= 1 {
        bonus += 15;
    }
    
    if us[4] >= 1 && (us[1] + us[2]) >= 1 && them[3] >= 2 {
        bonus += 10;
    }
    
    if us[3] >= 1 && (us[1] + us[2]) >= 1 && them[4] >= 1 {
        bonus -= 5;
    }
    
    if (us[1] + us[2]) >= 3 && them[4] >= 1 {
        bonus += 8;
    }
    
    let our_heavy = us[3] + us[4] * 2;
    let their_minors = them[1] + them[2];
    
    if our_heavy >= 2 && their_minors >= 3 {
        bonus -= 5;
    }
    
    if us[2] == 1 && them[2] == 1 {
        bonus += evaluate_opposite_bishops_bonus();
    }
    
    let total_material = (us[1] + us[2] + us[3] + us[4]) + (them[1] + them[2] + them[3] + them[4]);
    if total_material <= 2 {
        bonus += evaluate_kp_endgame_imbalance(us, them);
    }
    
    bonus
}

fn evaluate_opposite_bishops_bonus() -> i32 {
    5
}

fn evaluate_kp_endgame_imbalance(us: &[u32; 6], them: &[u32; 6]) -> i32 {
    let mut bonus = 0;
    
    let pawn_diff = us[0] as i32 - them[0] as i32;
    bonus += pawn_diff * 20;
    
    let minor_diff = (us[1] + us[2]) as i32 - (them[1] + them[2]) as i32;
    bonus += minor_diff * 50;
    
    bonus
}

pub fn calculate_compensation(pos: &Position, color: Color) -> Score {
    let our_material = calculate_total_material(pos, color);
    let their_material = calculate_total_material(pos, color.opposite());
    
    let material_diff = our_material - their_material;
    
    if material_diff >= 0 {
        return Score::zero();
    }
    
    let deficit = -material_diff;
    let mut compensation = Score::zero();
    
    compensation = compensation.add(evaluate_development_compensation(pos, color, deficit));
    
    compensation = compensation.add(evaluate_initiative_compensation(pos, color, deficit));
    
    compensation = compensation.add(evaluate_positional_compensation(pos, color, deficit));
    
    compensation
}

fn calculate_total_material(pos: &Position, color: Color) -> i32 {
    let mut material = 0;
    
    material += pos.piece_count(color, PieceType::Pawn) as i32 * 100;
    material += pos.piece_count(color, PieceType::Knight) as i32 * 320;
    material += pos.piece_count(color, PieceType::Bishop) as i32 * 330;
    material += pos.piece_count(color, PieceType::Rook) as i32 * 500;
    material += pos.piece_count(color, PieceType::Queen) as i32 * 900;
    
    material
}

fn evaluate_development_compensation(pos: &Position, color: Color, deficit: i32) -> Score {
    let our_developed = count_developed_pieces(pos, color);
    let their_developed = count_developed_pieces(pos, color.opposite());
    
    let development_advantage = our_developed - their_developed;
    let compensation = (development_advantage * deficit) / 200;
    
    Score::new(compensation, 0)
}

fn count_developed_pieces(pos: &Position, color: Color) -> i32 {
    let mut developed = 0;
    
    let knights = pos.pieces_colored(PieceType::Knight, color);
    let bishops = pos.pieces_colored(PieceType::Bishop, color);
    
    let starting_rank = match color {
        Color::White => 0,
        Color::Black => 7,
    };
    
    let starting_mask = 0xFFu64 << (starting_rank * 8);
    
    developed += (knights & !starting_mask).count_ones() as i32;
    developed += (bishops & !starting_mask).count_ones() as i32;
    
    let king_sq = pos.king_square(color);
    let king_file = king_sq % 8;
    
    if king_file <= 2 || king_file >= 6 {
        developed += 1;
    }
    
    developed
}

fn evaluate_initiative_compensation(pos: &Position, color: Color, deficit: i32) -> Score {
    let mut initiative = 0;
    
    if pos.side_to_move == color && pos.is_in_check(color.opposite()) {
        initiative += 1;
    }
    
    let our_attacks = count_total_attacks(pos, color);
    let their_attacks = count_total_attacks(pos, color.opposite());
    
    if our_attacks > their_attacks {
        initiative += (our_attacks - their_attacks) / 10;
    }
    
    let compensation = (initiative * deficit) / 300;
    Score::new(compensation, 0)
}

fn count_total_attacks(pos: &Position, color: Color) -> i32 {
    let mut attacks = 0;
    
    for piece_type in 2..=5 {
        let pieces = pos.pieces_colored(PieceType::from(piece_type), color);
        attacks += pieces.count_ones() as i32 * 5;
    }
    
    attacks
}

fn evaluate_positional_compensation(pos: &Position, color: Color, deficit: i32) -> Score {
    let mut compensation = 0;
    
    let our_pawns = pos.pieces_colored(PieceType::Pawn, color);
    let their_pawns = pos.pieces_colored(PieceType::Pawn, color.opposite());
    
    if count_pawn_islands(our_pawns) < count_pawn_islands(their_pawns) {
        compensation += deficit / 20;
    }
    
    if pos.is_in_check(color.opposite()) && !pos.is_in_check(color) {
        compensation += deficit / 15;
    }
    
    Score::new(compensation / 2, compensation)
}

fn count_pawn_islands(pawns: Bitboard) -> i32 {
    let mut islands = 0;
    let mut in_island = false;
    
    for file in 0..8 {
        let file_pawns = pawns & (0x0101010101010101u64 << file);
        
        if file_pawns != 0 {
            if !in_island {
                islands += 1;
                in_island = true;
            }
        } else {
            in_island = false;
        }
    }
    
    islands
}

pub fn has_drawing_tendency(pos: &Position) -> bool {
    let white_counts = get_piece_counts(pos, Color::White);
    let black_counts = get_piece_counts(pos, Color::Black);
    
    if white_counts[2] == 1 && black_counts[2] == 1 {
        let total_pawns = white_counts[0] + black_counts[0];
        if total_pawns <= 4 {
            return true;
        }
    }
    
    let white_material = white_counts[1] + white_counts[2] + white_counts[3] * 2 + white_counts[4] * 4;
    let black_material = black_counts[1] + black_counts[2] + black_counts[3] * 2 + black_counts[4] * 4;
    
    if white_material <= 3 && black_material <= 3 {
        let total_pawns = white_counts[0] + black_counts[0];
        if total_pawns <= 6 {
            return true;
        }
    }
    
    false
}

pub fn calculate_trade_value(us: &[u32; 6], them: &[u32; 6], piece_type: PieceType) -> i32 {
    let piece_idx = piece_type as usize;
    if piece_idx == 0 || piece_idx > 5 {
        return 0;
    }
    
    let mut value = 0;
    
    let base_values = [0, 100, 320, 330, 500, 900, 0];
    value += base_values[piece_idx];
    
    match piece_type {
        PieceType::Knight => {
            value += (us[0] as i32 - 8) * 5;
        }
        PieceType::Bishop => {
            value -= (us[0] as i32 - 8) * 3;
            
            if us[2] >= 2 {
                value += 30;
            }
        }
        PieceType::Rook => {
            let total_material = (us[1] + us[2] + us[3] + us[4]) + 
                               (them[1] + them[2] + them[3] + them[4]);
            value += (16 - total_material as i32) * 5;
        }
        PieceType::Queen => {
            let total_material = (us[1] + us[2] + us[3] + us[4]) + 
                               (them[1] + them[2] + them[3] + them[4]);
            value -= (16 - total_material as i32) * 3;
        }
        _ => {}
    }
    
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::position::Position;
    
    #[test]
    fn test_imbalance_evaluation() {
        let pos = Position::startpos();
        let imbalance = evaluate_imbalance(&pos);
        
        assert_eq!(imbalance.mg, 0);
        assert_eq!(imbalance.eg, 0);
    }
    
    #[test]
    fn test_piece_counts() {
        let pos = Position::startpos();
        let white_counts = get_piece_counts(&pos, Color::White);
        let black_counts = get_piece_counts(&pos, Color::Black);
        
        assert_eq!(white_counts, [8, 2, 2, 2, 1, 1]);
        assert_eq!(black_counts, [8, 2, 2, 2, 1, 1]);
    }
    
    #[test]
    fn test_compensation_calculation() {
        let pos = Position::startpos();
        let compensation = calculate_compensation(&pos, Color::White);
        
        assert_eq!(compensation.mg, 0);
        assert_eq!(compensation.eg, 0);
    }
    
    #[test]
    fn test_drawing_tendency() {
        let pos = Position::startpos();
        assert!(!has_drawing_tendency(&pos));
        
        let pos = Position::from_fen("8/8/8/8/8/8/1b6/B7 w - - 0 1").unwrap();
        assert!(has_drawing_tendency(&pos));
    }
    
    #[test]
    fn test_pawn_islands() {
        let pawns = 0x0101000000000101u64;
        assert_eq!(count_pawn_islands(pawns), 2);
        
        let pawns = 0x0303000000000000u64;
        assert_eq!(count_pawn_islands(pawns), 1);
    }
}