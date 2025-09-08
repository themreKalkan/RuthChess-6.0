use crate::board::position::{Position, Color, PieceType};
use crate::eval::evaluate::Score;
use crate::board::bitboard::Bitboard;


const INITIATIVE_BASE: Score = Score::new(18, 7);

const PAWN_ASYMMETRY_BONUS: Score = Score::new(12, 4);

const KING_SAFETY_IMBALANCE: Score = Score::new(15, 8);

const SPACE_ADVANTAGE_BONUS: Score = Score::new(10, 5);

const DEVELOPMENT_BONUS: Score = Score::new(14, 3);

const PIECE_ACTIVITY_BONUS: Score = Score::new(8, 6);

const ATTACK_POTENTIAL: Score = Score::new(20, 10);

const PASSED_PAWN_INITIATIVE: Score = Score::new(16, 12);

pub fn evaluate_initiative(pos: &Position, color: Color) -> Score {
    let mut initiative = Score::zero();
    
    if pos.side_to_move == color {
        initiative = initiative.add(INITIATIVE_BASE);
    }
    
    initiative = initiative.add(evaluate_pawn_asymmetry_initiative(pos, color));
    initiative = initiative.add(evaluate_king_safety_initiative(pos, color));
    initiative = initiative.add(evaluate_space_initiative(pos, color));
    initiative = initiative.add(evaluate_development_initiative(pos, color));
    initiative = initiative.add(evaluate_piece_activity_initiative(pos, color));
    initiative = initiative.add(evaluate_attack_potential_initiative(pos, color));
    initiative = initiative.add(evaluate_passed_pawn_initiative(pos, color));
    
    let complexity_factor = calculate_complexity_factor(pos);
    
    Score::new(
        (initiative.mg as f32 * complexity_factor) as i32,
        (initiative.eg as f32 * complexity_factor * 0.5) as i32,
    )
}

fn evaluate_pawn_asymmetry_initiative(pos: &Position, color: Color) -> Score {
    let white_pawns = pos.pieces_colored(PieceType::Pawn, Color::White);
    let black_pawns = pos.pieces_colored(PieceType::Pawn, Color::Black);
    
    let mut asymmetry_count = 0;
    
    for file in 0..8 {
        let file_mask = 0x0101010101010101u64 << file;
        let white_on_file = (white_pawns & file_mask).count_ones();
        let black_on_file = (black_pawns & file_mask).count_ones();
        
        if white_on_file != black_on_file {
            asymmetry_count += 1;
        }
    }
    
    if asymmetry_count >= 4 {
        PAWN_ASYMMETRY_BONUS
    } else if asymmetry_count >= 2 {
        Score::new(PAWN_ASYMMETRY_BONUS.mg / 2, PAWN_ASYMMETRY_BONUS.eg / 2)
    } else {
        Score::zero()
    }
}

fn evaluate_king_safety_initiative(pos: &Position, color: Color) -> Score {
    use crate::eval::king_safety;
    
    let our_king_safety = king_safety::calculate_king_danger_score(pos, color);
    let enemy_king_safety = king_safety::calculate_king_danger_score(pos, color.opposite());
    
    if enemy_king_safety > our_king_safety + 100 {
        KING_SAFETY_IMBALANCE
    } else if enemy_king_safety > our_king_safety + 50 {
        Score::new(KING_SAFETY_IMBALANCE.mg / 2, KING_SAFETY_IMBALANCE.eg / 2)
    } else {
        Score::zero()
    }
}

fn evaluate_space_initiative(pos: &Position, color: Color) -> Score {
    use crate::eval::space;
    
    let our_space = space::calculate_space_pressure(pos, color);
    let enemy_space = space::calculate_space_pressure(pos, color.opposite());
    
    if our_space > enemy_space + 10 {
        SPACE_ADVANTAGE_BONUS
    } else if our_space > enemy_space + 5 {
        Score::new(SPACE_ADVANTAGE_BONUS.mg / 2, SPACE_ADVANTAGE_BONUS.eg / 2)
    } else {
        Score::zero()
    }
}

fn evaluate_development_initiative(pos: &Position, color: Color) -> Score {
    let phase = crate::eval::material::calculate_phase(pos);
    
    if phase < 128 {
        return Score::zero();
    }
    
    let our_development = count_developed_pieces(pos, color);
    let enemy_development = count_developed_pieces(pos, color.opposite());
    
    if our_development > enemy_development + 2 {
        DEVELOPMENT_BONUS
    } else if our_development > enemy_development {
        Score::new(DEVELOPMENT_BONUS.mg / 2, DEVELOPMENT_BONUS.eg / 2)
    } else {
        Score::zero()
    }
}

fn count_developed_pieces(pos: &Position, color: Color) -> i32 {
    let mut developed = 0;
    
    let knights = pos.pieces_colored(PieceType::Knight, color);
    let back_rank = if color == Color::White { 0xFF } else { 0xFF00000000000000 };
    developed += (knights & !back_rank).count_ones() as i32;
    
    let bishops = pos.pieces_colored(PieceType::Bishop, color);
    developed += (bishops & !back_rank).count_ones() as i32;
    
    let rooks = pos.pieces_colored(PieceType::Rook, color);
    let central_files = 0x1818181818181818u64;
    let seventh_rank = if color == Color::White { 0x00FF000000000000 } else { 0x000000000000FF00 };
    developed += ((rooks & central_files) | (rooks & seventh_rank)).count_ones() as i32;
    
    let king_sq = pos.king_square(color);
    let king_file = king_sq % 8;
    let king_rank = king_sq / 8;
    let expected_rank = if color == Color::White { 0 } else { 7 };
    
    if king_rank == expected_rank && (king_file <= 2 || king_file >= 6) {
        developed += 2;
    }
    
    developed
}

fn evaluate_piece_activity_initiative(pos: &Position, color: Color) -> Score {
    use crate::movegen::magic::{all_attacks};
    
    let our_attacks = all_attacks(pos, color);
    let enemy_attacks = all_attacks(pos, color.opposite());
    
    let enemy_territory = if color == Color::White {
        0xFFFFFFFF00000000u64
    } else {
        0x00000000FFFFFFFFu64
    };
    
    let our_pressure = (our_attacks & enemy_territory).count_ones() as i32;
    let enemy_pressure = (enemy_attacks & !enemy_territory).count_ones() as i32;
    
    if our_pressure > enemy_pressure + 15 {
        PIECE_ACTIVITY_BONUS
    } else if our_pressure > enemy_pressure + 8 {
        Score::new(PIECE_ACTIVITY_BONUS.mg / 2, PIECE_ACTIVITY_BONUS.eg / 2)
    } else {
        Score::zero()
    }
}

fn evaluate_attack_potential_initiative(pos: &Position, color: Color) -> Score {
    use crate::eval::threats;
    
    let our_threats = threats::count_threats(pos, color);
    let enemy_threats = threats::count_threats(pos, color.opposite());
    
    if our_threats > enemy_threats + 3 {
        ATTACK_POTENTIAL
    } else if our_threats > enemy_threats + 1 {
        Score::new(ATTACK_POTENTIAL.mg / 2, ATTACK_POTENTIAL.eg / 2)
    } else {
        Score::zero()
    }
}

fn evaluate_passed_pawn_initiative(pos: &Position, color: Color) -> Score {
    let our_pawns = pos.pieces_colored(PieceType::Pawn, color);
    let enemy_pawns = pos.pieces_colored(PieceType::Pawn, color.opposite());
    
    let our_passed = count_passed_pawns(our_pawns, enemy_pawns, color);
    let enemy_passed = count_passed_pawns(enemy_pawns, our_pawns, color.opposite());
    
    if our_passed > enemy_passed + 1 {
        PASSED_PAWN_INITIATIVE
    } else if our_passed > enemy_passed {
        Score::new(PASSED_PAWN_INITIATIVE.mg / 2, PASSED_PAWN_INITIATIVE.eg / 2)
    } else {
        Score::zero()
    }
}

fn count_passed_pawns(our_pawns: Bitboard, enemy_pawns: Bitboard, color: Color) -> i32 {
    let mut passed_count = 0;
    let mut pawns_bb = our_pawns;
    
    while pawns_bb != 0 {
        let square = pawns_bb.trailing_zeros() as u8;
        pawns_bb &= pawns_bb - 1;
        
        if is_passed_pawn(square, color, enemy_pawns) {
            passed_count += 1;
        }
    }
    
    passed_count
}

fn is_passed_pawn(square: u8, color: Color, enemy_pawns: Bitboard) -> bool {
    let file = square % 8;
    let rank = square / 8;
    
    let mut passed_mask = 0u64;
    
    for f in file.saturating_sub(1)..=(file + 1).min(7) {
        if color == Color::White {
            for r in (rank + 1)..8 {
                passed_mask |= 1u64 << (r * 8 + f);
            }
        } else {
            for r in 0..rank {
                passed_mask |= 1u64 << (r * 8 + f);
            }
        }
    }
    
    (enemy_pawns & passed_mask) == 0
}

fn calculate_complexity_factor(pos: &Position) -> f32 {
    let mut complexity = 1.0;
    
    let white_material = calculate_material_count(pos, Color::White);
    let black_material = calculate_material_count(pos, Color::Black);
    let material_diff = (white_material - black_material).abs();
    
    if material_diff > 300 {
        complexity += 0.2;
    }
    
    let white_pawns = pos.pieces_colored(PieceType::Pawn, Color::White);
    let black_pawns = pos.pieces_colored(PieceType::Pawn, Color::Black);
    
    let mut file_asymmetry = 0;
    for file in 0..8 {
        let file_mask = 0x0101010101010101u64 << file;
        if ((white_pawns & file_mask) != 0) != ((black_pawns & file_mask) != 0) {
            file_asymmetry += 1;
        }
    }
    
    complexity += file_asymmetry as f32 * 0.05;
    
    let queens = pos.pieces_colored(PieceType::Queen, Color::White) |
                pos.pieces_colored(PieceType::Queen, Color::Black);
    if queens != 0 {
        complexity += 0.15;
    }
    
    let white_king = pos.king_square(Color::White);
    let black_king = pos.king_square(Color::Black);
    
    if (white_king % 8 < 3 && black_king % 8 > 4) ||
       (white_king % 8 > 4 && black_king % 8 < 3) {
        complexity += 0.1;
    }
    
    complexity.min(1.5).max(0.5)
}

fn calculate_material_count(pos: &Position, color: Color) -> i32 {
    pos.piece_count(color, PieceType::Pawn) as i32 * 100 +
    pos.piece_count(color, PieceType::Knight) as i32 * 320 +
    pos.piece_count(color, PieceType::Bishop) as i32 * 330 +
    pos.piece_count(color, PieceType::Rook) as i32 * 500 +
    pos.piece_count(color, PieceType::Queen) as i32 * 900
}

pub fn position_favors_initiative(pos: &Position) -> bool {
    let total_pawns = pos.piece_count(Color::White, PieceType::Pawn) +
                     pos.piece_count(Color::Black, PieceType::Pawn);
    
    if total_pawns <= 12 {
        return true;
    }
    
    let white_material = calculate_material_count(pos, Color::White);
    let black_material = calculate_material_count(pos, Color::Black);
    
    if (white_material - black_material).abs() > 200 {
        return true;
    }
    
    let white_king = pos.king_square(Color::White);
    let black_king = pos.king_square(Color::Black);
    
    if (white_king % 8 < 3 && black_king % 8 > 4) ||
       (white_king % 8 > 4 && black_king % 8 < 3) {
        return true;
    }
    
    false
}

pub fn evaluate_initiative_simple(pos: &Position) -> Score {
    let mut initiative = Score::zero();
    
    if pos.side_to_move == Color::White {
        initiative = initiative.add(Score::new(10, 5));
    } else {
        initiative = initiative.sub(Score::new(10, 5));
    }
    
    let white_pawns = pos.pieces_colored(PieceType::Pawn, Color::White);
    let black_pawns = pos.pieces_colored(PieceType::Pawn, Color::Black);
    
    let mut asymmetry = 0;
    for file in 0..8 {
        let file_mask = 0x0101010101010101u64 << file;
        if ((white_pawns & file_mask) != 0) != ((black_pawns & file_mask) != 0) {
            asymmetry += 1;
        }
    }
    
    if asymmetry >= 4 {
        initiative = initiative.add(Score::new(8, 3));
    }
    
    if pos.pieces_colored(PieceType::Queen, Color::White) != 0 ||
       pos.pieces_colored(PieceType::Queen, Color::Black) != 0 {
        initiative = Score::new((initiative.mg * 5) / 4, (initiative.eg * 5) / 4);
    }
    
    initiative
}

pub fn evaluate_tempo(pos: &Position) -> Score {
    let mut tempo = Score::new(10, 5);
    
    if pos.is_in_check(pos.side_to_move.opposite()) {
        tempo = tempo.add(Score::new(15, 8));
    }
    
    let phase = crate::eval::material::calculate_phase(pos);
    if phase > 128 {
        tempo = Score::new((tempo.mg * 5) / 4, tempo.eg);
    }
    
    tempo
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::position::Position;
    
    #[test]
    fn test_initiative_evaluation() {
        let pos = Position::startpos();
        let white_initiative = evaluate_initiative(&pos, Color::White);
        let black_initiative = evaluate_initiative(&pos, Color::Black);
        
        assert!(white_initiative.mg > black_initiative.mg);
    }
    
    #[test]
    fn test_complexity_factor() {
        let pos = Position::startpos();
        let complexity = calculate_complexity_factor(&pos);
        
        assert!(complexity >= 0.8 && complexity <= 1.2);
    }
    
    #[test]
    fn test_tempo_evaluation() {
        let pos = Position::startpos();
        let tempo = evaluate_tempo(&pos);
        
        assert!(tempo.mg > 0);
        assert!(tempo.eg > 0);
    }
    
    #[test]
    fn test_passed_pawn_detection() {
        let white_pawns = 1u64 << 36;
        let black_pawns = 0u64;
        
        assert!(is_passed_pawn(36, Color::White, black_pawns));
        
        let black_pawns = 1u64 << 27;
        assert!(!is_passed_pawn(36, Color::White, black_pawns));
    }
}
