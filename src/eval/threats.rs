
use crate::board::position::{Position, PieceType, Color};
use crate::board::bitboard::{Bitboard, EMPTY};
use crate::eval::evaluate::Score;
use crate::eval::pst::{file_of, rank_of};
use crate::movegen::magic::{
    all_attacks, get_knight_attacks, get_bishop_attacks, 
    get_rook_attacks, get_queen_attacks, get_king_attacks, get_pawn_attacks
};

const HANGING_PIECE_PENALTY: [Score; 7] = [
    Score::new(0, 0),
    Score::new(-15, -20),
    Score::new(-40, -50),
    Score::new(-40, -50),
    Score::new(-60, -70),
    Score::new(-100, -120),
    Score::new(-200, -300),
];

const WEAK_PIECE_PENALTY: [Score; 7] = [
    Score::new(0, 0),
    Score::new(-8, -12),
    Score::new(-20, -25),
    Score::new(-20, -25),
    Score::new(-30, -35),
    Score::new(-50, -60),
    Score::new(0, 0),
];

const THREAT_BONUS: [Score; 7] = [
    Score::new(0, 0),
    Score::new(5, 8),
    Score::new(15, 20),
    Score::new(15, 20),
    Score::new(25, 30),
    Score::new(40, 50),
    Score::new(0, 0),
];

const PAWN_THREAT_BONUS: [Score; 7] = [
    Score::new(0, 0),
    Score::new(0, 0),
    Score::new(20, 25),
    Score::new(20, 25),
    Score::new(30, 40),
    Score::new(50, 60),
    Score::new(0, 0),
];

const MINOR_THREAT_MAJOR: Score = Score::new(25, 35);
const ROOK_THREAT_QUEEN: Score = Score::new(35, 45);
const WEAK_SQUARE_PENALTY: Score = Score::new(-10, -15);

pub fn evaluate_threats(pos: &Position) -> Score {
    let white_threats = evaluate_threats_for_color(pos, Color::White);
    let black_threats = evaluate_threats_for_color(pos, Color::Black);
    
    white_threats.sub(black_threats)
}

fn evaluate_threats_for_color(pos: &Position, color: Color) -> Score {
    let mut threat_score = Score::zero();
    
    threat_score = threat_score.add(evaluate_hanging_pieces(pos, color));
    
    threat_score = threat_score.add(evaluate_weak_pieces(pos, color));
    
    threat_score = threat_score.add(evaluate_pawn_threats(pos, color));
    threat_score = threat_score.add(evaluate_piece_threats(pos, color));
    
    threat_score = threat_score.add(evaluate_weak_squares(pos, color));
    
    threat_score = threat_score.add(evaluate_tactical_threats(pos, color));
    
    threat_score
}

fn evaluate_hanging_pieces(pos: &Position, color: Color) -> Score {
    let mut penalty = Score::zero();
    let our_pieces = pos.pieces(color);
    let enemy_attacks = all_attacks(pos, color.opposite());
    let our_attacks = all_attacks(pos, color);
    
    let hanging = our_pieces & enemy_attacks & !our_attacks;
    
    let mut hanging_bb = hanging;
    while hanging_bb != 0 {
        let square = hanging_bb.trailing_zeros() as u8;
        hanging_bb &= hanging_bb - 1;
        
        let (piece_type, _) = pos.piece_at(square);
        penalty = penalty.add(HANGING_PIECE_PENALTY[piece_type as usize]);
    }
    
    penalty
}

fn evaluate_weak_pieces(pos: &Position, color: Color) -> Score {
    let mut penalty = Score::zero();
    
    for piece_type in 1..6 {
        let piece_type = PieceType::from(piece_type);
        let mut pieces = pos.pieces_colored(piece_type, color);
        
        while pieces != 0 {
            let square = pieces.trailing_zeros() as u8;
            pieces &= pieces - 1;
            
            if is_piece_weak(pos, square, piece_type, color) {
                penalty = penalty.add(WEAK_PIECE_PENALTY[piece_type as usize]);
            }
        }
    }
    
    penalty
}

fn is_piece_weak(pos: &Position, square: u8, piece_type: PieceType, color: Color) -> bool {
    let enemy_attacks = is_square_attacked_by_any(pos, square, color.opposite());
    if !enemy_attacks {
        return false;
    }
    
    let defenders = get_attackers(pos, square, color);
    if defenders.is_empty() {
        return true;
    }
    
    let piece_value = get_piece_value(piece_type);
    let cheapest_defender = find_cheapest_defender(pos, &defenders);
    
    match cheapest_defender {
        Some(defender_type) => get_piece_value(defender_type) > piece_value,
        None => true,
    }
}

fn get_piece_value(piece_type: PieceType) -> i32 {
    match piece_type {
        PieceType::Pawn => 100,
        PieceType::Knight => 320,
        PieceType::Bishop => 330,
        PieceType::Rook => 500,
        PieceType::Queen => 900,
        PieceType::King => 10000,
        _ => 0,
    }
}

fn get_attackers(pos: &Position, square: u8, color: Color) -> Vec<u8> {
    let mut attackers = Vec::new();
    
    for piece_type in 1..6 {
        let piece_type = PieceType::from(piece_type);
        let mut pieces = pos.pieces_colored(piece_type, color);
        
        while pieces != 0 {
            let attacker_sq = pieces.trailing_zeros() as u8;
            pieces &= pieces - 1;
            
            if can_attack(pos, attacker_sq, square, piece_type) {
                attackers.push(attacker_sq);
            }
        }
    }
    
    attackers
}

fn can_attack(pos: &Position, from: u8, to: u8, piece_type: PieceType) -> bool {
    let attacks = match piece_type {
        PieceType::Pawn => {
            let (_, color) = pos.piece_at(from);
            get_pawn_attacks(from, color)
        }
        PieceType::Knight => get_knight_attacks(from),
        PieceType::Bishop => get_bishop_attacks(from, pos.all_pieces()),
        PieceType::Rook => get_rook_attacks(from, pos.all_pieces()),
        PieceType::Queen => get_queen_attacks(from, pos.all_pieces()),
        PieceType::King => get_king_attacks(from),
        _ => return false,
    };
    
    (attacks & (1u64 << to)) != 0
}

fn find_cheapest_defender(pos: &Position, attackers: &[u8]) -> Option<PieceType> {
    let mut cheapest = None;
    let mut cheapest_value = i32::MAX;
    
    for &attacker in attackers {
        let (piece_type, _) = pos.piece_at(attacker);
        let value = get_piece_value(piece_type);
        
        if value < cheapest_value {
            cheapest_value = value;
            cheapest = Some(piece_type);
        }
    }
    
    cheapest
}

fn evaluate_pawn_threats(pos: &Position, color: Color) -> Score {
    let mut threat_score = Score::zero();
    let our_pawns = pos.pieces_colored(PieceType::Pawn, color);
    let enemy_pieces = pos.pieces(color.opposite());
    
    let mut pawns_bb = our_pawns;
    while pawns_bb != 0 {
        let square = pawns_bb.trailing_zeros() as u8;
        pawns_bb &= pawns_bb - 1;
        
        let pawn_attacks = get_pawn_attacks(square, color);
        let threatened_pieces = pawn_attacks & enemy_pieces;
        
        let mut threatened_bb = threatened_pieces;
        while threatened_bb != 0 {
            let threatened_sq = threatened_bb.trailing_zeros() as u8;
            threatened_bb &= threatened_bb - 1;
            
            let (piece_type, _) = pos.piece_at(threatened_sq);
            threat_score = threat_score.add(PAWN_THREAT_BONUS[piece_type as usize]);
        }
    }
    
    threat_score
}

fn evaluate_piece_threats(pos: &Position, color: Color) -> Score {
    let mut threat_score = Score::zero();
    
    let minors = pos.pieces_colored(PieceType::Knight, color) | 
                pos.pieces_colored(PieceType::Bishop, color);
    let enemy_majors = pos.pieces_colored(PieceType::Rook, color.opposite()) |
                      pos.pieces_colored(PieceType::Queen, color.opposite());
    
    let mut minors_bb = minors;
    while minors_bb != 0 {
        let square = minors_bb.trailing_zeros() as u8;
        minors_bb &= minors_bb - 1;
        
        let (piece_type, _) = pos.piece_at(square);
        let attacks = match piece_type {
            PieceType::Knight => get_knight_attacks(square),
            PieceType::Bishop => get_bishop_attacks(square, pos.all_pieces()),
            _ => continue,
        };
        
        if (attacks & enemy_majors) != 0 {
            threat_score = threat_score.add(MINOR_THREAT_MAJOR);
        }
    }
    
    let rooks = pos.pieces_colored(PieceType::Rook, color);
    let enemy_queens = pos.pieces_colored(PieceType::Queen, color.opposite());
    
    let mut rooks_bb = rooks;
    while rooks_bb != 0 {
        let square = rooks_bb.trailing_zeros() as u8;
        rooks_bb &= rooks_bb - 1;
        
        let attacks = get_rook_attacks(square, pos.all_pieces());
        if (attacks & enemy_queens) != 0 {
            threat_score = threat_score.add(ROOK_THREAT_QUEEN);
        }
    }
    
    threat_score
}

fn evaluate_weak_squares(pos: &Position, color: Color) -> Score {
    let mut penalty = Score::zero();
    let our_pieces = pos.pieces(color);
    let enemy_attacks = all_attacks(pos, color.opposite());
    let our_pawn_attacks = get_all_pawn_attacks(pos, color);
    
    let weak_squares = enemy_attacks & !our_pawn_attacks;
    
    let mut pieces_bb = our_pieces;
    while pieces_bb != 0 {
        let square = pieces_bb.trailing_zeros() as u8;
        pieces_bb &= pieces_bb - 1;
        
        let adjacent_squares = get_king_attacks(square);
        let weak_adjacent = adjacent_squares & weak_squares;
        
        let weak_count = weak_adjacent.count_ones() as i32;
        penalty = penalty.add(Score::new(
            weak_count * WEAK_SQUARE_PENALTY.mg,
            weak_count * WEAK_SQUARE_PENALTY.eg,
        ));
    }
    
    penalty
}

fn get_all_pawn_attacks(pos: &Position, color: Color) -> Bitboard {
    let pawns = pos.pieces_colored(PieceType::Pawn, color);
    let mut all_attacks = 0u64;
    
    let mut pawns_bb = pawns;
    while pawns_bb != 0 {
        let square = pawns_bb.trailing_zeros() as u8;
        pawns_bb &= pawns_bb - 1;
        
        all_attacks |= get_pawn_attacks(square, color);
    }
    
    all_attacks
}

fn evaluate_tactical_threats(pos: &Position, color: Color) -> Score {
    let mut threat_score = Score::zero();
    
    threat_score = threat_score.add(evaluate_pins(pos, color));
    threat_score = threat_score.add(evaluate_forks(pos, color));
    threat_score = threat_score.add(evaluate_skewers(pos, color));
    
    threat_score
}

fn evaluate_pins(pos: &Position, color: Color) -> Score {
    let mut pin_bonus = Score::zero();
    let enemy_king = pos.king_square(color.opposite());
    
    let our_bishops = pos.pieces_colored(PieceType::Bishop, color);
    let our_rooks = pos.pieces_colored(PieceType::Rook, color);
    let our_queens = pos.pieces_colored(PieceType::Queen, color);
    
    let diagonal_pieces = our_bishops | our_queens;
    let mut diag_bb = diagonal_pieces;
    while diag_bb != 0 {
        let square = diag_bb.trailing_zeros() as u8;
        diag_bb &= diag_bb - 1;
        
        let attacks = get_bishop_attacks(square, pos.all_pieces());
        if (attacks & (1u64 << enemy_king)) != 0 {
            let pinned_piece = find_pinned_piece(pos, square, enemy_king, true);
            if let Some(pinned_sq) = pinned_piece {
                let (piece_type, _) = pos.piece_at(pinned_sq);
                pin_bonus = pin_bonus.add(Score::new(
                    get_piece_value(piece_type) / 10,
                    get_piece_value(piece_type) / 15,
                ));
            }
        }
    }
    
    let straight_pieces = our_rooks | our_queens;
    let mut straight_bb = straight_pieces;
    while straight_bb != 0 {
        let square = straight_bb.trailing_zeros() as u8;
        straight_bb &= straight_bb - 1;
        
        let attacks = get_rook_attacks(square, pos.all_pieces());
        if (attacks & (1u64 << enemy_king)) != 0 {
            let pinned_piece = find_pinned_piece(pos, square, enemy_king, false);
            if let Some(pinned_sq) = pinned_piece {
                let (piece_type, _) = pos.piece_at(pinned_sq);
                pin_bonus = pin_bonus.add(Score::new(
                    get_piece_value(piece_type) / 10,
                    get_piece_value(piece_type) / 15,
                ));
            }
        }
    }
    
    pin_bonus
}

fn find_pinned_piece(pos: &Position, attacker: u8, target: u8, is_diagonal: bool) -> Option<u8> {
    let file_diff = (target % 8) as i8 - (attacker % 8) as i8;
    let rank_diff = (target / 8) as i8 - (attacker / 8) as i8;
    
    let (step_file, step_rank) = if is_diagonal {
        (file_diff.signum(), rank_diff.signum())
    } else {
        if file_diff == 0 {
            (0, rank_diff.signum())
        } else {
            (file_diff.signum(), 0)
        }
    };
    
    let mut current_file = (attacker % 8) as i8 + step_file;
    let mut current_rank = (attacker / 8) as i8 + step_rank;
    let mut pieces_found = 0;
    let mut potential_pin = None;
    
    while current_file >= 0 && current_file < 8 && current_rank >= 0 && current_rank < 8 {
        let current_square = (current_rank * 8 + current_file) as u8;
        
        if current_square == target {
            break;
        }
        
        let (piece_type, _) = pos.piece_at(current_square);
        if piece_type != PieceType::None {
            pieces_found += 1;
            if pieces_found == 1 {
                potential_pin = Some(current_square);
            } else {
                return None;
            }
        }
        
        current_file += step_file;
        current_rank += step_rank;
    }
    
    if pieces_found == 1 {
        potential_pin
    } else {
        None
    }
}

fn evaluate_forks(pos: &Position, color: Color) -> Score {
    let mut fork_bonus = Score::zero();
    
    let knights = pos.pieces_colored(PieceType::Knight, color);
    let enemy_pieces = pos.pieces(color.opposite());
    let enemy_king = 1u64 << pos.king_square(color.opposite());
    let enemy_valuable = pos.pieces_colored(PieceType::Queen, color.opposite()) |
                        pos.pieces_colored(PieceType::Rook, color.opposite());
    
    let mut knights_bb = knights;
    while knights_bb != 0 {
        let square = knights_bb.trailing_zeros() as u8;
        knights_bb &= knights_bb - 1;
        
        let attacks = get_knight_attacks(square);
        let attacked_pieces = attacks & enemy_pieces;
        
        let attacked_count = attacked_pieces.count_ones();
        if attacked_count >= 2 {
            let mut fork_value = attacked_count as i32 * 10;
            
            if (attacks & enemy_king) != 0 {
                fork_value += 30;
            }
            
            if (attacks & enemy_valuable) != 0 {
                fork_value += 20;
            }
            
            fork_bonus = fork_bonus.add(Score::new(fork_value, fork_value / 2));
        }
    }
    
    fork_bonus
}

fn evaluate_skewers(pos: &Position, color: Color) -> Score {
    let mut skewer_bonus = Score::zero();
    
    let sliding_pieces = pos.pieces_colored(PieceType::Bishop, color) |
                        pos.pieces_colored(PieceType::Rook, color) |
                        pos.pieces_colored(PieceType::Queen, color);
    
    let enemy_king = pos.king_square(color.opposite());
    let enemy_valuable = pos.pieces_colored(PieceType::Queen, color.opposite()) |
                        pos.pieces_colored(PieceType::Rook, color.opposite());
    
    let mut sliders_bb = sliding_pieces;
    while sliders_bb != 0 {
        let square = sliders_bb.trailing_zeros() as u8;
        sliders_bb &= sliders_bb - 1;
        
        let (piece_type, _) = pos.piece_at(square);
        let attacks = match piece_type {
            PieceType::Bishop => get_bishop_attacks(square, pos.all_pieces()),
            PieceType::Rook => get_rook_attacks(square, pos.all_pieces()),
            PieceType::Queen => get_queen_attacks(square, pos.all_pieces()),
            _ => continue,
        };
        
        if (attacks & (1u64 << enemy_king)) != 0 {
            if let Some(skewered_value) = find_skewered_piece_value(pos, square, enemy_king, piece_type) {
                skewer_bonus = skewer_bonus.add(Score::new(skewered_value / 5, skewered_value / 8));
            }
        }
    }
    
    skewer_bonus
}

fn find_skewered_piece_value(pos: &Position, attacker: u8, king: u8, attacker_type: PieceType) -> Option<i32> {
    let is_diagonal = attacker_type == PieceType::Bishop || 
                     (attacker_type == PieceType::Queen && 
                      ((attacker / 8) - (king / 8)) == ((attacker % 8) - (king % 8)));
    
    if let Some(skewered_sq) = find_piece_beyond_king(pos, attacker, king, is_diagonal) {
        let (piece_type, piece_color) = pos.piece_at(skewered_sq);
        if piece_color != pos.piece_at(attacker).1 && piece_type != PieceType::None {
            return Some(get_piece_value(piece_type));
        }
    }
    
    None
}

fn find_piece_beyond_king(pos: &Position, attacker: u8, king: u8, is_diagonal: bool) -> Option<u8> {
    None
}

fn is_square_attacked_by_any(pos: &Position, square: u8, by_color: Color) -> bool {
    pos.is_square_attacked(square, by_color)
}

pub fn count_threats(pos: &Position, color: Color) -> i32 {
    let enemy_pieces = pos.pieces(color.opposite());
    let our_attacks = all_attacks(pos, color);
    
    (our_attacks & enemy_pieces).count_ones() as i32
}

pub fn has_tactical_threats(pos: &Position, color: Color) -> bool {
    let threats = evaluate_threats_for_color(pos, color);
    threats.mg > 50 || threats.eg > 50
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::position::Position;
    
    #[test]
    fn test_threats_evaluation() {
        let pos = Position::startpos();
        let threats = evaluate_threats(&pos);
        
        assert!(threats.mg.abs() < 20);
        assert!(threats.eg.abs() < 20);
    }
    
    #[test]
    fn test_hanging_pieces() {
        let pos = Position::startpos();
        let hanging = evaluate_hanging_pieces(&pos, Color::White);
        
        assert_eq!(hanging.mg, 0);
        assert_eq!(hanging.eg, 0);
    }
    
    #[test]
    fn test_piece_value() {
        assert_eq!(get_piece_value(PieceType::Pawn), 100);
        assert_eq!(get_piece_value(PieceType::Queen), 900);
    }
    
    #[test]
    fn test_threat_counting() {
        let pos = Position::startpos();
        let white_threats = count_threats(&pos, Color::White);
        let black_threats = count_threats(&pos, Color::Black);
        
        assert!(white_threats > 0);
        assert!(black_threats > 0);
    }
}