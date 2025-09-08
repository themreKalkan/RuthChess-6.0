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
    Score::new(-20, -28),
    Score::new(-50, -65),
    Score::new(-50, -65),
    Score::new(-75, -90),
    Score::new(-125, -150),
    Score::new(-250, -350),
];

const WEAK_PIECE_PENALTY: [Score; 7] = [
    Score::new(0, 0),
    Score::new(-10, -15),
    Score::new(-25, -32),
    Score::new(-25, -32),
    Score::new(-35, -42),
    Score::new(-60, -75),
    Score::new(0, 0),
];

const THREAT_BONUS: [[Score; 7]; 7] = [
    [Score::new(0, 0), Score::new(0, 0), Score::new(0, 0), Score::new(0, 0), Score::new(0, 0), Score::new(0, 0), Score::new(0, 0)],
    [Score::new(0, 0), Score::new(0, 0), Score::new(25, 32), Score::new(25, 32), Score::new(35, 50), Score::new(55, 75), Score::new(0, 0)],
    [Score::new(0, 0), Score::new(8, 12), Score::new(0, 0), Score::new(12, 18), Score::new(20, 28), Score::new(35, 45), Score::new(0, 0)],
    [Score::new(0, 0), Score::new(8, 12), Score::new(12, 18), Score::new(0, 0), Score::new(20, 28), Score::new(35, 45), Score::new(0, 0)],
    [Score::new(0, 0), Score::new(6, 10), Score::new(15, 20), Score::new(15, 20), Score::new(0, 0), Score::new(25, 35), Score::new(0, 0)],
    [Score::new(0, 0), Score::new(5, 8), Score::new(10, 15), Score::new(10, 15), Score::new(15, 20), Score::new(0, 0), Score::new(0, 0)],
    [Score::new(0, 0), Score::new(0, 0), Score::new(0, 0), Score::new(0, 0), Score::new(0, 0), Score::new(0, 0), Score::new(0, 0)],
];

const RESTRICTED_PIECE_PENALTY: [Score; 7] = [
    Score::new(0, 0),
    Score::new(-5, -8),
    Score::new(-15, -20),
    Score::new(-15, -20),
    Score::new(-20, -25),
    Score::new(-30, -35),
    Score::new(0, 0),
];

const PIN_BONUS: Score = Score::new(20, 25);
const ABSOLUTE_PIN_BONUS: Score = Score::new(35, 40);
const FORK_BONUS: Score = Score::new(25, 30);
const ROYAL_FORK_BONUS: Score = Score::new(45, 50);
const SKEWER_BONUS: Score = Score::new(30, 35);
const DISCOVERED_ATTACK_BONUS: Score = Score::new(35, 40);

const SAFE_PAWN_THREAT: Score = Score::new(18, 22);

const KING_THREAT_BONUS: Score = Score::new(25, 15);

const OVERLOADED_PENALTY: Score = Score::new(-20, -25);

pub fn evaluate_threats(pos: &Position) -> Score {
    let white_threats = evaluate_threats_for_color(pos, Color::White);
    let black_threats = evaluate_threats_for_color(pos, Color::Black);
    
    white_threats.sub(black_threats)
}

pub fn evaluate_threats_for_color(pos: &Position, color: Color) -> Score {
    let mut threat_score = Score::zero();
    
    threat_score = threat_score.add(evaluate_hanging_pieces(pos, color));
    
    threat_score = threat_score.add(evaluate_weak_pieces(pos, color));
    
    threat_score = threat_score.add(evaluate_piece_threats(pos, color));
    
    threat_score = threat_score.add(evaluate_pawn_threats(pos, color));
    
    threat_score = threat_score.add(evaluate_tactical_patterns(pos, color));
    
    threat_score = threat_score.add(evaluate_restricted_pieces(pos, color));
    
    threat_score = threat_score.add(evaluate_overloaded_pieces(pos, color));
    
    threat_score = threat_score.add(evaluate_king_threats(pos, color));
    
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
        
        if is_piece_hanging(pos, square, piece_type, color) {
            penalty = penalty.add(HANGING_PIECE_PENALTY[piece_type as usize]);
        }
    }
    
    penalty
}

fn is_piece_hanging(pos: &Position, square: u8, piece_type: PieceType, color: Color) -> bool {
    let attackers = get_attackers_to_square(pos, square, color.opposite());
    let defenders = get_attackers_to_square(pos, square, color);
    
    if attackers.is_empty() {
        return false;
    }
    
    if defenders.is_empty() {
        return true;
    }
    
    let lowest_attacker = get_lowest_value_attacker(&attackers);
    let piece_value = get_piece_value(piece_type);
    
    get_piece_value(lowest_attacker) < piece_value
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
    let attackers = get_attackers_to_square(pos, square, color.opposite());
    let defenders = get_attackers_to_square(pos, square, color);
    
    if attackers.is_empty() {
        return false;
    }
    
    if attackers.len() > defenders.len() {
        return true;
    }
    
    if !defenders.is_empty() {
        let lowest_attacker = get_lowest_value_attacker(&attackers);
        let lowest_defender = get_lowest_value_attacker(&defenders);
        
        return get_piece_value(lowest_attacker) < get_piece_value(lowest_defender);
    }
    
    true
}

fn evaluate_piece_threats(pos: &Position, color: Color) -> Score {
    let mut threat_score = Score::zero();
    let enemy_pieces = pos.pieces(color.opposite());
    
    for attacker_type in 1..6 {
        let attacker_type = PieceType::from(attacker_type);
        let mut attackers = pos.pieces_colored(attacker_type, color);
        
        while attackers != 0 {
            let square = attackers.trailing_zeros() as u8;
            attackers &= attackers - 1;
            
            let attacks = get_piece_attacks(pos, square, attacker_type);
            let threatened = attacks & enemy_pieces;
            
            let mut threatened_bb = threatened;
            while threatened_bb != 0 {
                let target_sq = threatened_bb.trailing_zeros() as u8;
                threatened_bb &= threatened_bb - 1;
                
                let (target_type, _) = pos.piece_at(target_sq);
                
                threat_score = threat_score.add(THREAT_BONUS[attacker_type as usize][target_type as usize]);
                
                if !is_square_defended(pos, target_sq, color.opposite()) {
                    threat_score = threat_score.add(Score::new(10, 12));
                }
            }
        }
    }
    
    threat_score
}

fn evaluate_pawn_threats(pos: &Position, color: Color) -> Score {
    let mut threat_score = Score::zero();
    let our_pawns = pos.pieces_colored(PieceType::Pawn, color);
    let enemy_pieces = pos.pieces(color.opposite());
    let enemy_non_pawns = enemy_pieces & !pos.pieces_colored(PieceType::Pawn, color.opposite());
    
    let pawn_attacks = match color {
        Color::White => {
            let left = (our_pawns & !0x0101010101010101u64) << 7;
            let right = (our_pawns & !0x8080808080808080u64) << 9;
            left | right
        },
        Color::Black => {
            let left = (our_pawns & !0x8080808080808080u64) >> 7;
            let right = (our_pawns & !0x0101010101010101u64) >> 9;
            left | right
        }
    };
    
    let threatened_pieces = pawn_attacks & enemy_non_pawns;
    let mut threatened_bb = threatened_pieces;
    
    while threatened_bb != 0 {
        let target_sq = threatened_bb.trailing_zeros() as u8;
        threatened_bb &= threatened_bb - 1;
        
        let (piece_type, _) = pos.piece_at(target_sq);
        
        let bonus = match piece_type {
            PieceType::Knight | PieceType::Bishop => Score::new(25, 35),
            PieceType::Rook => Score::new(35, 50),
            PieceType::Queen => Score::new(55, 75),
            _ => Score::zero(),
        };
        
        threat_score = threat_score.add(bonus);
        
        if !is_square_defended(pos, target_sq, color.opposite()) {
            threat_score = threat_score.add(SAFE_PAWN_THREAT);
        }
    }
    
    threat_score
}

fn evaluate_tactical_patterns(pos: &Position, color: Color) -> Score {
    let mut pattern_score = Score::zero();
    
    pattern_score = pattern_score.add(evaluate_pins(pos, color));
    
    pattern_score = pattern_score.add(evaluate_forks(pos, color));
    
    pattern_score = pattern_score.add(evaluate_skewers(pos, color));
    
    pattern_score = pattern_score.add(evaluate_discovered_attacks(pos, color));
    
    pattern_score
}

fn evaluate_pins(pos: &Position, color: Color) -> Score {
    let mut pin_bonus = Score::zero();
    let enemy_king = pos.king_square(color.opposite());
    let enemy_pieces = pos.pieces(color.opposite());
    
    let our_bishops = pos.pieces_colored(PieceType::Bishop, color);
    let our_rooks = pos.pieces_colored(PieceType::Rook, color);
    let our_queens = pos.pieces_colored(PieceType::Queen, color);
    
    let diagonal_pieces = our_bishops | our_queens;
    let mut diag_bb = diagonal_pieces;
    while diag_bb != 0 {
        let square = diag_bb.trailing_zeros() as u8;
        diag_bb &= diag_bb - 1;
        
        let pin_ray = get_pin_ray(square, enemy_king, true);
        if pin_ray != 0 {
            let pinned = pin_ray & enemy_pieces;
            if pinned.count_ones() == 1 {
                let pinned_sq = pinned.trailing_zeros() as u8;
                let (pinned_type, _) = pos.piece_at(pinned_sq);
                
                if can_see_through(pos, square, enemy_king, pinned_sq) {
                    pin_bonus = pin_bonus.add(ABSOLUTE_PIN_BONUS);
                    pin_bonus = pin_bonus.add(Score::new(
                        get_piece_value(pinned_type) / 10,
                        get_piece_value(pinned_type) / 8
                    ));
                }
            }
        }
    }
    
    let straight_pieces = our_rooks | our_queens;
    let mut straight_bb = straight_pieces;
    while straight_bb != 0 {
        let square = straight_bb.trailing_zeros() as u8;
        straight_bb &= straight_bb - 1;
        
        let pin_ray = get_pin_ray(square, enemy_king, false);
        if pin_ray != 0 {
            let pinned = pin_ray & enemy_pieces;
            if pinned.count_ones() == 1 {
                let pinned_sq = pinned.trailing_zeros() as u8;
                let (pinned_type, _) = pos.piece_at(pinned_sq);
                
                if can_see_through(pos, square, enemy_king, pinned_sq) {
                    pin_bonus = pin_bonus.add(ABSOLUTE_PIN_BONUS);
                    pin_bonus = pin_bonus.add(Score::new(
                        get_piece_value(pinned_type) / 10,
                        get_piece_value(pinned_type) / 8
                    ));
                }
            }
        }
    }
    
    let valuable_pieces = pos.pieces_colored(PieceType::Queen, color.opposite()) |
                         pos.pieces_colored(PieceType::Rook, color.opposite());
    
    let mut valuable_bb = valuable_pieces;
    while valuable_bb != 0 {
        let target = valuable_bb.trailing_zeros() as u8;
        valuable_bb &= valuable_bb - 1;
        
        let pins = find_relative_pins(pos, target, color);
        pin_bonus = pin_bonus.add(Score::new(pins * 8, pins * 10));
    }
    
    pin_bonus
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
            let mut fork_value = FORK_BONUS;
            
            if (attacks & enemy_king) != 0 {
                fork_value = fork_value.add(ROYAL_FORK_BONUS);
            }
            
            let valuable_forked = (attacks & enemy_valuable).count_ones();
            if valuable_forked >= 2 {
                fork_value = fork_value.add(Score::new(20, 25));
            }
            
            fork_bonus = fork_bonus.add(fork_value);
        }
    }
    
    let pawns = pos.pieces_colored(PieceType::Pawn, color);
    let pawn_attacks = get_all_pawn_attacks(pawns, color);
    
    let mut attack_squares = pawn_attacks;
    while attack_squares != 0 {
        let square = attack_squares.trailing_zeros() as u8;
        attack_squares &= attack_squares - 1;
        
        let victims = get_pawn_fork_victims(pos, square, color);
        if victims >= 2 {
            fork_bonus = fork_bonus.add(Score::new(15, 20));
            
            if victims >= 3 {
                fork_bonus = fork_bonus.add(Score::new(10, 12));
            }
        }
    }
    
    fork_bonus
}

fn evaluate_skewers(pos: &Position, color: Color) -> Score {
    let mut skewer_bonus = Score::zero();
    
    let sliding_pieces = pos.pieces_colored(PieceType::Bishop, color) |
                        pos.pieces_colored(PieceType::Rook, color) |
                        pos.pieces_colored(PieceType::Queen, color);
    
    let enemy_valuable = pos.pieces_colored(PieceType::Queen, color.opposite()) |
                        pos.pieces_colored(PieceType::Rook, color.opposite());
    let enemy_king = pos.king_square(color.opposite());
    
    let mut sliders_bb = sliding_pieces;
    while sliders_bb != 0 {
        let square = sliders_bb.trailing_zeros() as u8;
        sliders_bb &= sliders_bb - 1;
        
        let (piece_type, _) = pos.piece_at(square);
        
        let mut targets = enemy_valuable;
        while targets != 0 {
            let target = targets.trailing_zeros() as u8;
            targets &= targets - 1;
            
            if let Some(skewered_value) = find_skewer(pos, square, target, piece_type) {
                skewer_bonus = skewer_bonus.add(SKEWER_BONUS);
                skewer_bonus = skewer_bonus.add(Score::new(
                    skewered_value / 20,
                    skewered_value / 15
                ));
            }
        }
    }
    
    skewer_bonus
}

fn evaluate_discovered_attacks(pos: &Position, color: Color) -> Score {
    let mut discovered_bonus = Score::zero();
    
    let our_pieces = pos.pieces(color);
    let enemy_valuable = pos.pieces_colored(PieceType::Queen, color.opposite()) |
                        pos.pieces_colored(PieceType::Rook, color.opposite()) |
                        pos.pieces_colored(PieceType::King, color.opposite());
    
    let our_sliders = pos.pieces_colored(PieceType::Bishop, color) |
                     pos.pieces_colored(PieceType::Rook, color) |
                     pos.pieces_colored(PieceType::Queen, color);
    
    let mut sliders_bb = our_sliders;
    while sliders_bb != 0 {
        let slider = sliders_bb.trailing_zeros() as u8;
        sliders_bb &= sliders_bb - 1;
        
        let potential_discoveries = count_discovered_attacks(pos, slider, color);
        
        if potential_discoveries > 0 {
            discovered_bonus = discovered_bonus.add(Score::new(
                potential_discoveries * 12,
                potential_discoveries * 8
            ));
        }
    }
    
    discovered_bonus
}

fn evaluate_restricted_pieces(pos: &Position, color: Color) -> Score {
    let mut restriction_penalty = Score::zero();
    
    for piece_type in 2..6 {
        let piece_type = PieceType::from(piece_type);
        let mut pieces = pos.pieces_colored(piece_type, color);
        
        while pieces != 0 {
            let square = pieces.trailing_zeros() as u8;
            pieces &= pieces - 1;
            
            let mobility = get_piece_mobility(pos, square, piece_type);
            
            let restricted = match piece_type {
                PieceType::Knight => mobility <= 2,
                PieceType::Bishop => mobility <= 3,
                PieceType::Rook => mobility <= 4,
                PieceType::Queen => mobility <= 5,
                _ => false,
            };
            
            if restricted {
                restriction_penalty = restriction_penalty.add(RESTRICTED_PIECE_PENALTY[piece_type as usize]);
                
                if mobility == 0 {
                    restriction_penalty = restriction_penalty.add(Score::new(-20, -25));
                }
            }
        }
    }
    
    restriction_penalty
}

fn evaluate_overloaded_pieces(pos: &Position, color: Color) -> Score {
    let mut overload_penalty = Score::zero();
    
    for piece_type in 1..6 {
        let piece_type = PieceType::from(piece_type);
        let mut pieces = pos.pieces_colored(piece_type, color);
        
        while pieces != 0 {
            let square = pieces.trailing_zeros() as u8;
            pieces &= pieces - 1;
            
            let defended_count = count_important_defenses(pos, square, color);
            
            if defended_count >= 3 {
                overload_penalty = overload_penalty.add(OVERLOADED_PENALTY);
                
                if defended_count >= 4 {
                    overload_penalty = overload_penalty.add(OVERLOADED_PENALTY);
                }
            }
        }
    }
    
    overload_penalty
}

fn evaluate_king_threats(pos: &Position, color: Color) -> Score {
    let mut king_threat_score = Score::zero();
    let enemy_king = pos.king_square(color.opposite());
    let enemy_king_zone = get_king_zone(enemy_king);
    
    let mut attacking_pieces = 0;
    let mut attack_weight = 0;
    
    for piece_type in 1..6 {
        let piece_type = PieceType::from(piece_type);
        let mut pieces = pos.pieces_colored(piece_type, color);
        
        while pieces != 0 {
            let square = pieces.trailing_zeros() as u8;
            pieces &= pieces - 1;
            
            let attacks = get_piece_attacks(pos, square, piece_type);
            
            if (attacks & enemy_king_zone) != 0 {
                attacking_pieces += 1;
                attack_weight += get_piece_value(piece_type) / 100;
                
                if (attacks & (1u64 << enemy_king)) != 0 {
                    king_threat_score = king_threat_score.add(KING_THREAT_BONUS);
                }
            }
        }
    }
    
    if attacking_pieces >= 3 {
        king_threat_score = king_threat_score.add(Score::new(
            attacking_pieces * 5 + attack_weight * 3,
            attacking_pieces * 3 + attack_weight * 2
        ));
    }
    
    king_threat_score
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

fn get_attackers_to_square(pos: &Position, square: u8, by_color: Color) -> Vec<PieceType> {
    let mut attackers = Vec::new();
    
    for piece_type in 1..6 {
        let piece_type = PieceType::from(piece_type);
        let mut pieces = pos.pieces_colored(piece_type, by_color);
        
        while pieces != 0 {
            let attacker_sq = pieces.trailing_zeros() as u8;
            pieces &= pieces - 1;
            
            if can_attack(pos, attacker_sq, square, piece_type) {
                attackers.push(piece_type);
            }
        }
    }
    
    attackers
}

fn can_attack(pos: &Position, from: u8, to: u8, piece_type: PieceType) -> bool {
    let attacks = get_piece_attacks(pos, from, piece_type);
    (attacks & (1u64 << to)) != 0
}

fn get_piece_attacks(pos: &Position, square: u8, piece_type: PieceType) -> Bitboard {
    match piece_type {
        PieceType::Pawn => {
            let (_, color) = pos.piece_at(square);
            get_pawn_attacks(square, color)
        }
        PieceType::Knight => get_knight_attacks(square),
        PieceType::Bishop => get_bishop_attacks(square, pos.all_pieces()),
        PieceType::Rook => get_rook_attacks(square, pos.all_pieces()),
        PieceType::Queen => get_queen_attacks(square, pos.all_pieces()),
        PieceType::King => get_king_attacks(square),
        _ => 0,
    }
}

fn get_lowest_value_attacker(attackers: &[PieceType]) -> PieceType {
    let mut lowest = PieceType::Queen;
    let mut lowest_value = i32::MAX;
    
    for &attacker in attackers {
        let value = get_piece_value(attacker);
        if value < lowest_value {
            lowest_value = value;
            lowest = attacker;
        }
    }
    
    lowest
}

fn is_square_defended(pos: &Position, square: u8, by_color: Color) -> bool {
    !get_attackers_to_square(pos, square, by_color).is_empty()
}

fn get_pin_ray(from: u8, to: u8, is_diagonal: bool) -> Bitboard {
    let from_file = from % 8;
    let from_rank = from / 8;
    let to_file = to % 8;
    let to_rank = to / 8;
    
    if is_diagonal {
        if (from_file as i8 - to_file as i8).abs() != (from_rank as i8 - to_rank as i8).abs() {
            return 0;
        }
    } else {
        if from_file != to_file && from_rank != to_rank {
            return 0;
        }
    }
    
    let mut ray = 0u64;
    let file_step = (to_file as i8 - from_file as i8).signum();
    let rank_step = (to_rank as i8 - from_rank as i8).signum();
    
    let mut current_file = from_file as i8 + file_step;
    let mut current_rank = from_rank as i8 + rank_step;
    
    while current_file != to_file as i8 || current_rank != to_rank as i8 {
        if current_file < 0 || current_file >= 8 || current_rank < 0 || current_rank >= 8 {
            break;
        }
        
        ray |= 1u64 << (current_rank * 8 + current_file);
        current_file += file_step;
        current_rank += rank_step;
    }
    
    ray
}

fn can_see_through(pos: &Position, from: u8, to: u8, through: u8) -> bool {
    let ray = get_pin_ray(from, to, 
        (from % 8 != to % 8) && (from / 8 != to / 8));
    
    if ray == 0 {
        return false;
    }
    
    if (ray & (1u64 << through)) == 0 {
        return false;
    }
    
    let pieces_on_ray = ray & pos.all_pieces();
    pieces_on_ray.count_ones() == 1
}

fn find_relative_pins(pos: &Position, target: u8, by_color: Color) -> i32 {
    let mut pins = 0;
    
    let sliders = pos.pieces_colored(PieceType::Bishop, by_color) |
                 pos.pieces_colored(PieceType::Rook, by_color) |
                 pos.pieces_colored(PieceType::Queen, by_color);
    
    let mut sliders_bb = sliders;
    while sliders_bb != 0 {
        let slider = sliders_bb.trailing_zeros() as u8;
        sliders_bb &= sliders_bb - 1;
        
        let ray = get_pin_ray(slider, target, 
            pos.piece_at(slider).0 == PieceType::Bishop || 
            (pos.piece_at(slider).0 == PieceType::Queen && 
             (slider % 8 != target % 8) && (slider / 8 != target / 8)));
        
        let pieces_between = ray & pos.all_pieces();
        if pieces_between.count_ones() == 1 {
            pins += 1;
        }
    }
    
    pins
}

fn get_all_pawn_attacks(pawns: Bitboard, color: Color) -> Bitboard {
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

fn get_pawn_fork_victims(pos: &Position, square: u8, color: Color) -> i32 {
    let attacks = get_pawn_attacks(square, color);
    let enemy_pieces = pos.pieces(color.opposite());
    
    (attacks & enemy_pieces).count_ones() as i32
}

fn find_skewer(pos: &Position, attacker: u8, front_piece: u8, attacker_type: PieceType) -> Option<i32> {
    let attacks = get_piece_attacks(pos, attacker, attacker_type);
    if (attacks & (1u64 << front_piece)) == 0 {
        return None;
    }
    
    let ray = get_pin_ray(attacker, front_piece, 
        attacker_type == PieceType::Bishop || 
        (attacker_type == PieceType::Queen && 
         (attacker % 8 != front_piece % 8) && (attacker / 8 != front_piece / 8)));
    
    let extended_ray = extend_ray_beyond(ray, attacker, front_piece);
    let behind_pieces = extended_ray & pos.all_pieces();
    
    if behind_pieces != 0 {
        let behind_sq = behind_pieces.trailing_zeros() as u8;
        let (piece_type, piece_color) = pos.piece_at(behind_sq);
        
        if piece_color != pos.piece_at(attacker).1 {
            return Some(get_piece_value(piece_type));
        }
    }
    
    None
}

fn extend_ray_beyond(ray: Bitboard, from: u8, through: u8) -> Bitboard {
    let from_file = from % 8;
    let from_rank = from / 8;
    let through_file = through % 8;
    let through_rank = through / 8;
    
    let file_step = (through_file as i8 - from_file as i8).signum();
    let rank_step = (through_rank as i8 - from_rank as i8).signum();
    
    let mut extended = 0u64;
    let mut current_file = through_file as i8 + file_step;
    let mut current_rank = through_rank as i8 + rank_step;
    
    while current_file >= 0 && current_file < 8 && 
          current_rank >= 0 && current_rank < 8 {
        extended |= 1u64 << (current_rank * 8 + current_file);
        current_file += file_step;
        current_rank += rank_step;
    }
    
    extended
}

fn count_discovered_attacks(pos: &Position, slider: u8, color: Color) -> i32 {
    let mut count = 0;
    let our_pieces = pos.pieces(color) & !(1u64 << slider);
    
    let mut pieces_bb = our_pieces;
    while pieces_bb != 0 {
        let blocker = pieces_bb.trailing_zeros() as u8;
        pieces_bb &= pieces_bb - 1;
        
        let slider_attacks = get_piece_attacks(pos, slider, pos.piece_at(slider).0);
        let attacks_without_blocker = get_attacks_without_blocker(pos, slider, blocker);
        
        let discovered = attacks_without_blocker & !slider_attacks;
        if discovered != 0 {
            count += discovered.count_ones() as i32;
        }
    }
    
    count
}

fn get_attacks_without_blocker(pos: &Position, attacker: u8, blocker: u8) -> Bitboard {
    let mut temp_all_pieces = pos.all_pieces() & !(1u64 << blocker);
    let (piece_type, _) = pos.piece_at(attacker);
    
    match piece_type {
        PieceType::Bishop => get_bishop_attacks(attacker, temp_all_pieces),
        PieceType::Rook => get_rook_attacks(attacker, temp_all_pieces),
        PieceType::Queen => get_queen_attacks(attacker, temp_all_pieces),
        _ => 0,
    }
}

fn get_piece_mobility(pos: &Position, square: u8, piece_type: PieceType) -> u32 {
    let our_pieces = pos.pieces(pos.piece_at(square).1);
    let attacks = get_piece_attacks(pos, square, piece_type);
    let enemy_pawn_attacks = get_enemy_pawn_control(pos, pos.piece_at(square).1.opposite());
    
    let safe_squares = attacks & !our_pieces & !enemy_pawn_attacks;
    safe_squares.count_ones()
}

fn get_enemy_pawn_control(pos: &Position, enemy_color: Color) -> Bitboard {
    let enemy_pawns = pos.pieces_colored(PieceType::Pawn, enemy_color);
    get_all_pawn_attacks(enemy_pawns, enemy_color)
}

fn count_important_defenses(pos: &Position, defender: u8, color: Color) -> i32 {
    let mut count = 0;
    let defends = get_piece_attacks(pos, defender, pos.piece_at(defender).0);
    
    let valuable = pos.pieces_colored(PieceType::Queen, color) |
                  pos.pieces_colored(PieceType::Rook, color);
    
    count += (defends & valuable).count_ones() as i32 * 2;
    
    let others = pos.pieces_colored(PieceType::Knight, color) |
                pos.pieces_colored(PieceType::Bishop, color);
    
    count += (defends & others).count_ones() as i32;
    
    count
}

fn get_king_zone(king_sq: u8) -> Bitboard {
    let king_attacks = get_king_attacks(king_sq);
    king_attacks | (1u64 << king_sq)
}

pub fn count_threats(pos: &Position, color: Color) -> i32 {
    let enemy_pieces = pos.pieces(color.opposite());
    let our_attacks = all_attacks(pos, color);
    
    (our_attacks & enemy_pieces).count_ones() as i32
}

pub fn has_tactical_threats(pos: &Position, color: Color) -> bool {
    let threats = evaluate_threats_for_color(pos, color);
    threats.mg > 60 || threats.eg > 60
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::position::Position;
    
    #[test]
    fn test_threats_evaluation() {
        let pos = Position::startpos();
        let threats = evaluate_threats(&pos);
        
        assert!(threats.mg.abs() < 30);
        assert!(threats.eg.abs() < 30);
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
        assert_eq!(get_piece_value(PieceType::Knight), 320);
        assert_eq!(get_piece_value(PieceType::Bishop), 330);
        assert_eq!(get_piece_value(PieceType::Rook), 500);
        assert_eq!(get_piece_value(PieceType::Queen), 900);
    }
    
    #[test]
    fn test_threat_counting() {
        let pos = Position::startpos();
        let white_threats = count_threats(&pos, Color::White);
        let black_threats = count_threats(&pos, Color::Black);
        
        assert!(white_threats >= 0);
        assert!(black_threats >= 0);
    }
}