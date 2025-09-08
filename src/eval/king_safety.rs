use crate::board::position::{Position, PieceType, Color};
use crate::board::bitboard::{Bitboard, EMPTY, FILE_A, FILE_H};
use crate::eval::evaluate::Score;
use crate::eval::pst::{file_of, rank_of, relative_rank};
use crate::movegen::magic::{
    all_attacks, get_knight_attacks, get_bishop_attacks, 
    get_rook_attacks, get_queen_attacks, get_king_attacks
};

const KING_DANGER_MULTIPLIER: [i32; 100] = [
    0, 0, 2, 3, 5, 8, 12, 16, 21, 27,
    34, 42, 51, 61, 72, 84, 97, 111, 126, 142,
    159, 177, 196, 216, 237, 259, 282, 306, 331, 357,
    384, 412, 441, 471, 502, 534, 567, 601, 636, 672,
    709, 747, 786, 826, 867, 909, 952, 996, 1041, 1087,
    1134, 1182, 1231, 1281, 1332, 1384, 1437, 1491, 1546, 1602,
    1659, 1717, 1776, 1836, 1897, 1959, 2022, 2086, 2151, 2217,
    2284, 2352, 2421, 2491, 2562, 2634, 2707, 2781, 2856, 2932,
    3009, 3087, 3166, 3246, 3327, 3409, 3492, 3576, 3661, 3747,
    3834, 3922, 4011, 4101, 4192, 4284, 4377, 4471, 4566, 4662
];

const PAWN_SHELTER_BONUS: [[Score; 8]; 4] = [
    [Score::new(0, 0), Score::new(0, 0), Score::new(0, 0), Score::new(0, 0), Score::new(0, 0), Score::new(0, 0), Score::new(0, 0), Score::new(0, 0)],
    [Score::new(20, 5), Score::new(40, 10), Score::new(50, 12), Score::new(40, 10), Score::new(40, 10), Score::new(50, 12), Score::new(40, 10), Score::new(20, 5)],
    [Score::new(15, 3), Score::new(30, 7), Score::new(35, 8), Score::new(30, 7), Score::new(30, 7), Score::new(35, 8), Score::new(30, 7), Score::new(15, 3)],
    [Score::new(5, 1), Score::new(15, 3), Score::new(20, 4), Score::new(15, 3), Score::new(15, 3), Score::new(20, 4), Score::new(15, 3), Score::new(5, 1)],
];

const PAWN_STORM_PENALTY: [[Score; 8]; 8] = [
    [Score::new(0, 0), Score::new(0, 0), Score::new(0, 0), Score::new(0, 0), Score::new(0, 0), Score::new(0, 0), Score::new(0, 0), Score::new(0, 0)],
    [Score::new(0, 0), Score::new(0, 0), Score::new(0, 0), Score::new(0, 0), Score::new(0, 0), Score::new(0, 0), Score::new(0, 0), Score::new(0, 0)],
    [Score::new(0, 0), Score::new(-5, -2), Score::new(-10, -4), Score::new(-8, -3), Score::new(-8, -3), Score::new(-10, -4), Score::new(-5, -2), Score::new(0, 0)],
    [Score::new(0, 0), Score::new(-10, -5), Score::new(-20, -10), Score::new(-16, -8), Score::new(-16, -8), Score::new(-20, -10), Score::new(-10, -5), Score::new(0, 0)],
    [Score::new(0, 0), Score::new(-15, -8), Score::new(-35, -18), Score::new(-28, -14), Score::new(-28, -14), Score::new(-35, -18), Score::new(-15, -8), Score::new(0, 0)],
    [Score::new(0, 0), Score::new(-22, -12), Score::new(-50, -25), Score::new(-40, -20), Score::new(-40, -20), Score::new(-50, -25), Score::new(-22, -12), Score::new(0, 0)],
    [Score::new(0, 0), Score::new(-30, -15), Score::new(-65, -35), Score::new(-52, -28), Score::new(-52, -28), Score::new(-65, -35), Score::new(-30, -15), Score::new(0, 0)],
    [Score::new(0, 0), Score::new(-35, -18), Score::new(-75, -40), Score::new(-60, -32), Score::new(-60, -32), Score::new(-75, -40), Score::new(-35, -18), Score::new(0, 0)],
];

const ATTACK_WEIGHTS: [i32; 7] = [
    0,
    2,
    3,
    3,
    4,
    6,
    0,
];

const ATTACK_UNITS: [i32; 7] = [
    0,
    25,
    35,
    35,
    45,
    90,
    0,
];

const SAFE_CHECK_BONUS: [i32; 7] = [
    0,
    0,
    45,
    40,
    60,
    80,
    0,
];

pub fn evaluate_king_safety(pos: &Position) -> Score {
    let white_safety = evaluate_king_safety_for_color(pos, Color::White);
    let black_safety = evaluate_king_safety_for_color(pos, Color::Black);
    
    white_safety.sub(black_safety)
}

pub fn evaluate_king_safety_for_color(pos: &Position, color: Color) -> Score {
    let king_sq = pos.king_square(color);
    let mut safety_score = Score::zero();
    
    safety_score = safety_score.add(evaluate_pawn_shelter(pos, king_sq, color));
    
    safety_score = safety_score.add(evaluate_pawn_storm(pos, king_sq, color));
    
    safety_score = safety_score.add(evaluate_king_zone_attacks(pos, king_sq, color));
    
    safety_score = safety_score.add(evaluate_king_position(pos, king_sq, color));
    
    safety_score = safety_score.add(evaluate_king_file_safety(pos, king_sq, color));
    
    safety_score = safety_score.add(evaluate_king_mobility(pos, king_sq, color));
    
    safety_score = safety_score.add(evaluate_safe_checks(pos, king_sq, color));
    
    safety_score = safety_score.add(evaluate_weak_king_squares(pos, king_sq, color));
    
    safety_score
}

fn evaluate_pawn_shelter(pos: &Position, king_sq: u8, color: Color) -> Score {
    let king_file = king_sq % 8;
    let king_rank = king_sq / 8;
    let our_pawns = pos.pieces_colored(PieceType::Pawn, color);
    
    let is_castled = match color {
        Color::White => king_rank == 0 && (king_file <= 2 || king_file >= 6),
        Color::Black => king_rank == 7 && (king_file <= 2 || king_file >= 6),
    };
    
    if !is_castled {
        let phase = crate::eval::material::calculate_phase(pos);
        if phase > 128 {
            return Score::new(-25, -5);
        }
        return Score::zero();
    }
    
    let mut shelter_score = Score::zero();
    
    for file_offset in -1..=1 {
        let file = (king_file as i32 + file_offset) as usize;
        if file >= 8 {
            continue;
        }
        
        let file_mask = 0x0101010101010101u64 << file;
        let pawns_on_file = our_pawns & file_mask;
        
        if pawns_on_file != 0 {
            let pawn_sq = if color == Color::White {
                pawns_on_file.trailing_zeros() as u8
            } else {
                63 - pawns_on_file.leading_zeros() as u8
            };
            
            let pawn_rank = pawn_sq / 8;
            let distance = if color == Color::White {
                pawn_rank.saturating_sub(king_rank)
            } else {
                king_rank.saturating_sub(pawn_rank)
            };
            
            if distance <= 3 {
                shelter_score = shelter_score.add(PAWN_SHELTER_BONUS[distance as usize][file]);
            }
        } else {
            let penalty = if file_offset == 0 {
                Score::new(-35, -15)
            } else {
                Score::new(-25, -10)
            };
            shelter_score = shelter_score.add(penalty);
        }
    }
    
    shelter_score
}

fn evaluate_pawn_storm(pos: &Position, king_sq: u8, color: Color) -> Score {
    let king_file = king_sq % 8;
    let king_rank = king_sq / 8;
    let enemy_pawns = pos.pieces_colored(PieceType::Pawn, color.opposite());
    
    let mut storm_penalty = Score::zero();
    
    for file_offset in -2..=2 {
        let file = (king_file as i32 + file_offset) as usize;
        if file >= 8 {
            continue;
        }
        
        let file_mask = 0x0101010101010101u64 << file;
        let pawns_on_file = enemy_pawns & file_mask;
        
        if pawns_on_file != 0 {
            let pawn_sq = if color == Color::White {
                63 - pawns_on_file.leading_zeros() as u8
            } else {
                pawns_on_file.trailing_zeros() as u8
            };
            
            let pawn_rank = pawn_sq / 8;
            let advancement = if color == Color::White {
                7 - pawn_rank
            } else {
                pawn_rank
            };
            
            if advancement >= 2 && advancement <= 7 {
                let mut penalty = PAWN_STORM_PENALTY[advancement as usize][file];
                
                let file_distance = file_offset.abs();
                penalty = Score::new(
                    penalty.mg * (3 - file_distance.min(2)) / 3,
                    penalty.eg * (3 - file_distance.min(2)) / 3
                );
                
                storm_penalty = storm_penalty.add(penalty);
            }
        }
    }
    
    storm_penalty
}

fn evaluate_king_zone_attacks(pos: &Position, king_sq: u8, color: Color) -> Score {
    let king_zone = calculate_extended_king_zone(king_sq);
    let enemy_color = color.opposite();
    
    let mut attack_units = 0;
    let mut attacker_count = 0;
    let mut attacker_weight = 0;
    
    for piece_type in 1..6 {
        let piece_type = PieceType::from(piece_type);
        let mut pieces = pos.pieces_colored(piece_type, enemy_color);
        
        while pieces != 0 {
            let square = pieces.trailing_zeros() as u8;
            pieces &= pieces - 1;
            
            let attacks = match piece_type {
                PieceType::Pawn => get_pawn_attacks(square, enemy_color),
                PieceType::Knight => get_knight_attacks(square),
                PieceType::Bishop => get_bishop_attacks(square, pos.all_pieces()),
                PieceType::Rook => get_rook_attacks(square, pos.all_pieces()),
                PieceType::Queen => get_queen_attacks(square, pos.all_pieces()),
                _ => 0,
            };
            
            let zone_attacks = attacks & king_zone;
            if zone_attacks != 0 {
                attacker_count += 1;
                attacker_weight += ATTACK_WEIGHTS[piece_type as usize];
                
                let attack_count = zone_attacks.count_ones() as i32;
                attack_units += ATTACK_UNITS[piece_type as usize] * attack_count;
                
                if (attacks & (1u64 << king_sq)) != 0 {
                    attack_units += ATTACK_UNITS[piece_type as usize] * 2;
                }
            }
        }
    }
    
    if attacker_count < 2 {
        attack_units = attack_units / 2;
    } else {
        attack_units = (attack_units * (attacker_count + attacker_weight)) / 10;
    }
    
    let our_defenders = count_defenders(pos, king_zone, color);
    attack_units = attack_units.saturating_sub(our_defenders * 10);
    
    if attack_units < 50 {
        return Score::zero();
    }
    
    let danger_index = ((attack_units - 50) / 15).min(99) as usize;
    let danger_score = -KING_DANGER_MULTIPLIER[danger_index];
    
    Score::new(danger_score, danger_score / 8)
}

fn calculate_extended_king_zone(king_sq: u8) -> Bitboard {
    let king_attacks = get_king_attacks(king_sq);
    let mut extended_zone = king_attacks | (1u64 << king_sq);
    
    let file = king_sq % 8;
    let rank = king_sq / 8;
    
    if file >= 2 { extended_zone |= 1u64 << (king_sq - 2); }
    if file <= 5 { extended_zone |= 1u64 << (king_sq + 2); }
    
    if rank >= 2 { extended_zone |= 1u64 << (king_sq - 16); }
    if rank <= 5 { extended_zone |= 1u64 << (king_sq + 16); }
    
    extended_zone
}

fn count_defenders(pos: &Position, zone: Bitboard, color: Color) -> i32 {
    let mut defenders = 0;
    
    for piece_type in 1..6 {
        let piece_type = PieceType::from(piece_type);
        let mut pieces = pos.pieces_colored(piece_type, color);
        
        while pieces != 0 {
            let square = pieces.trailing_zeros() as u8;
            pieces &= pieces - 1;
            
            let defends = match piece_type {
                PieceType::Pawn => get_pawn_attacks(square, color),
                PieceType::Knight => get_knight_attacks(square),
                PieceType::Bishop => get_bishop_attacks(square, pos.all_pieces()),
                PieceType::Rook => get_rook_attacks(square, pos.all_pieces()),
                PieceType::Queen => get_queen_attacks(square, pos.all_pieces()),
                _ => 0,
            };
            
            if (defends & zone) != 0 {
                defenders += ATTACK_WEIGHTS[piece_type as usize];
            }
        }
    }
    
    defenders
}

fn get_pawn_attacks(square: u8, color: Color) -> Bitboard {
    let rank = square / 8;
    let file = square % 8;
    
    let mut attacks = 0u64;
    
    match color {
        Color::White => {
            if rank < 7 {
                if file > 0 { attacks |= 1u64 << (square + 7); }
                if file < 7 { attacks |= 1u64 << (square + 9); }
            }
        },
        Color::Black => {
            if rank > 0 {
                if file > 0 { attacks |= 1u64 << (square - 9); }
                if file < 7 { attacks |= 1u64 << (square - 7); }
            }
        }
    }
    
    attacks
}

fn evaluate_king_position(pos: &Position, king_sq: u8, color: Color) -> Score {
    let king_rank = king_sq / 8;
    let king_file = king_sq % 8;
    
    let mut position_score = Score::zero();
    
    let is_castled = match color {
        Color::White => king_rank == 0 && (king_file <= 2 || king_file >= 6),
        Color::Black => king_rank == 7 && (king_file <= 2 || king_file >= 6),
    };
    
    if is_castled {
        position_score = position_score.add(Score::new(45, 5));
        
        if king_file >= 6 {
            position_score = position_score.add(Score::new(10, 0));
        }
    } else {
        let phase = crate::eval::material::calculate_phase(pos);
        
        if phase > 100 {
            position_score = position_score.sub(Score::new(30, 0));
            
            if king_file >= 3 && king_file <= 4 {
                position_score = position_score.sub(Score::new(25, 0));
            }
        }
    }
    
    let advanced_rank = if color == Color::White { king_rank } else { 7 - king_rank };
    if advanced_rank >= 2 {
        let phase = crate::eval::material::calculate_phase(pos);
        if phase > 64 {
            position_score = position_score.sub(Score::new(advanced_rank as i32 * 15, 0));
        }
    }
    
    position_score
}

fn evaluate_king_file_safety(pos: &Position, king_sq: u8, color: Color) -> Score {
    let king_file = king_sq % 8;
    let our_pawns = pos.pieces_colored(PieceType::Pawn, color);
    let enemy_pawns = pos.pieces_colored(PieceType::Pawn, color.opposite());
    let enemy_rooks = pos.pieces_colored(PieceType::Rook, color.opposite());
    let enemy_queens = pos.pieces_colored(PieceType::Queen, color.opposite());
    
    let mut file_safety = Score::zero();
    
    for file_offset in -1..=1 {
        let file = (king_file as i32 + file_offset) as usize;
        if file >= 8 {
            continue;
        }
        
        let file_mask = 0x0101010101010101u64 << file;
        let our_pawns_on_file = our_pawns & file_mask;
        let enemy_pawns_on_file = enemy_pawns & file_mask;
        let enemy_heavy = (enemy_rooks | enemy_queens) & file_mask;
        
        if our_pawns_on_file == 0 {
            if enemy_pawns_on_file == 0 {
                if enemy_heavy != 0 {
                    file_safety = file_safety.sub(Score::new(40, 20));
                } else {
                    file_safety = file_safety.sub(Score::new(20, 10));
                }
            } else {
                if enemy_heavy != 0 {
                    file_safety = file_safety.sub(Score::new(25, 12));
                } else {
                    file_safety = file_safety.sub(Score::new(12, 6));
                }
            }
        }
    }
    
    file_safety
}

fn evaluate_king_mobility(pos: &Position, king_sq: u8, color: Color) -> Score {
    let king_attacks = get_king_attacks(king_sq);
    let our_pieces = pos.pieces(color);
    let enemy_attacks = all_attacks(pos, color.opposite());
    
    let escape_squares = king_attacks & !our_pieces & !enemy_attacks;
    let safe_squares = escape_squares.count_ones() as i32;
    
    let total_squares = (king_attacks & !our_pieces).count_ones() as i32;
    
    let mut mobility_score = Score::zero();
    
    if safe_squares == 0 {
        mobility_score = mobility_score.sub(Score::new(50, 30));
        
        if total_squares <= 1 {
            mobility_score = mobility_score.sub(Score::new(40, 25));
        }
    } else if safe_squares <= 2 {
        mobility_score = mobility_score.sub(Score::new(20, 10));
    } else if safe_squares >= 5 {
        mobility_score = mobility_score.add(Score::new(10, 5));
    }
    
    mobility_score
}

fn evaluate_safe_checks(pos: &Position, king_sq: u8, color: Color) -> Score {
    let enemy_color = color.opposite();
    let our_attacks = all_attacks(pos, color);
    let mut safe_check_score = 0;
    
    for piece_type in 2..6 {
        let piece_type = PieceType::from(piece_type);
        let mut pieces = pos.pieces_colored(piece_type, enemy_color);
        
        while pieces != 0 {
            let square = pieces.trailing_zeros() as u8;
            pieces &= pieces - 1;
            
            let attacks = match piece_type {
                PieceType::Knight => get_knight_attacks(square),
                PieceType::Bishop => get_bishop_attacks(square, pos.all_pieces()),
                PieceType::Rook => get_rook_attacks(square, pos.all_pieces()),
                PieceType::Queen => get_queen_attacks(square, pos.all_pieces()),
                _ => 0,
            };
            
            if (attacks & (1u64 << king_sq)) != 0 {
                if (attacks & !our_attacks) != 0 {
                    safe_check_score += SAFE_CHECK_BONUS[piece_type as usize];
                }
            }
        }
    }
    
    Score::new(-safe_check_score, -safe_check_score / 4)
}

fn evaluate_weak_king_squares(pos: &Position, king_sq: u8, color: Color) -> Score {
    let king_zone = get_king_attacks(king_sq);
    let our_pawns = pos.pieces_colored(PieceType::Pawn, color);
    let our_pawn_attacks = get_all_pawn_attacks(pos, color);
    
    let weak_squares = king_zone & !our_pawn_attacks;
    let weak_count = weak_squares.count_ones() as i32;
    
    let enemy_attacks = all_attacks(pos, color.opposite());
    let exploitable_weak = weak_squares & enemy_attacks;
    let exploitable_count = exploitable_weak.count_ones() as i32;
    
    Score::new(
        -(weak_count * 5 + exploitable_count * 10),
        -(weak_count * 2 + exploitable_count * 5)
    )
}

fn get_all_pawn_attacks(pos: &Position, color: Color) -> Bitboard {
    let pawns = pos.pieces_colored(PieceType::Pawn, color);
    
    match color {
        Color::White => {
            let left_attacks = (pawns & !0x0101010101010101u64) << 7;
            let right_attacks = (pawns & !0x8080808080808080u64) << 9;
            left_attacks | right_attacks
        },
        Color::Black => {
            let left_attacks = (pawns & !0x8080808080808080u64) >> 7;
            let right_attacks = (pawns & !0x0101010101010101u64) >> 9;
            left_attacks | right_attacks
        }
    }
}

pub fn is_king_in_danger(pos: &Position, color: Color) -> bool {
    let king_sq = pos.king_square(color);
    
    if pos.is_square_attacked(king_sq, color.opposite()) {
        return true;
    }
    
    let king_zone = calculate_extended_king_zone(king_sq);
    let enemy_attacks = all_attacks(pos, color.opposite());
    let attacked_zone = enemy_attacks & king_zone;
    
    attacked_zone.count_ones() >= 5
}

pub fn calculate_king_danger_score(pos: &Position, color: Color) -> i32 {
    let safety_score = evaluate_king_safety_for_color(pos, color);
    -safety_score.mg
}

pub fn evaluate_back_rank_threats(pos: &Position, color: Color) -> Score {
    let king_sq = pos.king_square(color);
    let king_rank = rank_of(king_sq);
    let our_pawns = pos.pieces_colored(PieceType::Pawn, color);
    let enemy_rooks = pos.pieces_colored(PieceType::Rook, color.opposite());
    let enemy_queens = pos.pieces_colored(PieceType::Queen, color.opposite());
    
    let back_rank = match color {
        Color::White => 0,
        Color::Black => 7,
    };
    
    if king_rank != back_rank {
        return Score::zero();
    }
    
    let back_rank_mask = 0xFFu64 << (back_rank * 8);
    let second_rank_mask = match color {
        Color::White => 0xFF00u64,
        Color::Black => 0x00FF000000000000u64,
    };
    
    let escape_pawns = our_pawns & second_rank_mask;
    let king_file = king_sq % 8;
    
    let mut blocked_escapes = 0;
    for file_offset in -1..=1 {
        let file = (king_file as i32 + file_offset) as usize;
        if file >= 8 {
            continue;
        }
        
        let file_mask = 0x0101010101010101u64 << file;
        if (escape_pawns & file_mask) != 0 {
            blocked_escapes += 1;
        }
    }
    
    let mut threat_score = Score::zero();
    
    let enemy_heavy_on_back = (enemy_rooks | enemy_queens) & back_rank_mask;
    
    if enemy_heavy_on_back != 0 {
        if blocked_escapes == 3 {
            threat_score = threat_score.sub(Score::new(80, 50));
        } else if blocked_escapes == 2 {
            threat_score = threat_score.sub(Score::new(40, 25));
        }
        
        if enemy_heavy_on_back.count_ones() >= 2 {
            threat_score = threat_score.sub(Score::new(30, 20));
        }
    }
    
    threat_score
}

pub fn evaluate_castling_safety(pos: &Position, color: Color) -> Score {
    let king_sq = pos.king_square(color);
    let king_file = king_sq % 8;
    let king_rank = king_sq / 8;
    
    let expected_rank = match color {
        Color::White => 0,
        Color::Black => 7,
    };
    
    if king_rank == expected_rank {
        if king_file <= 2 {
            return evaluate_queenside_safety(pos, color);
        } else if king_file >= 6 {
            return evaluate_kingside_safety(pos, color);
        }
    }
    
    let phase = crate::eval::material::calculate_phase(pos);
    if phase > 100 {
        return Score::new(-30, -5);
    }
    
    Score::zero()
}

fn evaluate_kingside_safety(pos: &Position, color: Color) -> Score {
    let our_pawns = pos.pieces_colored(PieceType::Pawn, color);
    let mut safety_score = Score::new(30, 5);
    
    let pawn_squares = match color {
        Color::White => [13, 14, 15],
        Color::Black => [53, 54, 55],
    };
    
    for &square in &pawn_squares {
        if (our_pawns & (1u64 << square)) != 0 {
            safety_score = safety_score.add(Score::new(10, 2));
        } else {
            safety_score = safety_score.sub(Score::new(15, 5));
        }
    }
    
    let fianchetto_square = match color {
        Color::White => 14,
        Color::Black => 54,
    };
    
    if (our_pawns & (1u64 << fianchetto_square)) != 0 {
        let bishop_square = match color {
            Color::White => 14,
            Color::Black => 54,
        };
        let bishops = pos.pieces_colored(PieceType::Bishop, color);
        if (bishops & (1u64 << bishop_square)) != 0 {
            safety_score = safety_score.add(Score::new(20, 5));
        }
    }
    
    safety_score
}

fn evaluate_queenside_safety(pos: &Position, color: Color) -> Score {
    let our_pawns = pos.pieces_colored(PieceType::Pawn, color);
    let mut safety_score = Score::new(25, 5);
    
    let pawn_squares = match color {
        Color::White => [8, 9, 10],
        Color::Black => [48, 49, 50],
    };
    
    for &square in &pawn_squares {
        if (our_pawns & (1u64 << square)) != 0 {
            safety_score = safety_score.add(Score::new(8, 2));
        } else {
            safety_score = safety_score.sub(Score::new(12, 4));
        }
    }
    
    safety_score = safety_score.sub(Score::new(5, 0));
    
    safety_score
}