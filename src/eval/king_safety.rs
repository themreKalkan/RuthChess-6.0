
use crate::board::position::{Position, PieceType, Color};
use crate::board::bitboard::{Bitboard, EMPTY, FILE_A, FILE_H};
use crate::eval::evaluate::Score;
use crate::eval::pst::{file_of, rank_of, relative_rank};
use crate::movegen::magic::{
    all_attacks, get_knight_attacks, get_bishop_attacks, 
    get_rook_attacks, get_queen_attacks, get_king_attacks
};

const KING_DANGER_MULTIPLIER: [i32; 100] = [
    0, 0, 1, 2, 3, 5, 7, 9, 12, 15,
    18, 22, 26, 30, 35, 39, 44, 50, 56, 62,
    68, 75, 82, 85, 89, 97, 105, 113, 122, 131,
    140, 150, 169, 180, 191, 202, 213, 225, 237, 248,
    260, 272, 283, 295, 307, 319, 330, 342, 354, 366,
    377, 389, 401, 412, 424, 436, 448, 459, 471, 483,
    494, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    500, 500, 500, 500, 500, 500, 500, 500, 500, 500
];

const PAWN_SHIELD_BONUS: [[i32; 8]; 4] = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [15, 35, 45, 35, 35, 45, 35, 15],
    [10, 25, 30, 25, 25, 30, 25, 10],
    [0, 10, 15, 10, 10, 15, 10, 0],
];

const PAWN_STORM_PENALTY: [[i32; 8]; 8] = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, -3, -6, -5, -5, -6, -3, 0],
    [0, -8, -15, -12, -12, -15, -8, 0],
    [0, -12, -25, -20, -20, -25, -12, 0],
    [0, -18, -35, -30, -30, -35, -18, 0],
    [0, -25, -45, -40, -40, -45, -25, 0],
    [0, -30, -55, -50, -50, -55, -30, 0],
];

const ATTACK_WEIGHTS: [i32; 7] = [
    0,
    1,
    2,
    2,
    3,
    5,
    0,
];

const ATTACK_UNITS: [i32; 7] = [
    0,
    20,
    30,
    30,
    40,
    80,
    0,
];

pub fn evaluate_king_safety(pos: &Position) -> Score {
    let white_safety = evaluate_king_safety_for_color(pos, Color::White);
    let black_safety = evaluate_king_safety_for_color(pos, Color::Black);

    
    let result = white_safety.sub(black_safety);
    
    #[cfg(debug_assertions)]
    {
        if pos.fullmove_number <= 2 {
            println!("=== King Safety Debug ===");
            println!("White: MG={:+3}, EG={:+3}", white_safety.mg, white_safety.eg);
            println!("Black: MG={:+3}, EG={:+3}", black_safety.mg, black_safety.eg);
            println!("Diff:  MG={:+3}, EG={:+3}", result.mg, result.eg);
            println!("========================");
        }
    }
    
    result
}

fn evaluate_king_safety_for_color(pos: &Position, color: Color) -> Score {
    let king_sq = pos.king_square(color);
    let mut safety_score = Score::zero();
    
    
    safety_score = safety_score.add(evaluate_pawn_storm(pos, king_sq, color));
    
    safety_score = safety_score.add(evaluate_king_zone_attacks(pos, king_sq, color));
    
    safety_score = safety_score.add(evaluate_king_position(pos, king_sq, color));
    
    safety_score = safety_score.add(evaluate_king_file_safety(pos, king_sq, color));
    
    safety_score = safety_score.add(evaluate_king_mobility(pos, king_sq, color));
    
    safety_score
}


fn evaluate_pawn_storm(pos: &Position, king_sq: u8, color: Color) -> Score {
    let flipped_king_sq = flip_square_for_color(king_sq, color);
    let flipped_enemy_pawns = get_flipped_enemy_pawns(pos, color);
    
    let king_file = flipped_king_sq % 8;
    let king_rank = flipped_king_sq / 8;
    
    let mut storm_penalty = 0;
    
    for file_offset in -2..=2 {
        let file = (king_file as i32 + file_offset) as usize;
        if file >= 8 {
            continue;
        }
        
        let file_mask = 0x0101010101010101u64 << file;
        let file_pawns = flipped_enemy_pawns & file_mask;
        
        if file_pawns != 0 {
            let most_advanced_rank = 7 - (file_pawns.leading_zeros() / 8) as u8;
            
            if most_advanced_rank < 8 {
                let advancement = if most_advanced_rank <= 6 { 6 - most_advanced_rank } else { 0 };
                
                if advancement >= 2 {
                    let base_penalty = PAWN_STORM_PENALTY[most_advanced_rank as usize][file];
                    
                    let distance_scale = match file_offset.abs() {
                        0 => 100,
                        1 => 70,
                        2 => 40,
                        _ => 20,
                    };
                    
                    storm_penalty += (base_penalty * distance_scale) / 100;
                    
                    let rank_distance = (most_advanced_rank as i32 - king_rank as i32).abs();
                    if rank_distance <= 2 && file_offset.abs() <= 1 {
                        storm_penalty -= 15;
                    }
                }
            }
        }
    }
    
    Score::new(storm_penalty, storm_penalty / 4)
}

fn evaluate_king_zone_attacks(pos: &Position, king_sq: u8, color: Color) -> Score {
    let king_zone = calculate_king_zone(king_sq);
    let enemy_color = color.opposite();
    
    let mut attack_units = 0;
    let mut attacker_count = 0;
    
    for piece_type in 1..6 {
        let piece_type = PieceType::from(piece_type);
        let mut pieces = pos.pieces_colored(piece_type, enemy_color);
        
        while pieces != 0 {
            let square = pieces.trailing_zeros() as u8;
            pieces &= pieces - 1;
            
            let attacks = match piece_type {
                PieceType::Pawn => get_pawn_attacks_for_safety(square, enemy_color),
                PieceType::Knight => get_knight_attacks(square),
                PieceType::Bishop => get_bishop_attacks(square, pos.all_pieces()),
                PieceType::Rook => get_rook_attacks(square, pos.all_pieces()),
                PieceType::Queen => get_queen_attacks(square, pos.all_pieces()),
                _ => 0,
            };
            
            let zone_attacks = attacks & king_zone;
            if zone_attacks != 0 {
                attacker_count += 1;
                let attack_count = zone_attacks.count_ones() as i32;
                attack_units += ATTACK_UNITS[piece_type as usize] * attack_count;
                
                if (attacks & (1u64 << king_sq)) != 0 {
                    attack_units += ATTACK_UNITS[piece_type as usize] ;
                }
            }
        }
    }
    
    if attacker_count >= 3 {
        attack_units = attack_units * (attacker_count + 3) / 6;
    }
    
    if attack_units < 40 {
        attack_units = 0;
    } else {
        attack_units -= 40;
    }
    
    let danger_index = (attack_units / 12).min(99) as usize;
    let danger_score = -KING_DANGER_MULTIPLIER[danger_index];
    
    Score::new(danger_score, danger_score / 6)
}

fn calculate_king_zone(king_sq: u8) -> Bitboard {
    let king_attacks = get_king_attacks(king_sq);
    king_attacks | (1u64 << king_sq)
}

#[inline(always)]
fn flip_square_for_color(square: u8, color: Color) -> u8 {
    match color {
        Color::White => square,
        Color::Black => square ^ 56,  // Dikey flip (rank'leri ters çevir)
    }
}

#[inline(always)]
pub fn flip_bitboard_for_color(bb: Bitboard, color: Color) -> Bitboard {
    match color {
        Color::White => bb,
        Color::Black => bb.swap_bytes(),
    }
}

pub fn flip_board(pos:Position, color:Color)->Position{
    match color {
        Color::White => pos,
        Color::Black => {
            let mut flipped = Position::new();
            let mut temp_bb = pos.piece_bb;
            for i in 0..7{
                flipped.piece_bb[i] = flip_bitboard_for_color(temp_bb[i], color);
            }
            (flipped)
        }
    }
}

fn get_flipped_pawns(pos: &Position, color: Color) -> Bitboard {
    let pawns = pos.pieces_colored(PieceType::Pawn, color);
    flip_bitboard_for_color(pawns, color)
}

fn get_flipped_enemy_pawns(pos: &Position, color: Color) -> Bitboard {
    let enemy_pawns = pos.pieces_colored(PieceType::Pawn, color.opposite());
    flip_bitboard_for_color(enemy_pawns, color)
}

fn get_pawn_attacks_for_safety(square: u8, color: Color) -> Bitboard {
    let flipped_square = flip_square_for_color(square, color);
    let rank = flipped_square / 8;
    let file = flipped_square % 8;
    
    if rank >= 7 { return 0; }
    
    let mut attacks = 0u64;
    let target_rank = rank + 1;
    
    if file > 0 {
        attacks |= 1u64 << (target_rank * 8 + file - 1);
    }
    
    if file < 7 {
        attacks |= 1u64 << (target_rank * 8 + file + 1);
    }
    
    flip_bitboard_for_color(attacks, color)
}

fn evaluate_king_position(pos: &Position, king_sq: u8, color: Color) -> Score {
    let flipped_king_sq = flip_square_for_color(king_sq, color);
    let king_rank = flipped_king_sq / 8;
    let king_file = flipped_king_sq % 8;
    
    let mut position_score = 0;
    
    let on_starting_square = flipped_king_sq == 4;
    
    let is_castled = (king_file <= 2 && king_rank == 0) || (king_file >= 6 && king_rank == 0);
    
    if is_castled {
        position_score += 40;
    } else if on_starting_square {
        position_score -= 5;
    } else {
        if king_file >= 3 && king_file <= 4 {
            position_score -= 20;
        }
        
        if king_rank > 0 {
            position_score -= king_rank as i32 * 8;
        }
    }
    
    if king_rank == 0 {
        position_score += 10;
    }
    
    if king_file >= 3 && king_file <= 4 && !is_castled {
        position_score -= 15;
    }
    
    Score::new(position_score, -position_score / 3)
}

fn evaluate_king_file_safety(pos: &Position, king_sq: u8, color: Color) -> Score {
    let flipped_king_sq = flip_square_for_color(king_sq, color);
    let king_file = flipped_king_sq % 8;
    let flipped_our_pawns = get_flipped_pawns(pos, color);
    let flipped_enemy_rooks = flip_bitboard_for_color(pos.pieces_colored(PieceType::Rook, color.opposite()), color);
    let flipped_enemy_queens = flip_bitboard_for_color(pos.pieces_colored(PieceType::Queen, color.opposite()), color);
    
    let mut file_safety = 0;
    
    for file_offset in -1..=1 {
        let file = (king_file as i32 + file_offset) as usize;
        if file >= 8 {
            continue;
        }
        
        let file_mask = 0x0101010101010101u64 << file;
        let file_pawns = flipped_our_pawns & file_mask;
        let enemy_heavy = (flipped_enemy_rooks | flipped_enemy_queens) & file_mask;
        
        if file_pawns == 0 {
            if enemy_heavy != 0 {
                file_safety -= 25;
            } else {
                file_safety -= 10;
            }
        } else if file_offset == 0 {
            let king_rank = flipped_king_sq / 8;
            let file_pawns_ahead = file_pawns & (0xFFFFFFFFFFFFFF00u64 << (king_rank * 8));
            
            if file_pawns_ahead == 0 && enemy_heavy != 0 {
                file_safety -= 15;
            }
        }
    }
    
    Score::new(file_safety, file_safety / 3)
}

fn evaluate_king_mobility(pos: &Position, king_sq: u8, color: Color) -> Score {
    let king_attacks = get_king_attacks(king_sq);
    let our_pieces = pos.pieces(color);
    let enemy_attacks = all_attacks(pos, color.opposite());
    
    let escape_squares = king_attacks & !our_pieces & !enemy_attacks;
    let safe_squares = escape_squares.count_ones() as i32;
    
    let total_squares = (king_attacks & !our_pieces).count_ones() as i32;
    
    let base_mobility = if king_sq == 4 || king_sq == 60 {
        (safe_squares - 2) * 4
    } else {
        (safe_squares - 3) * 6
    };
    
    let trapped_penalty = if safe_squares == 0 && total_squares <= 1 {
        -20
    } else {
        0
    };
    
    let mobility_score = base_mobility + trapped_penalty;
    
    Score::new(mobility_score, mobility_score / 2)
}

pub fn is_king_in_danger(pos: &Position, color: Color) -> bool {
    let king_sq = pos.king_square(color);
    let king_zone = calculate_king_zone(king_sq);
    let enemy_attacks = all_attacks(pos, color.opposite());
    
    let attacked_zone = enemy_attacks & king_zone;
    attacked_zone.count_ones() >= 4
}

pub fn calculate_king_danger_score(pos: &Position, color: Color) -> i32 {
    let safety_score = evaluate_king_safety_for_color(pos, color);
    safety_score.mg
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
    let escape_pawns = our_pawns & back_rank_mask;
    let enemy_heavy_pieces = (enemy_rooks | enemy_queens) & back_rank_mask;
    
    let mut threat_score = 0;
    
    if enemy_heavy_pieces != 0 {
        if escape_pawns == 0 {
            threat_score -= 50;
        } else if escape_pawns.count_ones() == 1 {
            threat_score -= 25;
        }
        
        if enemy_heavy_pieces.count_ones() >= 2 {
            threat_score -= 20;
        }
    }
    
    Score::new(threat_score, threat_score / 2)
}

pub fn evaluate_castling_safety(pos: &Position, color: Color) -> Score {
    let king_sq = pos.king_square(color);
    let flipped_king_sq = flip_square_for_color(king_sq, color);
    let king_file = flipped_king_sq % 8;
    let king_rank = flipped_king_sq / 8;
    
    let (kingside_castle, queenside_castle) = if king_rank == 0 {
        (king_file >= 6, king_file <= 2)
    } else {
        (false, false)
    };
    
    if !kingside_castle && !queenside_castle {
        let on_starting_square = flipped_king_sq == 4;
        
        if on_starting_square {
            return Score::new(-5, 0);
        } else {
            return Score::new(-25, 0);
        }
    }
    
    let mut castle_safety = Score::new(25, 0);
    
    if kingside_castle {
        castle_safety = castle_safety.add(evaluate_kingside_structure(pos, color));
    } else if queenside_castle {
        castle_safety = castle_safety.add(evaluate_queenside_structure(pos, color));
    }
    
    castle_safety
}

fn evaluate_kingside_structure(pos: &Position, color: Color) -> Score {
    let flipped_pawns = get_flipped_pawns(pos, color);
    let mut structure_score = 0;
    
    let pawn_squares = [13, 14, 15];
    
    for &square in &pawn_squares {
        if (flipped_pawns & (1u64 << square)) != 0 {
            structure_score += 10;
        } else {
            structure_score -= 12;
        }
    }
    
    Score::new(structure_score, 0)
}

fn evaluate_queenside_structure(pos: &Position, color: Color) -> Score {
    let flipped_pawns = get_flipped_pawns(pos, color);
    let mut structure_score = 0;
    
    let pawn_squares = [8, 9, 10];
    
    for &square in &pawn_squares {
        if (flipped_pawns & (1u64 << square)) != 0 {
            structure_score += 8;
        } else {
            structure_score -= 10;
        }
    }
    
    Score::new(structure_score, 0)
}
