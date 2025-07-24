
use crate::board::position::{Position, Color, PieceType};
use crate::board::bitboard::{Bitboard, EMPTY,FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H,RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8};
use crate::eval::evaluate::Score;
use crate::eval::pst::{file_of, rank_of, relative_rank};

const ISOLATED_PAWN_PENALTY: Score = Score::new(-5, -10);
const DOUBLED_PAWN_PENALTY: Score = Score::new(-10, -20);
const BACKWARD_PAWN_PENALTY: Score = Score::new(-8, -12);
const CONNECTED_PAWN_BONUS: Score = Score::new(7, 8);
const PASSED_PAWN_BONUS: [Score; 8] = [
    Score::new(0, 0),
    Score::new(0, 0),
    Score::new(10, 15),
    Score::new(15, 25),
    Score::new(25, 40),
    Score::new(40, 65),
    Score::new(65, 100),
    Score::new(0, 0),
];

const PAWN_CHAIN_BASE: [Score; 8] = [
    Score::new(0, 0),
    Score::new(0, 0),
    Score::new(12, 8),
    Score::new(18, 12),
    Score::new(25, 18),
    Score::new(35, 25),
    Score::new(45, 35),
    Score::new(60, 50),
];

const PAWN_STORM_BONUS: Score = Score::new(15, 5);
const PAWN_SHIELD_BONUS: Score = Score::new(20, 5);

#[derive(Debug, Clone)]
pub struct FileMasks {
    pub files: [Bitboard; 8],
    pub adjacent_files: [Bitboard; 8],
    pub isolated_mask: [Bitboard; 8],
    pub passed_mask: [[Bitboard; 64]; 2],
    pub outpost_mask: [[Bitboard; 64]; 2],
}

static mut FILE_MASKS: Option<FileMasks> = None;

pub fn init_pawn_masks() {
    unsafe {
        let masks_ptr = std::ptr::addr_of!(FILE_MASKS);
        if (*masks_ptr).is_some() {
            return;
        }
        
        let mut masks = FileMasks {
            files: [0; 8],
            adjacent_files: [0; 8],
            isolated_mask: [0; 8],
            passed_mask: [[0; 64]; 2],
            outpost_mask: [[0; 64]; 2],
        };
        
        for file in 0..8 {
            masks.files[file] = 0x0101010101010101u64 << file;
            
            if file > 0 {
                masks.adjacent_files[file] |= masks.files[file - 1];
            }
            if file < 7 {
                masks.adjacent_files[file] |= masks.files[file + 1];
            }
            
            masks.isolated_mask[file] = masks.files[file] | masks.adjacent_files[file];
        }
        
        for sq in 0..64 {
            let file = file_of(sq as u8) as usize;
            let rank = rank_of(sq as u8);
            
            let mut mask = 0u64;
            for r in (rank + 1)..8 {
                mask |= masks.files[file] & (0xFFu64 << (r * 8));
                if file > 0 {
                    mask |= masks.files[file - 1] & (0xFFu64 << (r * 8));
                }
                if file < 7 {
                    mask |= masks.files[file + 1] & (0xFFu64 << (r * 8));
                }
            }
            masks.passed_mask[Color::White as usize][sq] = mask;
            
            mask = 0u64;
            for r in 0..rank {
                mask |= masks.files[file] & (0xFFu64 << (r * 8));
                if file > 0 {
                    mask |= masks.files[file - 1] & (0xFFu64 << (r * 8));
                }
                if file < 7 {
                    mask |= masks.files[file + 1] & (0xFFu64 << (r * 8));
                }
            }
            masks.passed_mask[Color::Black as usize][sq] = mask;
            
            masks.outpost_mask[Color::White as usize][sq] = 
                calculate_outpost_mask(sq as u8, Color::White);
            masks.outpost_mask[Color::Black as usize][sq] = 
                calculate_outpost_mask(sq as u8, Color::Black);
        }
        
        let masks_ptr_mut = std::ptr::addr_of_mut!(FILE_MASKS);
        (*masks_ptr_mut) = Some(masks);
    }
}

fn calculate_outpost_mask(square: u8, color: Color) -> Bitboard {
    let file = file_of(square) as usize;
    let rank = rank_of(square);
    let mut mask = 0u64;
    
    let direction = match color {
        Color::White => -1i8,
        Color::Black => 1i8,
    };
    
    for f in (file.saturating_sub(1))..=(file + 1).min(7) {
        if f != file {
            let defender_rank = (rank as i8 + direction) as u8;
            if defender_rank < 8 {
                mask |= 1u64 << (defender_rank * 8 + f as u8);
            }
        }
    }
    
    mask
}

pub fn evaluate_pawns(pos: &Position) -> Score {
    init_pawn_masks();
    
    let mut score = Score::zero();
    
    let white_pawns = pos.pieces_colored(PieceType::Pawn, Color::White);
    let black_pawns = pos.pieces_colored(PieceType::Pawn, Color::Black);
    
    score = score.add(evaluate_pawn_structure(pos, Color::White, white_pawns, black_pawns));
    
    score = score.sub(evaluate_pawn_structure(pos, Color::Black, black_pawns, white_pawns));
    
    score
}

fn evaluate_pawn_structure(pos: &Position, color: Color, our_pawns: Bitboard, enemy_pawns: Bitboard) -> Score {
    let mut score = Score::zero();
    let mut pawns_bb = our_pawns;
    
    while pawns_bb != 0 {
        let square = pawns_bb.trailing_zeros() as u8;
        pawns_bb &= pawns_bb - 1;
        
        let file = file_of(square) as usize;
        let rank = rank_of(square);
        let rel_rank = relative_rank(square, color);
        
        unsafe {
            let masks_ptr = std::ptr::addr_of!(FILE_MASKS);
            let masks = (*masks_ptr).as_ref().unwrap();
            
            if (our_pawns & masks.adjacent_files[file]) == 0 {
                score = score.add(ISOLATED_PAWN_PENALTY);
            }
            
            if (our_pawns & masks.files[file]).count_ones() > 1 {
                score = score.add(DOUBLED_PAWN_PENALTY);
            }
            
            if is_backward_pawn(square, color, our_pawns, enemy_pawns, masks) {
                score = score.add(BACKWARD_PAWN_PENALTY);
            }
            
            if (enemy_pawns & masks.passed_mask[color as usize][square as usize]) == 0 {
                let bonus_idx = rel_rank.min(7) as usize;
                score = score.add(PASSED_PAWN_BONUS[bonus_idx]);
                
                if rel_rank >= 5 {
                    let extra = Score::new(
                        (rel_rank as i32 - 4) * 10,
                        (rel_rank as i32 - 4) * 20
                    );
                    score = score.add(extra);
                }
            }
            
            if is_connected_pawn(square, color, our_pawns) {
                let mut bonus = CONNECTED_PAWN_BONUS;
                bonus.mg = bonus.mg * (rel_rank as i32 + 2) / 4;
                bonus.eg = bonus.eg * (rel_rank as i32 + 2) / 4;
                score = score.add(bonus);
            }
        }
    }
    
    score = score.add(evaluate_pawn_chains(our_pawns, color));
    
    score = score.add(evaluate_pawn_storms(pos, color));
    
    score
}

fn is_backward_pawn(square: u8, color: Color, our_pawns: Bitboard, enemy_pawns: Bitboard, masks: &FileMasks) -> bool {
    let file = file_of(square) as usize;
    let rank = rank_of(square);
    
    let advance_square = match color {
        Color::White => {
            if rank >= 7 { return false; }
            square + 8
        }
        Color::Black => {
            if rank == 0 { return false; }
            square - 8
        }
    };
    
    let enemy_pawn_attacks = match color {
        Color::White => {
            let left_attack = if file > 0 { 
                (enemy_pawns & masks.files[file - 1]) >> 7 
            } else { 0 };
            let right_attack = if file < 7 { 
                (enemy_pawns & masks.files[file + 1]) >> 9 
            } else { 0 };
            left_attack | right_attack
        }
        Color::Black => {
            let left_attack = if file > 0 { 
                (enemy_pawns & masks.files[file - 1]) << 9 
            } else { 0 };
            let right_attack = if file < 7 { 
                (enemy_pawns & masks.files[file + 1]) << 7 
            } else { 0 };
            left_attack | right_attack
        }
    };
    
    if (enemy_pawn_attacks & (1u64 << advance_square)) != 0 {
        let support_mask = masks.adjacent_files[file];
        let behind_ranks = match color {
            Color::White => (0xFFu64 << ((rank.saturating_sub(1)) * 8)) >> 8,
            Color::Black => 0xFFu64 >> ((8 - rank - 1) * 8),
        };
        
        return (our_pawns & support_mask & behind_ranks) == 0;
    }
    
    false
}

fn is_connected_pawn(square: u8, color: Color, our_pawns: Bitboard) -> bool {
    let file = file_of(square) as usize;
    let rank = rank_of(square);
    
    let same_rank_mask = 0xFFu64 << (rank * 8);
    let adjacent_files = if file > 0 { 1u64 << ((rank * 8) + (file - 1) as u8) } else { 0 }
                       | if file < 7 { 1u64 << ((rank * 8) + (file + 1) as u8) } else { 0 };
    
    if (our_pawns & adjacent_files) != 0 {
        return true;
    }
    
    let support_rank = match color {
        Color::White => if rank > 0 { rank - 1 } else { return false; },
        Color::Black => if rank < 7 { rank + 1 } else { return false; },
    };
    
    let diagonal_support = if file > 0 { 1u64 << ((support_rank * 8) + (file - 1) as u8) } else { 0 }
                         | if file < 7 { 1u64 << ((support_rank * 8) + (file + 1) as u8) } else { 0 };
    
    (our_pawns & diagonal_support) != 0
}

fn evaluate_pawn_chains(our_pawns: Bitboard, color: Color) -> Score {
    let mut chains = Vec::new();
    let mut processed = 0u64;
    
    let mut pawns_bb = our_pawns;
    while pawns_bb != 0 {
        let square = pawns_bb.trailing_zeros() as u8;
        pawns_bb &= pawns_bb - 1;
        
        if (processed & (1u64 << square)) != 0 {
            continue;
        }
        
        let chain_length = trace_pawn_chain(square, our_pawns, color, &mut processed);
        if chain_length >= 2 {
            chains.push(chain_length);
        }
    }
    
    let mut score = Score::zero();
    for &length in &chains {
        let idx = (length - 1).min(7) as usize;
        score = score.add(PAWN_CHAIN_BASE[idx]);
    }
    
    score
}

fn trace_pawn_chain(start_square: u8, pawns: Bitboard, color: Color, processed: &mut Bitboard) -> usize {
    let mut length = 0;
    let mut current = start_square;
    
    loop {
        *processed |= 1u64 << current;
        length += 1;
        
        let file = file_of(current) as usize;
        let rank = rank_of(current);
        
        let next_rank = match color {
            Color::White => rank + 1,
            Color::Black => rank.saturating_sub(1),
        };
        
        if next_rank >= 8 {
            break;
        }
        
        let mut found_next = false;
        for next_file in (file.saturating_sub(1))..=(file + 1).min(7) {
            let next_square = next_rank * 8 + next_file as u8;
            if (pawns & (1u64 << next_square)) != 0 && 
               (*processed & (1u64 << next_square)) == 0 {
                current = next_square;
                found_next = true;
                break;
            }
        }
        
        if !found_next {
            break;
        }
    }
    
    length
}

fn evaluate_pawn_storms(pos: &Position, color: Color) -> Score {
    let mut score = Score::zero();
    
    let enemy_king_sq = pos.king_square(color.opposite());
    let king_file = file_of(enemy_king_sq) as usize;
    let our_pawns = pos.pieces_colored(PieceType::Pawn, color);
    
    for file_offset in -1..=1 {
        let target_file = (king_file as i32 + file_offset) as usize;
        if target_file >= 8 {
            continue;
        }
        
        let file_mask = 0x0101010101010101u64 << target_file;
        let storm_pawns = our_pawns & file_mask;
        
        if storm_pawns != 0 {
            let most_advanced = match color {
                Color::White => storm_pawns.leading_zeros() / 8,
                Color::Black => storm_pawns.trailing_zeros() / 8,
            };
            
            let storm_bonus = Score::new(
                PAWN_STORM_BONUS.mg * (8 - most_advanced as i32) / 8,
                PAWN_STORM_BONUS.eg * (8 - most_advanced as i32) / 8,
            );
            score = score.add(storm_bonus);
        }
    }
    
    score
}

pub fn evaluate_king_pawn_shelter(pos: &Position, color: Color) -> Score {
    let king_sq = pos.king_square(color);
    let king_file = file_of(king_sq) as usize;
    let our_pawns = pos.pieces_colored(PieceType::Pawn, color);
    
    let mut score = Score::zero();
    
    for file_offset in -1..=1 {
        let shield_file = (king_file as i32 + file_offset) as usize;
        if shield_file >= 8 {
            continue;
        }
        
        let file_mask = 0x0101010101010101u64 << shield_file;
        let shield_pawns = our_pawns & file_mask;
        
        if shield_pawns != 0 {
            let closest_pawn = match color {
                Color::White => shield_pawns.trailing_zeros() / 8,
                Color::Black => shield_pawns.leading_zeros() / 8,
            };
            
            let distance = (closest_pawn as i32 - rank_of(king_sq) as i32).abs();
            if distance <= 2 {
                let shelter_bonus = Score::new(
                    PAWN_SHIELD_BONUS.mg * (3 - distance) / 3,
                    PAWN_SHIELD_BONUS.eg * (3 - distance) / 3,
                );
                score = score.add(shelter_bonus);
            }
        }
    }
    
    score
}

pub fn is_outpost_square(pos: &Position, square: u8, color: Color) -> bool {
    init_pawn_masks();
    
    let enemy_pawns = pos.pieces_colored(PieceType::Pawn, color.opposite());
    
    unsafe {
        let masks_ptr = std::ptr::addr_of!(FILE_MASKS);
        let masks = (*masks_ptr).as_ref().unwrap();
        (enemy_pawns & masks.outpost_mask[color as usize][square as usize]) == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::position::Position;
    
    #[test]
    fn test_pawn_mask_initialization() {
        init_pawn_masks();
        unsafe {
            let masks_ptr = std::ptr::addr_of!(FILE_MASKS);
            let masks = (*masks_ptr).as_ref().unwrap();
            
            assert_eq!(masks.files[0], FILE_A);
            assert_eq!(masks.files[7], FILE_H);
            
            assert_eq!(masks.adjacent_files[0], FILE_B);
            assert_eq!(masks.adjacent_files[7], FILE_G);
            assert_eq!(masks.adjacent_files[3], FILE_C | FILE_E);
        }
    }
    
    #[test]
    fn test_connected_pawns() {
        init_pawn_masks();
        let pos = Position::startpos();
        
        let white_pawns = pos.pieces_colored(PieceType::Pawn, Color::White);
        assert!(is_connected_pawn(8, Color::White, white_pawns));
        assert!(is_connected_pawn(15, Color::White, white_pawns));
    }
    
    #[test]
    fn test_pawn_evaluation() {
        init_pawn_masks();
        let pos = Position::startpos();
        let score = evaluate_pawns(&pos);
        
        assert!(score.mg.abs() < 50);
        assert!(score.eg.abs() < 50);
    }
}