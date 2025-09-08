use crate::board::position::{Position, Color, PieceType};
use crate::board::bitboard::{Bitboard, EMPTY, FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H};
use crate::eval::evaluate::Score;
use crate::eval::pst::{file_of, rank_of, relative_rank};

const ISOLATED_PAWN_PENALTY: Score = Score::new(-13, -18);
const DOUBLED_PAWN_PENALTY: Score = Score::new(-18, -38);
const DOUBLED_ISOLATED_PENALTY: Score = Score::new(-25, -48);
const TRIPLED_PAWN_PENALTY: Score = Score::new(-35, -58);
const BACKWARD_PAWN_PENALTY: Score = Score::new(-17, -22);
const BACKWARD_ON_OPEN_FILE_PENALTY: Score = Score::new(-25, -32);

const CONNECTED_BONUS: [[Score; 8]; 8] = [
    [Score::new(0, 0); 8],
    [Score::new(7, 0), Score::new(8, 0), Score::new(12, 0), Score::new(29, 0), Score::new(48, 0), Score::new(86, 0), Score::new(0, 0), Score::new(0, 0)],
    [Score::new(7, 0), Score::new(8, 0), Score::new(12, 0), Score::new(29, 0), Score::new(48, 0), Score::new(86, 0), Score::new(0, 0), Score::new(0, 0)],
    [Score::new(9, 0), Score::new(10, 2), Score::new(15, 4), Score::new(31, 9), Score::new(50, 16), Score::new(88, 28), Score::new(0, 0), Score::new(0, 0)],
    [Score::new(11, 3), Score::new(13, 6), Score::new(19, 11), Score::new(35, 19), Score::new(54, 32), Score::new(92, 54), Score::new(0, 0), Score::new(0, 0)],
    [Score::new(14, 7), Score::new(16, 11), Score::new(23, 20), Score::new(41, 33), Score::new(62, 56), Score::new(105, 96), Score::new(0, 0), Score::new(0, 0)],
    [Score::new(17, 12), Score::new(20, 19), Score::new(29, 33), Score::new(51, 55), Score::new(74, 94), Score::new(123, 164), Score::new(0, 0), Score::new(0, 0)],
    [Score::new(0, 0); 8],
];

const PASSED_PAWN_BONUS: [Score; 8] = [
    Score::new(0, 0),
    Score::new(0, 0),
    Score::new(17, 38),
    Score::new(33, 58),
    Score::new(50, 86),
    Score::new(81, 138),
    Score::new(144, 232),
    Score::new(0, 0),
];

const CANDIDATE_PASSED_BONUS: [Score; 8] = [
    Score::new(0, 0),
    Score::new(0, 0),
    Score::new(8, 13),
    Score::new(13, 20),
    Score::new(21, 31),
    Score::new(34, 48),
    Score::new(55, 75),
    Score::new(0, 0),
];

const WEAK_PAWN_ON_FILE: [Score; 8] = [
    Score::new(-3, -7),
    Score::new(-1, -3),
    Score::new(0, -1),
    Score::new(1, 0),
    Score::new(1, 0),
    Score::new(0, -1),
    Score::new(-1, -3),
    Score::new(-3, -7),
];

const SHELTER_STRENGTH: [[i32; 8]; 4] = [
    [  -6,  81,  93,  58,  39,  18,   25, 0],
    [ -43,  61,  35, -49, -29, -11,  -63, 0],
    [ -10,  75,  23,  -2,  32,   3,  -45, 0],
    [ -39, -13, -29, -52, -48, -67, -166, 0],
];

const STORM_DANGER: [[i32; 8]; 4] = [
    [  4,  73, 132, 46, 31, 0, 0, 0],
    [ 26,  57, 119, 31, 24, 0, 0, 0],
    [  5,  45, 100, 39, 12, 0, 0, 0],
    [  0,  17,  66, 28,  0, 0, 0, 0],
];

#[derive(Debug, Clone)]
pub struct FileMasks {
    pub files: [Bitboard; 8],
    pub adjacent_files: [Bitboard; 8],
    pub isolated_mask: [Bitboard; 8],
    pub passed_mask: [[Bitboard; 64]; 2],
    pub outpost_mask: [[Bitboard; 64]; 2],
    pub front_span: [[Bitboard; 64]; 2],
    pub pawn_attack_span: [[Bitboard; 64]; 2],
    pub pawn_attacks: [[Bitboard; 64]; 2],
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
            front_span: [[0; 64]; 2],
            pawn_attack_span: [[0; 64]; 2],
            pawn_attacks: [[0; 64]; 2],
        };
        
        for file in 0..8 {
            masks.files[file] = 0x0101010101010101u64 << file;
            
            if file > 0 {
                masks.adjacent_files[file] |= masks.files[file - 1];
            }
            if file < 7 {
                masks.adjacent_files[file] |= masks.files[file + 1];
            }
            
            masks.isolated_mask[file] = masks.adjacent_files[file];
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
            
            let mut attack_span = 0u64;
            for r in (rank + 1)..8 {
                if file > 0 {
                    attack_span |= 1u64 << (r * 8 + file as u8 - 1);
                }
                if file < 7 {
                    attack_span |= 1u64 << (r * 8 + file as u8 + 1);
                }
            }
            masks.pawn_attack_span[Color::White as usize][sq] = attack_span;
            
            attack_span = 0u64;
            for r in 0..rank {
                if file > 0 {
                    attack_span |= 1u64 << (r * 8 + file as u8 - 1);
                }
                if file < 7 {
                    attack_span |= 1u64 << (r * 8 + file as u8 + 1);
                }
            }
            masks.pawn_attack_span[Color::Black as usize][sq] = attack_span;
            
            let mut attacks = 0u64;
            if rank < 7 {
                if file > 0 {
                    attacks |= 1u64 << (sq + 7);
                }
                if file < 7 {
                    attacks |= 1u64 << (sq + 9);
                }
            }
            masks.pawn_attacks[Color::White as usize][sq] = attacks;
            
            attacks = 0u64;
            if rank > 0 {
                if file > 0 {
                    attacks |= 1u64 << (sq - 9);
                }
                if file < 7 {
                    attacks |= 1u64 << (sq - 7);
                }
            }
            masks.pawn_attacks[Color::Black as usize][sq] = attacks;
            
            masks.outpost_mask[Color::White as usize][sq] = 
                calculate_outpost_mask(sq as u8, Color::White);
            masks.outpost_mask[Color::Black as usize][sq] = 
                calculate_outpost_mask(sq as u8, Color::Black);
            
            let mut front_mask = 0u64;
            for r in (rank + 1)..8 {
                front_mask |= masks.files[file] & (0xFFu64 << (r * 8));
            }
            masks.front_span[Color::White as usize][sq] = front_mask;
            
            front_mask = 0u64;
            for r in 0..rank {
                front_mask |= masks.files[file] & (0xFFu64 << (r * 8));
            }
            masks.front_span[Color::Black as usize][sq] = front_mask;
        }
        
        let masks_ptr_mut = std::ptr::addr_of_mut!(FILE_MASKS);
        (*masks_ptr_mut) = Some(masks);
    }
}

fn calculate_outpost_mask(square: u8, color: Color) -> Bitboard {
    let file = file_of(square) as usize;
    let rank = rank_of(square);
    let mut mask = 0u64;
    
    match color {
        Color::White => {
            if rank > 0 {
                if file > 0 {
                    for r in 0..rank {
                        mask |= 1u64 << (r * 8 + (file as u8) - 1);
                    }
                }
                if file < 7 {
                    for r in 0..rank {
                        mask |= 1u64 << (r * 8 + (file as u8) + 1);
                    }
                }
            }
        },
        Color::Black => {
            if rank < 7 {
                if file > 0 {
                    for r in (rank + 1)..8 {
                        mask |= 1u64 << (r * 8 + (file as u8) - 1);
                    }
                }
                if file < 7 {
                    for r in (rank + 1)..8 {
                        mask |= 1u64 << (r * 8 + (file as u8) + 1);
                    }
                }
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
    
    score = score.add(evaluate_pawn_asymmetry(white_pawns, black_pawns));
    
    score
}

pub fn evaluate_pawn_structure_for_color(pos: &Position, color: Color) -> Score {
    init_pawn_masks();
    
    let our_pawns = pos.pieces_colored(PieceType::Pawn, color);
    let enemy_pawns = pos.pieces_colored(PieceType::Pawn, color.opposite());
    
    evaluate_pawn_structure(pos, color, our_pawns, enemy_pawns)
}

pub fn evaluate_pawn_structure(pos: &Position, color: Color, our_pawns: Bitboard, enemy_pawns: Bitboard) -> Score {
    let mut score = Score::zero();
    let mut pawns_bb = our_pawns;
    
    let mut doubled_count = 0;
    
    for file in 0..8 {
        unsafe {
            let masks_ptr = std::ptr::addr_of!(FILE_MASKS);
            let masks = (*masks_ptr).as_ref().unwrap();
            
            let file_pawns = our_pawns & masks.files[file];
            let pawn_count = file_pawns.count_ones();
            
            if pawn_count >= 2 {
                doubled_count += pawn_count - 1;
                score = score.add(Score::new(
                    DOUBLED_PAWN_PENALTY.mg * (pawn_count as i32 - 1),
                    DOUBLED_PAWN_PENALTY.eg * (pawn_count as i32 - 1)
                ));
                
                if pawn_count >= 3 {
                    score = score.add(TRIPLED_PAWN_PENALTY);
                }
                
                if (our_pawns & masks.adjacent_files[file]) == 0 {
                    score = score.add(DOUBLED_ISOLATED_PENALTY);
                }
            }
        }
    }
    
    while pawns_bb != 0 {
        let square = pawns_bb.trailing_zeros() as u8;
        pawns_bb &= pawns_bb - 1;
        let square_bb = 1u64 << square;
        
        let file = file_of(square) as usize;
        let rank = rank_of(square);
        let rel_rank = relative_rank(square, color);
        
        unsafe {
            let masks_ptr = std::ptr::addr_of!(FILE_MASKS);
            let masks = (*masks_ptr).as_ref().unwrap();
            
            if (our_pawns & masks.adjacent_files[file]) == 0 {
                let mut penalty = ISOLATED_PAWN_PENALTY;
                
                if (enemy_pawns & masks.files[file]) == 0 {
                    penalty = Score::new(penalty.mg * 3 / 2, penalty.eg * 3 / 2);
                }
                
                penalty = penalty.add(WEAK_PAWN_ON_FILE[file]);
                
                score = score.add(penalty);
            }
            
            if is_backward_pawn(square, color, our_pawns, enemy_pawns, masks) {
                let mut penalty = BACKWARD_PAWN_PENALTY;
                
                if (enemy_pawns & masks.files[file]) == 0 {
                    penalty = BACKWARD_ON_OPEN_FILE_PENALTY;
                }
                
                score = score.add(penalty);
            }
            
            if is_connected_pawn(square, color, our_pawns) {
                let supported = is_pawn_supported(square, color, our_pawns);
                let phalanx = is_phalanx_pawn(square, our_pawns);
                
                if (supported || phalanx) && rel_rank >= 2 && rel_rank <= 6 {
                    let mut bonus = CONNECTED_BONUS[rel_rank as usize][file];
                    
                    if supported {
                        bonus = Score::new((bonus.mg * 3) / 2, (bonus.eg * 3) / 2);
                    }
                    
                    score = score.add(bonus);
                }
            }
            
            if (enemy_pawns & masks.passed_mask[color as usize][square as usize]) == 0 {
                let mut bonus = PASSED_PAWN_BONUS[rel_rank.min(7) as usize];
                
                let front_span = masks.front_span[color as usize][square as usize];
                if (front_span & pos.all_pieces()) == 0 {
                    bonus = Score::new((bonus.mg * 5) / 4, (bonus.eg * 5) / 4);
                }
                
                if is_pawn_supported(square, color, our_pawns) {
                    bonus = bonus.add(Score::new(12, 18));
                }
                
                let phase = crate::eval::material::calculate_phase(pos);
                if phase < 128 {
                    let our_king = pos.king_square(color);
                    let enemy_king = pos.king_square(color.opposite());
                    
                    let our_king_distance = king_distance(our_king, square);
                    let enemy_king_distance = king_distance(enemy_king, square);
                    
                    bonus = bonus.add(Score::new(
                        0,
                        (enemy_king_distance as i32 - our_king_distance as i32) * 5
                    ));
                }
                
                score = score.add(bonus);
            }
            else if is_candidate_passed(square, color, our_pawns, enemy_pawns, masks) {
                score = score.add(CANDIDATE_PASSED_BONUS[rel_rank.min(7) as usize]);
            }
        }
    }
    
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
    
    let enemy_pawn_attacks = masks.pawn_attacks[color.opposite() as usize][advance_square as usize];
    
    if (enemy_pawn_attacks & enemy_pawns) == 0 {
        return false;
    }
    
    let support_mask = masks.adjacent_files[file];
    let behind_mask = match color {
        Color::White => {
            let mut mask = 0u64;
            for r in 0..rank {
                mask |= 0xFFu64 << (r * 8);
            }
            mask
        },
        Color::Black => {
            let mut mask = 0u64;
            for r in (rank + 1)..8 {
                mask |= 0xFFu64 << (r * 8);
            }
            mask
        }
    };
    
    (our_pawns & support_mask & behind_mask) == 0
}

fn is_connected_pawn(square: u8, color: Color, our_pawns: Bitboard) -> bool {
    let file = file_of(square) as usize;
    let rank = rank_of(square);
    
    let same_rank_mask = 0xFFu64 << (rank * 8);
    let adjacent_files = if file > 0 { 1u64 << (rank * 8 + (file as u8) - 1) } else { 0 }
                       | if file < 7 { 1u64 << (rank * 8 + (file as u8) + 1) } else { 0 };
    
    if (our_pawns & adjacent_files) != 0 {
        return true;
    }
    
    let support_rank = match color {
        Color::White => if rank > 0 { rank - 1 } else { return false; },
        Color::Black => if rank < 7 { rank + 1 } else { return false; },
    };
    
    let diagonal_support = if file > 0 { 1u64 << (support_rank * 8 + (file as u8) - 1) } else { 0 }
                         | if file < 7 { 1u64 << (support_rank * 8 + (file as u8) + 1) } else { 0 };
    
    (our_pawns & diagonal_support) != 0
}

fn is_phalanx_pawn(square: u8, our_pawns: Bitboard) -> bool {
    let file = square % 8;
    let rank = square / 8;
    
    let left = if file > 0 { 1u64 << (rank * 8 + file - 1) } else { 0 };
    let right = if file < 7 { 1u64 << (rank * 8 + file + 1) } else { 0 };
    
    ((our_pawns & left) != 0) || ((our_pawns & right) != 0)
}

fn is_pawn_supported(square: u8, color: Color, our_pawns: Bitboard) -> bool {
    let file = square % 8;
    let rank = square / 8;
    
    let support_rank = match color {
        Color::White => if rank > 0 { rank - 1 } else { return false; },
        Color::Black => if rank < 7 { rank + 1 } else { return false; },
    };
    
    let support_squares = if file > 0 { 1u64 << (support_rank * 8 + file - 1) } else { 0 }
                        | if file < 7 { 1u64 << (support_rank * 8 + file + 1) } else { 0 };
    
    (our_pawns & support_squares) != 0
}

fn is_candidate_passed(square: u8, color: Color, our_pawns: Bitboard, enemy_pawns: Bitboard, masks: &FileMasks) -> bool {
    let file = file_of(square) as usize;
    let rank = rank_of(square);
    
    let mut support_count = 0;
    let mut defenders = 0;
    
    for adjacent_file in [file.saturating_sub(1), (file + 1).min(7)] {
        if adjacent_file == file { continue; }
        
        let file_pawns = our_pawns & masks.files[adjacent_file];
        let enemy_file_pawns = enemy_pawns & masks.files[adjacent_file];
        
        let mut our_bb = file_pawns;
        while our_bb != 0 {
            let pawn_sq = our_bb.trailing_zeros() as u8;
            our_bb &= our_bb - 1;
            
            let pawn_rank = pawn_sq / 8;
            let can_support = match color {
                Color::White => pawn_rank <= rank,
                Color::Black => pawn_rank >= rank,
            };
            
            if can_support {
                support_count += 1;
            }
        }
        
        let mut enemy_bb = enemy_file_pawns;
        while enemy_bb != 0 {
            let pawn_sq = enemy_bb.trailing_zeros() as u8;
            enemy_bb &= enemy_bb - 1;
            
            let pawn_rank = pawn_sq / 8;
            let can_defend = match color {
                Color::White => pawn_rank > rank,
                Color::Black => pawn_rank < rank,
            };
            
            if can_defend {
                defenders += 1;
            }
        }
    }
    
    support_count >= defenders
}

pub fn evaluate_pawn_asymmetry(white_pawns: Bitboard, black_pawns: Bitboard) -> Score {
    let mut asymmetry = 0;
    
    for file in 0..8 {
        let file_mask = 0x0101010101010101u64 << file;
        let white_on_file = (white_pawns & file_mask).count_ones();
        let black_on_file = (black_pawns & file_mask).count_ones();
        
        if white_on_file != black_on_file {
            asymmetry += 1;
        }
    }
    
    Score::new(asymmetry * 3, asymmetry * 2)
}

pub fn evaluate_king_pawn_shelter(pos: &Position, color: Color) -> Score {
    let king_sq = pos.king_square(color);
    let king_file = file_of(king_sq) as usize;
    let king_rank = rank_of(king_sq);
    let our_pawns = pos.pieces_colored(PieceType::Pawn, color);
    let enemy_pawns = pos.pieces_colored(PieceType::Pawn, color.opposite());
    
    let mut shelter_value = 0i32;
    let mut storm_value = 0i32;
    
    for file_offset in -1i32..=1 {
        let file = (king_file as i32 + file_offset) as usize;
        if file >= 8 {
            continue;
        }
        
        let file_mask = 0x0101010101010101u64 << file;
        let our_file_pawns = our_pawns & file_mask;
        let enemy_file_pawns = enemy_pawns & file_mask;
        
        let our_pawn_rank = if our_file_pawns != 0 {
            match color {
                Color::White => (our_file_pawns.trailing_zeros() / 8) as i32,
                Color::Black => 7 - (our_file_pawns.leading_zeros() / 8) as i32,
            }
        } else {
            -1
        };
        
        let enemy_pawn_rank = if enemy_file_pawns != 0 {
            match color {
                Color::White => 7 - (enemy_file_pawns.leading_zeros() / 8) as i32,
                Color::Black => (enemy_file_pawns.trailing_zeros() / 8) as i32,
            }
        } else {
            -1
        };
        
        if our_pawn_rank >= 0 {
            let distance = (our_pawn_rank - king_rank as i32).abs().min(3) as usize;
            let file_index = (file_offset + 1) as usize;
            if file_index < 4 && distance < 8 {
                shelter_value += SHELTER_STRENGTH[file_index][distance];
            }
        }
        
        if enemy_pawn_rank >= 0 {
            let storm_rank = match color {
                Color::White => enemy_pawn_rank,
                Color::Black => 7 - enemy_pawn_rank,
            };
            if storm_rank >= 2 && storm_rank <= 5 {
                let file_index = (file_offset + 1) as usize;
                if file_index < 4 {
                    storm_value += STORM_DANGER[file_index][(storm_rank - 2) as usize];
                }
            }
        }
    }
    
    Score::new(shelter_value - storm_value, 0)
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

fn king_distance(from: u8, to: u8) -> u8 {
    let from_file = from % 8;
    let from_rank = from / 8;
    let to_file = to % 8;
    let to_rank = to / 8;
    
    let file_dist = if from_file > to_file { from_file - to_file } else { to_file - from_file };
    let rank_dist = if from_rank > to_rank { from_rank - to_rank } else { to_rank - from_rank };
    
    file_dist.max(rank_dist)
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
        let pos = Position::from_fen("8/8/8/8/3PP3/8/8/8 w - - 0 1").unwrap();
        
        let white_pawns = pos.pieces_colored(PieceType::Pawn, Color::White);
        assert!(is_connected_pawn(27, Color::White, white_pawns));
        assert!(is_connected_pawn(28, Color::White, white_pawns));
    }
    
    #[test]
    fn test_pawn_evaluation() {
        init_pawn_masks();
        let pos = Position::startpos();
        let score = evaluate_pawns(&pos);
        
        assert!(score.mg.abs() < 50);
        assert!(score.eg.abs() < 50);
    }
    
    #[test]
    fn test_passed_pawn_detection() {
        init_pawn_masks();
        let pos = Position::from_fen("8/8/8/4P3/8/8/8/8 w - - 0 1").unwrap();
        let score = evaluate_pawns(&pos);
        
        assert!(score.mg > 0);
        assert!(score.eg > 0);
    }
}
