
use crate::board::position::{Position, PieceType, Color};
use crate::board::bitboard::{Bitboard, EMPTY, CENTER, EXTENDED_CENTER};
use crate::eval::evaluate::Score;
use crate::movegen::magic::{all_attacks, get_pawn_attacks};

const SPACE_BONUS: Score = Score::new(2, 1);
const CENTER_SPACE_BONUS: Score = Score::new(4, 2);
const ADVANCED_SPACE_BONUS: Score = Score::new(3, 1);
const SAFE_SPACE_BONUS: Score = Score::new(1, 1);

const WHITE_TERRITORY: Bitboard = 0x0000_FFFF_FFFF_0000;
const BLACK_TERRITORY: Bitboard = 0x0000_FFFF_FFFF_0000;
const WHITE_ADVANCED: Bitboard = 0x0000_0000_FFFF_0000;
const BLACK_ADVANCED: Bitboard = 0x0000_FFFF_0000_0000;

const CENTRAL_FILES: Bitboard = 0x1818181818181818;
const WING_FILES: Bitboard = 0xC3C3C3C3C3C3C3C3;

pub fn evaluate_space(pos: &Position) -> Score {
    let mut score = Score::zero();
    
    let white_space = calculate_space_for_color(pos, Color::White);
    let black_space = calculate_space_for_color(pos, Color::Black);
    
    score = score.add(white_space).sub(black_space);
    
    score = score.add(evaluate_central_control(pos));
    score = score.add(evaluate_file_control(pos));
    score = score.add(evaluate_outpost_control(pos));
    
    score
}

// Public wrapper for per-color space evaluation
pub fn evaluate_space_for_color(pos: &Position, color: Color) -> Score {
    calculate_space_for_color(pos, color)
}

fn calculate_space_for_color(pos: &Position, color: Color) -> Score {
    let mut space_score = Score::zero();
    
    let our_attacks = all_attacks(pos, color);
    let enemy_attacks = all_attacks(pos, color.opposite());
    
    let (territory_mask, advanced_mask) = match color {
        Color::White => (WHITE_TERRITORY, WHITE_ADVANCED),
        Color::Black => (BLACK_TERRITORY, BLACK_ADVANCED),
    };
    
    let controlled_territory = our_attacks & territory_mask;
    let controlled_advanced = our_attacks & advanced_mask;
    let controlled_center = our_attacks & CENTER;
    let controlled_extended = our_attacks & EXTENDED_CENTER;
    
    let territory_count = controlled_territory.count_ones() as i32;
    space_score = space_score.add(Score::new(
        territory_count * SPACE_BONUS.mg,
        territory_count * SPACE_BONUS.eg,
    ));
    
    let advanced_count = controlled_advanced.count_ones() as i32;
    space_score = space_score.add(Score::new(
        advanced_count * ADVANCED_SPACE_BONUS.mg,
        advanced_count * ADVANCED_SPACE_BONUS.eg,
    ));
    
    let center_count = controlled_center.count_ones() as i32;
    space_score = space_score.add(Score::new(
        center_count * CENTER_SPACE_BONUS.mg,
        center_count * CENTER_SPACE_BONUS.eg,
    ));
    
    let extended_center_count = (controlled_extended & !CENTER).count_ones() as i32;
    space_score = space_score.add(Score::new(
        extended_center_count * (CENTER_SPACE_BONUS.mg / 2),
        extended_center_count * (CENTER_SPACE_BONUS.eg / 2),
    ));
    
    let safe_squares = our_attacks & territory_mask & !enemy_attacks;
    let safe_count = safe_squares.count_ones() as i32;
    space_score = space_score.add(Score::new(
        safe_count * SAFE_SPACE_BONUS.mg,
        safe_count * SAFE_SPACE_BONUS.eg,
    ));
    
    space_score
}

fn evaluate_central_control(pos: &Position) -> Score {
    let mut score = Score::zero();
    
    const CENTRAL_SQUARES: [u8; 4] = [27, 28, 35, 36];
    
    for &square in &CENTRAL_SQUARES {
        let square_bb = 1u64 << square;
        
        let (piece_type, piece_color) = pos.piece_at(square);
        if piece_type != PieceType::None {
            let occupation_bonus = match piece_type {
                PieceType::Pawn => Score::new(20, 15),
                PieceType::Knight => Score::new(25, 20),
                PieceType::Bishop => Score::new(15, 12),
                _ => Score::new(10, 8),
            };
            
            if piece_color == Color::White {
                score = score.add(occupation_bonus);
            } else {
                score = score.sub(occupation_bonus);
            }
        }
        
        let white_attacks = pos.is_square_attacked(square, Color::White);
        let black_attacks = pos.is_square_attacked(square, Color::Black);
        
        let control_bonus = Score::new(8, 5);
        if white_attacks && !black_attacks {
            score = score.add(control_bonus);
        } else if black_attacks && !white_attacks {
            score = score.sub(control_bonus);
        }
    }
    
    score
}

fn evaluate_file_control(pos: &Position) -> Score {
    let mut score = Score::zero();
    
    for file in 0..8 {
        let file_mask = 0x0101010101010101u64 << file;
        let file_control = evaluate_single_file_control(pos, file_mask);
        
        let file_weight = if (file_mask & CENTRAL_FILES) != 0 {
            Score::new(15, 10)
        } else {
            Score::new(8, 5)
        };
        
        score = score.add(Score::new(
            file_control * file_weight.mg,
            file_control * file_weight.eg,
        ));
    }
    
    score
}

fn evaluate_single_file_control(pos: &Position, file_mask: Bitboard) -> i32 {
    let white_pawns = pos.pieces_colored(PieceType::Pawn, Color::White) & file_mask;
    let black_pawns = pos.pieces_colored(PieceType::Pawn, Color::Black) & file_mask;
    let white_rooks = pos.pieces_colored(PieceType::Rook, Color::White) & file_mask;
    let black_rooks = pos.pieces_colored(PieceType::Rook, Color::Black) & file_mask;
    let white_queens = pos.pieces_colored(PieceType::Queen, Color::White) & file_mask;
    let black_queens = pos.pieces_colored(PieceType::Queen, Color::Black) & file_mask;
    
    let mut control_score = 0;
    
    if white_pawns == 0 && black_pawns == 0 {
        if white_rooks != 0 || white_queens != 0 {
            control_score += 3;
        }
        if black_rooks != 0 || black_queens != 0 {
            control_score -= 3;
        }
    } else if white_pawns == 0 {
        if white_rooks != 0 || white_queens != 0 {
            control_score += 2;
        }
    } else if black_pawns == 0 {
        if black_rooks != 0 || black_queens != 0 {
            control_score -= 2;
        }
    }
    
    let white_pawn_count = white_pawns.count_ones() as i32;
    let black_pawn_count = black_pawns.count_ones() as i32;
    control_score += white_pawn_count - black_pawn_count;
    
    control_score
}

fn evaluate_outpost_control(pos: &Position) -> Score {
    let mut score = Score::zero();
    
    const WHITE_OUTPOSTS: [u8; 8] = [26, 28, 34, 36, 35, 37, 42, 44];
    const BLACK_OUTPOSTS: [u8; 8] = [19, 21, 27, 29, 28, 30, 35, 37];
    
    for &square in &WHITE_OUTPOSTS {
        if is_good_outpost(pos, square, Color::White) {
            let (piece_type, piece_color) = pos.piece_at(square);
            if piece_color == Color::White {
                let outpost_bonus = match piece_type {
                    PieceType::Knight => Score::new(30, 20),
                    PieceType::Bishop => Score::new(20, 15),
                    _ => Score::new(10, 8),
                };
                score = score.add(outpost_bonus);
            }
        }
    }
    
    for &square in &BLACK_OUTPOSTS {
        if is_good_outpost(pos, square, Color::Black) {
            let (piece_type, piece_color) = pos.piece_at(square);
            if piece_color == Color::Black {
                let outpost_bonus = match piece_type {
                    PieceType::Knight => Score::new(30, 20),
                    PieceType::Bishop => Score::new(20, 15),
                    _ => Score::new(10, 8),
                };
                score = score.sub(outpost_bonus);
            }
        }
    }
    
    score
}

fn is_good_outpost(pos: &Position, square: u8, color: Color) -> bool {
    let file = square % 8;
    let rank = square / 8;
    
    let enemy_territory = match color {
        Color::White => rank >= 4,
        Color::Black => rank <= 3,
    };
    
    if !enemy_territory {
        return false;
    }
    
    let our_pawns = pos.pieces_colored(PieceType::Pawn, color);
    let support_squares = match color {
        Color::White => {
            let mut support = 0u64;
            if rank > 0 {
                if file > 0 {
                    support |= 1u64 << ((rank - 1) * 8 + file - 1);
                }
                if file < 7 {
                    support |= 1u64 << ((rank - 1) * 8 + file + 1);
                }
            }
            support
        }
        Color::Black => {
            let mut support = 0u64;
            if rank < 7 {
                if file > 0 {
                    support |= 1u64 << ((rank + 1) * 8 + file - 1);
                }
                if file < 7 {
                    support |= 1u64 << ((rank + 1) * 8 + file + 1);
                }
            }
            support
        }
    };
    
    if (our_pawns & support_squares) == 0 {
        return false;
    }
    
    let enemy_pawns = pos.pieces_colored(PieceType::Pawn, color.opposite());
    let enemy_pawn_attacks = match color {
        Color::White => {
            let mut attacks = 0u64;
            if rank < 7 {
                if file > 0 {
                    attacks |= 1u64 << ((rank + 1) * 8 + file - 1);
                }
                if file < 7 {
                    attacks |= 1u64 << ((rank + 1) * 8 + file + 1);
                }
            }
            attacks
        }
        Color::Black => {
            let mut attacks = 0u64;
            if rank > 0 {
                if file > 0 {
                    attacks |= 1u64 << ((rank - 1) * 8 + file - 1);
                }
                if file < 7 {
                    attacks |= 1u64 << ((rank - 1) * 8 + file + 1);
                }
            }
            attacks
        }
    };
    
    (enemy_pawns & enemy_pawn_attacks) == 0
}

pub fn calculate_space_pressure(pos: &Position, color: Color) -> i32 {
    let our_attacks = all_attacks(pos, color);
    
    let enemy_territory = match color {
        Color::White => 0xFFFF_0000_0000_0000,
        Color::Black => 0x0000_0000_0000_FFFF,
    };
    
    (our_attacks & enemy_territory).count_ones() as i32
}

pub fn evaluate_pawn_space(pos: &Position) -> Score {
    let mut score = Score::zero();
    
    let white_pawns = pos.pieces_colored(PieceType::Pawn, Color::White);
    let black_pawns = pos.pieces_colored(PieceType::Pawn, Color::Black);
    
    let mut white_advanced = 0;
    let mut white_pawns_bb = white_pawns;
    while white_pawns_bb != 0 {
        let square = white_pawns_bb.trailing_zeros() as u8;
        white_pawns_bb &= white_pawns_bb - 1;
        
        let rank = square / 8;
        if rank >= 4 {
            white_advanced += (rank - 3) as i32;
        }
    }
    
    let mut black_advanced = 0;
    let mut black_pawns_bb = black_pawns;
    while black_pawns_bb != 0 {
        let square = black_pawns_bb.trailing_zeros() as u8;
        black_pawns_bb &= black_pawns_bb - 1;
        
        let rank = square / 8;
        if rank <= 3 {
            black_advanced += (4 - rank) as i32;
        }
    }
    
    let pawn_space_bonus = Score::new(3, 2);
    score = score.add(Score::new(
        (white_advanced - black_advanced) * pawn_space_bonus.mg,
        (white_advanced - black_advanced) * pawn_space_bonus.eg,
    ));
    
    score
}

pub fn evaluate_piece_activity(pos: &Position) -> Score {
    let mut score = Score::zero();
    
    for piece_type in [PieceType::Knight, PieceType::Bishop] {
        let mut white_pieces = pos.pieces_colored(piece_type, Color::White);
        while white_pieces != 0 {
            let square = white_pieces.trailing_zeros() as u8;
            white_pieces &= white_pieces - 1;
            
            let activity_bonus = calculate_piece_activity_bonus(square, Color::White);
            score = score.add(activity_bonus);
        }
        
        let mut black_pieces = pos.pieces_colored(piece_type, Color::Black);
        while black_pieces != 0 {
            let square = black_pieces.trailing_zeros() as u8;
            black_pieces &= black_pieces - 1;
            
            let activity_bonus = calculate_piece_activity_bonus(square, Color::Black);
            score = score.sub(activity_bonus);
        }
    }
    
    score
}

fn calculate_piece_activity_bonus(square: u8, color: Color) -> Score {
    let rank = square / 8;
    let file = square % 8;
    
    let central_bonus = if file >= 2 && file <= 5 {
        Score::new(5, 3)
    } else {
        Score::zero()
    };
    
    let advanced_bonus = match color {
        Color::White => {
            if rank >= 4 {
                Score::new((rank - 3) as i32 * 3, (rank - 3) as i32 * 2)
            } else {
                Score::zero()
            }
        }
        Color::Black => {
            if rank <= 3 {
                Score::new((4 - rank) as i32 * 3, (4 - rank) as i32 * 2)
            } else {
                Score::zero()
            }
        }
    };
    
    central_bonus.add(advanced_bonus)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::position::Position;
    
    #[test]
    fn test_space_evaluation() {
        let pos = Position::startpos();
        let score = evaluate_space(&pos);
        
        assert!(score.mg.abs() < 50);
        assert!(score.eg.abs() < 50);
    }
    
    #[test]
    fn test_central_control() {
        let pos = Position::startpos();
        let score = evaluate_central_control(&pos);
        
        assert_eq!(score.mg, 0);
        assert_eq!(score.eg, 0);
    }
    
    #[test]
    fn test_file_control() {
        let pos = Position::startpos();
        let score = evaluate_file_control(&pos);
        
        assert!(score.mg.abs() < 20);
        assert!(score.eg.abs() < 20);
    }
    
    #[test]
    fn test_space_pressure() {
        let pos = Position::startpos();
        let white_pressure = calculate_space_pressure(&pos, Color::White);
        let black_pressure = calculate_space_pressure(&pos, Color::Black);
        
        assert!(white_pressure > 0);
        assert!(black_pressure > 0);
        assert!((white_pressure - black_pressure).abs() < 5);
    }
    
    #[test]
    fn test_outpost_detection() {
        let pos = Position::from_fen("8/8/8/8/8/3P4/8/8 w - - 0 1").unwrap();
        assert!(!is_good_outpost(&pos, 27, Color::White));
    }
}