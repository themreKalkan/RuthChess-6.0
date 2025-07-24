
use crate::board::position::{Position, PieceType, Color};
use crate::board::bitboard::Bitboard;
use crate::eval::evaluate::Score;



pub const PIECE_VALUES: [Score; 7] = [
    Score::new(0, 0),
    Score::new(124, 206),
    Score::new(781, 854),
    Score::new(825, 915),
    Score::new(1276, 1380),
    Score::new(2538, 2682),
    Score::new(0, 0),
];

pub const MAX_PHASE: u32 = 256;
const PHASE_VALUES: [u32; 7] = [0, 0, 1, 1, 2, 4, 0];

#[inline(always)]
pub fn calculate_phase(pos: &Position) -> u32 {
    const PHASE_VALUES: [u32; 6] = [0, 1, 1, 2, 4, 0];
    const MAX_PHASE: u32 = 24;

    let mut phase = 0;

    for piece_type in [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
        let idx = piece_type as usize;
        let white_count = pos.piece_count(Color::White, piece_type);
        let black_count = pos.piece_count(Color::Black, piece_type);
        phase += (white_count + black_count) * PHASE_VALUES[idx];
    }
    phase = (phase/MAX_PHASE)*256;

    phase.min(256)
}


#[inline(always)]
pub fn evaluate_material(pos: &Position) -> Score {
    let mut score = Score::zero();
    
    for color in [Color::White, Color::Black] {
        let mut material = Score::zero();
        
        material = material.add(Score::new(
            pos.piece_count(color, PieceType::Pawn) as i32 * PIECE_VALUES[1].mg,
            pos.piece_count(color, PieceType::Pawn) as i32 * PIECE_VALUES[1].eg,
        ));
        
        material = material.add(Score::new(
            pos.piece_count(color, PieceType::Knight) as i32 * PIECE_VALUES[2].mg,
            pos.piece_count(color, PieceType::Knight) as i32 * PIECE_VALUES[2].eg,
        ));
        
        material = material.add(Score::new(
            pos.piece_count(color, PieceType::Bishop) as i32 * PIECE_VALUES[3].mg,
            pos.piece_count(color, PieceType::Bishop) as i32 * PIECE_VALUES[3].eg,
        ));
        
        material = material.add(Score::new(
            pos.piece_count(color, PieceType::Rook) as i32 * PIECE_VALUES[4].mg,
            pos.piece_count(color, PieceType::Rook) as i32 * PIECE_VALUES[4].eg,
        ));
        
        material = material.add(Score::new(
            pos.piece_count(color, PieceType::Queen) as i32 * PIECE_VALUES[5].mg,
            pos.piece_count(color, PieceType::Queen) as i32 * PIECE_VALUES[5].eg,
        ));
        
        if color == Color::White {
            score = score.add(material);
        } else {
            score = score.sub(material);
        }
    }
    
    score
}

const BISHOP_PAIR_BONUS: Score = Score::new(60, 70);

#[inline(always)]
pub fn evaluate_bishop_pair(pos: &Position) -> Score {
    let mut score = Score::zero();
    
    if pos.piece_count(Color::White, PieceType::Bishop) >= 2 {
        score = score.add(BISHOP_PAIR_BONUS);
    }
    
    if pos.piece_count(Color::Black, PieceType::Bishop) >= 2 {
        score = score.sub(BISHOP_PAIR_BONUS);
    }
    
    score
}

pub fn evaluate_minor_piece_adjustments(pos: &Position) -> Score {
    let mut score = Score::zero();
    
    let total_pawns = pos.piece_count(Color::White, PieceType::Pawn) + 
                     pos.piece_count(Color::Black, PieceType::Pawn);
    
    let knight_bonus = Score::new(
        (total_pawns as i32 - 8) * 4,
        (total_pawns as i32 - 8) * 2
    );
    
    let white_knights = pos.piece_count(Color::White, PieceType::Knight) as i32;
    let black_knights = pos.piece_count(Color::Black, PieceType::Knight) as i32;
    
    score = score.add(Score::new(
        (white_knights - black_knights) * knight_bonus.mg,
        (white_knights - black_knights) * knight_bonus.eg,
    ));
    
    score
}

pub fn evaluate_piece_imbalance(pos: &Position) -> Score {
    let mut score = Score::zero();
    
    let white_rooks = pos.piece_count(Color::White, PieceType::Rook) as i32;
    let black_rooks = pos.piece_count(Color::Black, PieceType::Rook) as i32;
    let white_minors = (pos.piece_count(Color::White, PieceType::Knight) + 
                       pos.piece_count(Color::White, PieceType::Bishop)) as i32;
    let black_minors = (pos.piece_count(Color::Black, PieceType::Knight) + 
                       pos.piece_count(Color::Black, PieceType::Bishop)) as i32;
    
    let rook_vs_minors = Score::new(-25, 15);
    
    let white_rook_imbalance = white_rooks.saturating_sub(black_minors / 2);
    let black_rook_imbalance = black_rooks.saturating_sub(white_minors / 2);
    
    score = score.add(Score::new(
        (white_rook_imbalance - black_rook_imbalance) * rook_vs_minors.mg,
        (white_rook_imbalance - black_rook_imbalance) * rook_vs_minors.eg,
    ));
    
    score
}

pub fn is_material_draw(pos: &Position) -> bool {
    let white_pawns = pos.piece_count(Color::White, PieceType::Pawn);
    let black_pawns = pos.piece_count(Color::Black, PieceType::Pawn);
    
    if white_pawns == 0 && black_pawns == 0 {
        let white_knights = pos.piece_count(Color::White, PieceType::Knight);
        let black_knights = pos.piece_count(Color::Black, PieceType::Knight);
        let white_bishops = pos.piece_count(Color::White, PieceType::Bishop);
        let black_bishops = pos.piece_count(Color::Black, PieceType::Bishop);
        let white_rooks = pos.piece_count(Color::White, PieceType::Rook);
        let black_rooks = pos.piece_count(Color::Black, PieceType::Rook);
        let white_queens = pos.piece_count(Color::White, PieceType::Queen);
        let black_queens = pos.piece_count(Color::Black, PieceType::Queen);
        
        let white_material = white_knights + white_bishops + white_rooks * 2 + white_queens * 4;
        let black_material = black_knights + black_bishops + black_rooks * 2 + black_queens * 4;
        
        if white_material <= 1 && black_material <= 1 {
            return true;
        }
        
        if white_material == 1 && black_material == 1 && 
           white_bishops == 1 && black_bishops == 1 {
            return are_bishops_same_color(pos);
        }
    }
    
    false
}

fn are_bishops_same_color(pos: &Position) -> bool {
    let white_bishops = pos.pieces_colored(PieceType::Bishop, Color::White);
    let black_bishops = pos.pieces_colored(PieceType::Bishop, Color::Black);
    
    if white_bishops.count_ones() != 1 || black_bishops.count_ones() != 1 {
        return false;
    }
    
    let white_square = white_bishops.trailing_zeros() as u8;
    let black_square = black_bishops.trailing_zeros() as u8;
    
    let white_color = (white_square / 8 + white_square % 8) % 2;
    let black_color = (black_square / 8 + black_square % 8) % 2;
    
    white_color == black_color
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::position::Position;
    
    #[test]
    fn test_score_operations() {
        let s1 = Score::new(100, 200);
        let s2 = Score::new(50, 75);
        
        assert_eq!(s1 + s2, Score::new(150, 275));
        assert_eq!(s1 - s2, Score::new(50, 125));
        assert_eq!(-s1, Score::new(-100, -200));
    }
    
    #[test]
    fn test_interpolation() {
        let score = Score::new(100, 200);
        assert_eq!(score.interpolate(MAX_PHASE), 100);
        assert_eq!(score.interpolate(0), 200);
        assert_eq!(score.interpolate(MAX_PHASE / 2), 150);
    }
    
    #[test]
    fn test_phase_calculation() {
        let pos = Position::startpos();
        let phase = calculate_phase(&pos);
        assert_eq!(phase, MAX_PHASE);
    }
    
    #[test]
    fn test_material_draw() {
        let pos = Position::from_fen("8/8/8/8/8/8/8/4k1K1 w - - 0 1").unwrap();
        assert!(is_material_draw(&pos));
        
        let pos = Position::startpos();
        assert!(!is_material_draw(&pos));
    }
}