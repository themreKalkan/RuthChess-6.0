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

pub const PAWN_VALUE_MG: i32 = 124;
pub const PAWN_VALUE_EG: i32 = 206;
pub const KNIGHT_VALUE_MG: i32 = 781;
pub const KNIGHT_VALUE_EG: i32 = 854;
pub const BISHOP_VALUE_MG: i32 = 825;
pub const BISHOP_VALUE_EG: i32 = 915;
pub const ROOK_VALUE_MG: i32 = 1276;
pub const ROOK_VALUE_EG: i32 = 1380;
pub const QUEEN_VALUE_MG: i32 = 2538;
pub const QUEEN_VALUE_EG: i32 = 2682;

pub const MIDGAME_LIMIT: i32 = 15258;
pub const ENDGAME_LIMIT: i32 = 3915;

pub const KNIGHT_PAWN_ADJUSTMENT: [Score; 17] = [
    Score::new(-30, -35),
    Score::new(-25, -30),
    Score::new(-20, -25),
    Score::new(-15, -20),
    Score::new(-10, -15),
    Score::new(-5, -10),
    Score::new(0, -5),
    Score::new(3, 0),
    Score::new(5, 0),
    Score::new(7, 0),
    Score::new(10, 5),
    Score::new(13, 8),
    Score::new(16, 10),
    Score::new(19, 12),
    Score::new(22, 14),
    Score::new(25, 16),
    Score::new(28, 18),
];

pub const BISHOP_PAWN_ADJUSTMENT: [Score; 17] = [
    Score::new(28, 30),
    Score::new(24, 26),
    Score::new(20, 22),
    Score::new(16, 18),
    Score::new(12, 14),
    Score::new(8, 10),
    Score::new(4, 6),
    Score::new(2, 3),
    Score::new(0, 0),
    Score::new(-2, -3),
    Score::new(-4, -6),
    Score::new(-6, -9),
    Score::new(-8, -12),
    Score::new(-10, -15),
    Score::new(-12, -18),
    Score::new(-14, -21),
    Score::new(-16, -24),
];

pub const ROOK_PAWN_ADJUSTMENT: [Score; 17] = [
    Score::new(35, 40),
    Score::new(30, 35),
    Score::new(25, 30),
    Score::new(20, 25),
    Score::new(15, 20),
    Score::new(10, 15),
    Score::new(5, 10),
    Score::new(2, 5),
    Score::new(0, 0),
    Score::new(-3, -5),
    Score::new(-6, -10),
    Score::new(-9, -15),
    Score::new(-12, -20),
    Score::new(-15, -25),
    Score::new(-18, -30),
    Score::new(-21, -35),
    Score::new(-24, -40),
];

pub const BISHOP_PAIR_BONUS: Score = Score::new(48, 56);

pub const REDUNDANT_KNIGHT_PENALTY: Score = Score::new(-8, -12);
pub const REDUNDANT_BISHOP_PENALTY: Score = Score::new(-6, -8);
pub const REDUNDANT_ROOK_PENALTY: Score = Score::new(-10, -5);
pub const REDUNDANT_QUEEN_PENALTY: Score = Score::new(-25, -15);

pub const MAX_PHASE: u32 = 256;
const PHASE_VALUES: [u32; 6] = [0, 1, 1, 2, 4, 0];

#[inline(always)]
pub fn calculate_phase(pos: &Position) -> u32 {
    let npm_w = non_pawn_material(pos, Color::White);
    let npm_b = non_pawn_material(pos, Color::Black);
    let npm = npm_w + npm_b;
    
    if npm <= ENDGAME_LIMIT {
        return 0;
    }
    
    if npm >= MIDGAME_LIMIT {
        return 256;
    }
    
    ((npm - ENDGAME_LIMIT) * 256 / (MIDGAME_LIMIT - ENDGAME_LIMIT)) as u32
}

fn non_pawn_material(pos: &Position, color: Color) -> i32 {
    let knights = pos.piece_count(color, PieceType::Knight) as i32;
    let bishops = pos.piece_count(color, PieceType::Bishop) as i32;
    let rooks = pos.piece_count(color, PieceType::Rook) as i32;
    let queens = pos.piece_count(color, PieceType::Queen) as i32;
    
    knights * KNIGHT_VALUE_MG +
    bishops * BISHOP_VALUE_MG +
    rooks * ROOK_VALUE_MG +
    queens * QUEEN_VALUE_MG
}

#[inline(always)]
pub fn evaluate_material(pos: &Position) -> Score {
    let mut score = Score::zero();
    
    let total_pawns = pos.piece_count(Color::White, PieceType::Pawn) + 
                     pos.piece_count(Color::Black, PieceType::Pawn);
    let pawn_idx = total_pawns.min(16) as usize;
    
    for color in [Color::White, Color::Black] {
        let mut material = Score::zero();
        
        let pawn_count = pos.piece_count(color, PieceType::Pawn);
        material = material.add(Score::new(
            pawn_count as i32 * PIECE_VALUES[1].mg,
            pawn_count as i32 * PIECE_VALUES[1].eg,
        ));
        
        let knight_count = pos.piece_count(color, PieceType::Knight);
        if knight_count > 0 {
            let base_value = PIECE_VALUES[2];
            let adjustment = KNIGHT_PAWN_ADJUSTMENT[pawn_idx];
            material = material.add(Score::new(
                knight_count as i32 * (base_value.mg + adjustment.mg),
                knight_count as i32 * (base_value.eg + adjustment.eg),
            ));
            
            if knight_count >= 3 {
                material = material.add(Score::new(
                    (knight_count as i32 - 2) * REDUNDANT_KNIGHT_PENALTY.mg,
                    (knight_count as i32 - 2) * REDUNDANT_KNIGHT_PENALTY.eg,
                ));
            }
        }
        
        let bishop_count = pos.piece_count(color, PieceType::Bishop);
        if bishop_count > 0 {
            let base_value = PIECE_VALUES[3];
            let adjustment = BISHOP_PAWN_ADJUSTMENT[pawn_idx];
            material = material.add(Score::new(
                bishop_count as i32 * (base_value.mg + adjustment.mg),
                bishop_count as i32 * (base_value.eg + adjustment.eg),
            ));
            
            if bishop_count >= 2 {
                material = material.add(BISHOP_PAIR_BONUS);
                
                if are_bishops_opposite_colors(pos, color) {
                    material = material.add(Score::new(12, 18));
                }
            }
            
            if bishop_count >= 3 {
                material = material.add(REDUNDANT_BISHOP_PENALTY);
            }
        }
        
        let rook_count = pos.piece_count(color, PieceType::Rook);
        if rook_count > 0 {
            let base_value = PIECE_VALUES[4];
            let adjustment = ROOK_PAWN_ADJUSTMENT[pawn_idx];
            material = material.add(Score::new(
                rook_count as i32 * (base_value.mg + adjustment.mg),
                rook_count as i32 * (base_value.eg + adjustment.eg),
            ));
            
            if rook_count >= 3 {
                material = material.add(REDUNDANT_ROOK_PENALTY);
            }
        }
        
        let queen_count = pos.piece_count(color, PieceType::Queen);
        if queen_count > 0 {
            material = material.add(Score::new(
                queen_count as i32 * PIECE_VALUES[5].mg,
                queen_count as i32 * PIECE_VALUES[5].eg,
            ));
            
            if queen_count >= 2 {
                material = material.add(REDUNDANT_QUEEN_PENALTY);
            }
        }
        
        if color == Color::White {
            score = score.add(material);
        } else {
            score = score.sub(material);
        }
    }
    
    score = score.add(evaluate_material_imbalance(pos));
    
    score
}

fn are_bishops_opposite_colors(pos: &Position, color: Color) -> bool {
    let bishops = pos.pieces_colored(PieceType::Bishop, color);
    if bishops.count_ones() != 2 {
        return false;
    }
    
    let bishop1 = bishops.trailing_zeros() as u8;
    let bishop2 = (bishops & (bishops - 1)).trailing_zeros() as u8;
    
    let color1 = (bishop1 / 8 + bishop1 % 8) % 2;
    let color2 = (bishop2 / 8 + bishop2 % 8) % 2;
    
    color1 != color2
}

pub fn evaluate_material_imbalance(pos: &Position) -> Score {
    let mut score = Score::zero();
    
    let white_knights = pos.piece_count(Color::White, PieceType::Knight) as i32;
    let white_bishops = pos.piece_count(Color::White, PieceType::Bishop) as i32;
    let white_rooks = pos.piece_count(Color::White, PieceType::Rook) as i32;
    let white_queens = pos.piece_count(Color::White, PieceType::Queen) as i32;
    
    let black_knights = pos.piece_count(Color::Black, PieceType::Knight) as i32;
    let black_bishops = pos.piece_count(Color::Black, PieceType::Bishop) as i32;
    let black_rooks = pos.piece_count(Color::Black, PieceType::Rook) as i32;
    let black_queens = pos.piece_count(Color::Black, PieceType::Queen) as i32;
    
    if white_queens > black_queens {
        let black_compensation = black_rooks * 2 + black_bishops + black_knights;
        let white_pieces = white_rooks * 2 + white_bishops + white_knights;
        if black_compensation > white_pieces + 2 {
            score = score.sub(Score::new(16, 24));
        }
    }
    
    if black_queens > white_queens {
        let white_compensation = white_rooks * 2 + white_bishops + white_knights;
        let black_pieces = black_rooks * 2 + black_bishops + black_knights;
        if white_compensation > black_pieces + 2 {
            score = score.add(Score::new(16, 24));
        }
    }
    
    let white_minors = white_knights + white_bishops;
    let black_minors = black_knights + black_bishops;
    
    if white_rooks > black_rooks && black_minors > white_minors + 1 {
        let imbalance_factor = (white_rooks - black_rooks).min(black_minors - white_minors - 1);
        score = score.sub(Score::new(imbalance_factor * 10, imbalance_factor * 20));
    }
    
    if black_rooks > white_rooks && white_minors > black_minors + 1 {
        let imbalance_factor = (black_rooks - white_rooks).min(white_minors - black_minors - 1);
        score = score.add(Score::new(imbalance_factor * 10, imbalance_factor * 20));
    }
    
    let bishop_advantage = white_bishops - black_bishops;
    let knight_advantage = white_knights - black_knights;
    
    let total_pawns = pos.piece_count(Color::White, PieceType::Pawn) + 
                     pos.piece_count(Color::Black, PieceType::Pawn);
    
    if total_pawns <= 8 {
        score = score.add(Score::new(bishop_advantage * 8, bishop_advantage * 12));
    } else if total_pawns >= 12 {
        score = score.add(Score::new(knight_advantage * 6, knight_advantage * 8));
    }
    
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
        
        if white_knights + white_bishops + white_rooks + white_queens == 0 &&
           black_knights + black_bishops + black_rooks + black_queens == 0 {
            return true;
        }
        
        if white_queens == 0 && black_queens == 0 && white_rooks == 0 && black_rooks == 0 {
            if (white_knights + white_bishops == 1 && black_knights + black_bishops == 0) ||
               (black_knights + black_bishops == 1 && white_knights + white_bishops == 0) {
                return true;
            }
            
            if white_knights == 0 && black_knights == 0 && 
               white_bishops == 1 && black_bishops == 1 {
                return are_bishops_same_color(pos);
            }
            
            if (white_knights == 2 && white_bishops == 0 && black_knights + black_bishops == 0) ||
               (black_knights == 2 && black_bishops == 0 && white_knights + white_bishops == 0) {
                return true;
            }
        }
    }
    
    if white_pawns > 0 && black_pawns > 0 {
        let white_bishops = pos.piece_count(Color::White, PieceType::Bishop);
        let black_bishops = pos.piece_count(Color::Black, PieceType::Bishop);
        
        if white_bishops == 1 && black_bishops == 1 {
            if !are_bishops_same_color(pos) && is_position_blocked(pos) {
                return true;
            }
        }
    }
    
    if white_pawns + black_pawns <= 3 {
        let white_material = white_material_count(pos);
        let black_material = black_material_count(pos);
        
        if (white_material == 5 && black_material == 3) ||
           (black_material == 5 && white_material == 3) {
            return true;
        }
    }
    
    false
}

fn white_material_count(pos: &Position) -> u32 {
    pos.piece_count(Color::White, PieceType::Knight) * 3 +
    pos.piece_count(Color::White, PieceType::Bishop) * 3 +
    pos.piece_count(Color::White, PieceType::Rook) * 5 +
    pos.piece_count(Color::White, PieceType::Queen) * 9
}

fn black_material_count(pos: &Position) -> u32 {
    pos.piece_count(Color::Black, PieceType::Knight) * 3 +
    pos.piece_count(Color::Black, PieceType::Bishop) * 3 +
    pos.piece_count(Color::Black, PieceType::Rook) * 5 +
    pos.piece_count(Color::Black, PieceType::Queen) * 9
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

fn is_position_blocked(pos: &Position) -> bool {
    let white_pawns = pos.pieces_colored(PieceType::Pawn, Color::White);
    let black_pawns = pos.pieces_colored(PieceType::Pawn, Color::Black);
    
    let mut blocked_files = 0;
    for file in 0..8 {
        let file_mask = 0x0101010101010101u64 << file;
        let white_on_file = white_pawns & file_mask;
        let black_on_file = black_pawns & file_mask;
        
        if white_on_file != 0 && black_on_file != 0 {
            let white_front = white_on_file << 8;
            let black_front = black_on_file >> 8;
            
            if (white_front & black_on_file) != 0 || (black_front & white_on_file) != 0 {
                blocked_files += 1;
            }
        }
    }
    
    blocked_files >= 4
}

pub fn get_material_signature(pos: &Position) -> u64 {
    let mut signature = 0u64;
    
    for color in [Color::White, Color::Black] {
        let color_shift = if color == Color::White { 0 } else { 32 };
        
        signature |= (pos.piece_count(color, PieceType::Pawn) as u64) << (color_shift + 0);
        signature |= (pos.piece_count(color, PieceType::Knight) as u64) << (color_shift + 4);
        signature |= (pos.piece_count(color, PieceType::Bishop) as u64) << (color_shift + 8);
        signature |= (pos.piece_count(color, PieceType::Rook) as u64) << (color_shift + 12);
        signature |= (pos.piece_count(color, PieceType::Queen) as u64) << (color_shift + 16);
    }
    
    signature
}

pub fn recognize_endgame_type(pos: &Position) -> EndgameType {
    let white_pawns = pos.piece_count(Color::White, PieceType::Pawn);
    let black_pawns = pos.piece_count(Color::Black, PieceType::Pawn);
    let white_material = white_material_count(pos);
    let black_material = black_material_count(pos);
    
    if white_pawns > 0 && black_pawns == 0 && black_material == 0 && white_material == 0 {
        return EndgameType::KPK;
    }
    if black_pawns > 0 && white_pawns == 0 && white_material == 0 && black_material == 0 {
        return EndgameType::KPK;
    }
    
    if (white_material == 6 && black_material == 0 && white_pawns == 0 && black_pawns == 0) ||
       (black_material == 6 && white_material == 0 && white_pawns == 0 && black_pawns == 0) {
        let white_knights = pos.piece_count(Color::White, PieceType::Knight);
        let white_bishops = pos.piece_count(Color::White, PieceType::Bishop);
        let black_knights = pos.piece_count(Color::Black, PieceType::Knight);
        let black_bishops = pos.piece_count(Color::Black, PieceType::Bishop);
        
        if (white_knights == 1 && white_bishops == 1) || (black_knights == 1 && black_bishops == 1) {
            return EndgameType::KBNK;
        }
    }
    
    if white_material == 5 && black_material == 5 && white_pawns == 0 && black_pawns == 0 {
        let white_rooks = pos.piece_count(Color::White, PieceType::Rook);
        let black_rooks = pos.piece_count(Color::Black, PieceType::Rook);
        if white_rooks == 1 && black_rooks == 1 {
            return EndgameType::KRKR;
        }
    }
    
    if white_material == 9 && black_material == 9 && white_pawns == 0 && black_pawns == 0 {
        let white_queens = pos.piece_count(Color::White, PieceType::Queen);
        let black_queens = pos.piece_count(Color::Black, PieceType::Queen);
        if white_queens == 1 && black_queens == 1 {
            return EndgameType::KQKQ;
        }
    }
    
    if white_material == 5 && black_material == 0 && black_pawns == 1 && white_pawns == 0 {
        return EndgameType::KRKP;
    }
    if black_material == 5 && white_material == 0 && white_pawns == 1 && black_pawns == 0 {
        return EndgameType::KRKP;
    }
    
    if white_material == 9 && black_material == 0 && black_pawns == 1 && white_pawns == 0 {
        return EndgameType::KQKP;
    }
    if black_material == 9 && white_material == 0 && white_pawns == 1 && black_pawns == 0 {
        return EndgameType::KQKP;
    }
    
    EndgameType::None
}

pub fn get_endgame_scale_factor(pos: &Position, endgame_type: EndgameType) -> f32 {
    match endgame_type {
        EndgameType::KPK => 1.0,
        EndgameType::KBNK => 1.0,
        EndgameType::KRKR => 0.9,
        EndgameType::KQKQ => 0.85,
        EndgameType::KRKP => 0.95,
        EndgameType::KQKP => 0.92,
        EndgameType::None => 1.0,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndgameType {
    None,
    KPK,
    KBNK,
    KRKR,
    KQKQ,
    KRKP,
    KQKP,
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
    fn test_phase_calculation() {
        let pos = Position::startpos();
        let phase = calculate_phase(&pos);
        assert!(phase > 200);
        
        let endgame_pos = Position::from_fen("8/8/8/8/8/8/8/4k1K1 w - - 0 1").unwrap();
        let endgame_phase = calculate_phase(&endgame_pos);
        assert_eq!(endgame_phase, 0);
    }
    
    #[test]
    fn test_material_draw() {
        let pos = Position::from_fen("8/8/8/8/8/8/8/4k1K1 w - - 0 1").unwrap();
        assert!(is_material_draw(&pos));
        
        let pos = Position::from_fen("8/8/8/8/8/8/1B6/4k1K1 w - - 0 1").unwrap();
        assert!(is_material_draw(&pos));
        
        let pos = Position::from_fen("8/8/8/8/8/8/1N6/4k1K1 w - - 0 1").unwrap();
        assert!(is_material_draw(&pos));
        
        let pos = Position::startpos();
        assert!(!is_material_draw(&pos));
    }
    
    #[test]
    fn test_endgame_recognition() {
        let pos = Position::from_fen("8/8/8/8/8/8/1P6/4k1K1 w - - 0 1").unwrap();
        assert_eq!(recognize_endgame_type(&pos), EndgameType::KPK);
        
        let pos = Position::from_fen("8/8/8/8/8/8/8/r3k1KR w - - 0 1").unwrap();
        assert_eq!(recognize_endgame_type(&pos), EndgameType::KRKR);
        
        let pos = Position::startpos();
        assert_eq!(recognize_endgame_type(&pos), EndgameType::None);
    }
    
    #[test]
    fn test_bishop_pair() {
        let pos = Position::from_fen("8/8/8/8/8/8/1BB5/4k1K1 w - - 0 1").unwrap();
        let material = evaluate_material(&pos);
        
        assert!(material.mg > BISHOP_VALUE_MG * 2);
        assert!(material.eg > BISHOP_VALUE_EG * 2);
    }
}
