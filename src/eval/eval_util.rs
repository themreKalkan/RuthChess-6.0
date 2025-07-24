
use crate::board::position::Position;
use crate::eval::material::{ calculate_phase, MAX_PHASE};
use std::sync::Once;
use crate::eval::evaluate::Score;

#[derive(Debug, Clone, Copy)]
pub struct EvalCacheEntry {
    pub hash: u64,
    pub score: Score,
    pub phase: u32,
}

impl Default for EvalCacheEntry {
    fn default() -> Self {
        Self {
            hash: 0,
            score: Score::zero(),
            phase: 0,
        }
    }
}

const EVAL_CACHE_SIZE: usize = 1024 * 1024*16;
static mut EVAL_CACHE: [EvalCacheEntry; EVAL_CACHE_SIZE] = [EvalCacheEntry {
    hash: 0,
    score: Score { mg: 0, eg: 0 },
    phase: 0,
}; EVAL_CACHE_SIZE];

static EVAL_CACHE_INIT: Once = Once::new();

pub struct EvalCache;

impl EvalCache {
    pub fn probe(hash: u64) -> Option<(Score, u32)> {
        unsafe {
            let index = (hash as usize) & (EVAL_CACHE_SIZE - 1);
            let cache_ptr = std::ptr::addr_of!(EVAL_CACHE);
            let entry = &(*cache_ptr)[index];
            
            if entry.hash == hash {
                Some((entry.score, entry.phase))
            } else {
                None
            }
        }
    }
    
    pub fn store(hash: u64, score: Score, phase: u32) {
        unsafe {
            let index = (hash as usize) & (EVAL_CACHE_SIZE - 1);
            let cache_ptr = std::ptr::addr_of_mut!(EVAL_CACHE);
            (*cache_ptr)[index] = EvalCacheEntry { hash, score, phase };
        }
    }
    
    pub fn clear() {
        unsafe {
            let cache_ptr = std::ptr::addr_of_mut!(EVAL_CACHE);
            for i in 0..EVAL_CACHE_SIZE {
                (*cache_ptr)[i] = EvalCacheEntry::default();
            }
        }
    }
}

pub fn init_eval_cache() {
    EVAL_CACHE_INIT.call_once(|| {
        EvalCache::clear();
    });
}

#[inline(always)]
pub fn interpolate_score(score: Score, phase: u32) -> i32 {
    score.interpolate(phase)
}

#[inline(always)]
pub fn phase_value(pos: &Position) -> u32 {
    calculate_phase(pos)
}

#[inline(always)]
pub fn is_endgame(phase: u32) -> bool {
    phase < MAX_PHASE / 4
}

#[inline(always)]
pub fn is_middlegame(phase: u32) -> bool {
    phase > (MAX_PHASE * 3) / 4
}

pub fn is_opening(pos: &Position) -> bool {
    let phase = calculate_phase(pos);
    if phase < (MAX_PHASE * 7) / 8 {
        return false;
    }
    
    let white_knights = pos.pieces_colored(crate::board::position::PieceType::Knight, crate::board::position::Color::White);
    let black_knights = pos.pieces_colored(crate::board::position::PieceType::Knight, crate::board::position::Color::Black);
    let white_bishops = pos.pieces_colored(crate::board::position::PieceType::Bishop, crate::board::position::Color::White);
    let black_bishops = pos.pieces_colored(crate::board::position::PieceType::Bishop, crate::board::position::Color::Black);
    
    let white_starting_knights = 0x42u64;
    let black_starting_knights = 0x4200000000000000u64;
    let white_starting_bishops = 0x24u64;
    let black_starting_bishops = 0x2400000000000000u64;
    
    let undeveloped_pieces = (white_knights & white_starting_knights).count_ones() +
                           (black_knights & black_starting_knights).count_ones() +
                           (white_bishops & white_starting_bishops).count_ones() +
                           (black_bishops & black_starting_bishops).count_ones();
    
    undeveloped_pieces >= 6
}

pub fn scale_evaluation(score: i32, pos: &Position) -> i32 {
    let white_material = calculate_material_value(pos, crate::board::position::Color::White);
    let black_material = calculate_material_value(pos, crate::board::position::Color::Black);
    
    let material_diff = (white_material - black_material).abs();
    if material_diff < 300 {
        return score;
    }
    
    if is_likely_draw(pos) {
        return score / 4;
    }
    
    let total_material = white_material + black_material;
    if total_material < 2000 {
        let scale_factor = (total_material as f32 / 2000.0).sqrt();
        (score as f32 * scale_factor) as i32
    } else {
        score
    }
}

fn calculate_material_value(pos: &Position, color: crate::board::position::Color) -> i32 {
    use crate::board::position::PieceType;
    
    let mut material = 0;
    material += pos.piece_count(color, PieceType::Pawn) as i32 * 100;
    material += pos.piece_count(color, PieceType::Knight) as i32 * 320;
    material += pos.piece_count(color, PieceType::Bishop) as i32 * 330;
    material += pos.piece_count(color, PieceType::Rook) as i32 * 500;
    material += pos.piece_count(color, PieceType::Queen) as i32 * 900;
    
    material
}

pub fn is_likely_draw(pos: &Position) -> bool {
    use crate::board::position::{PieceType, Color};
    
    let white_pawns = pos.piece_count(Color::White, PieceType::Pawn);
    let black_pawns = pos.piece_count(Color::Black, PieceType::Pawn);
    
    if white_pawns == 0 && black_pawns == 0 {
        let white_material = pos.piece_count(Color::White, PieceType::Knight) +
                           pos.piece_count(Color::White, PieceType::Bishop) +
                           pos.piece_count(Color::White, PieceType::Rook) * 2 +
                           pos.piece_count(Color::White, PieceType::Queen) * 4;
        
        let black_material = pos.piece_count(Color::Black, PieceType::Knight) +
                           pos.piece_count(Color::Black, PieceType::Bishop) +
                           pos.piece_count(Color::Black, PieceType::Rook) * 2 +
                           pos.piece_count(Color::Black, PieceType::Queen) * 4;
        
        if white_material <= 1 && black_material <= 1 {
            return true;
        }
        
        if white_material == 1 && black_material == 1 &&
           pos.piece_count(Color::White, PieceType::Bishop) == 1 &&
           pos.piece_count(Color::Black, PieceType::Bishop) == 1 {
            return are_bishops_same_color(pos);
        }
    }
    
    if pos.halfmove_clock >= 80 {
        return true;
    }
    
    false
}

fn are_bishops_same_color(pos: &Position) -> bool {
    use crate::board::position::{PieceType, Color};
    
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

pub fn apply_contempt(score: i32, pos: &Position) -> i32 {
    const CONTEMPT_VALUE: i32 = 10;
    
    if score.abs() < 50 {
        if pos.side_to_move == crate::board::position::Color::White {
            score + CONTEMPT_VALUE
        } else {
            score - CONTEMPT_VALUE
        }
    } else {
        score
    }
}

pub fn normalize_score(score: i32) -> i32 {
    const MAX_EVAL: i32 = 32000;
    score.clamp(-MAX_EVAL, MAX_EVAL)
}

pub fn get_mobility_factor(pos: &Position) -> f32 {
    use crate::board::position::{PieceType, Color};
    use crate::movegen::magic::{get_knight_attacks, get_bishop_attacks, get_rook_attacks, get_queen_attacks};
    
    let mut total_mobility = 0;
    let mut piece_count = 0;
    
    for color in [Color::White, Color::Black] {
        let all_pieces = pos.all_pieces();
        let our_pieces = pos.pieces(color);
        
        let mut knights = pos.pieces_colored(PieceType::Knight, color);
        while knights != 0 {
            let square = knights.trailing_zeros() as u8;
            knights &= knights - 1;
            total_mobility += (get_knight_attacks(square) & !our_pieces).count_ones();
            piece_count += 1;
        }
        
        let mut bishops = pos.pieces_colored(PieceType::Bishop, color);
        while bishops != 0 {
            let square = bishops.trailing_zeros() as u8;
            bishops &= bishops - 1;
            total_mobility += (get_bishop_attacks(square, all_pieces) & !our_pieces).count_ones();
            piece_count += 1;
        }
        
        let mut rooks = pos.pieces_colored(PieceType::Rook, color);
        while rooks != 0 {
            let square = rooks.trailing_zeros() as u8;
            rooks &= rooks - 1;
            total_mobility += (get_rook_attacks(square, all_pieces) & !our_pieces).count_ones();
            piece_count += 1;
        }
        
        let mut queens = pos.pieces_colored(PieceType::Queen, color);
        while queens != 0 {
            let square = queens.trailing_zeros() as u8;
            queens &= queens - 1;
            total_mobility += (get_queen_attacks(square, all_pieces) & !our_pieces).count_ones();
            piece_count += 1;
        }
    }
    
    if piece_count > 0 {
        total_mobility as f32 / piece_count as f32
    } else {
        0.0
    }
}

pub fn calculate_complexity(pos: &Position) -> i32 {
    use crate::board::position::{PieceType, Color};
    
    let mut complexity = 0;
    
    for piece_type in 1..6 {
        let piece_type = PieceType::from(piece_type);
        complexity += (pos.piece_count(Color::White, piece_type) + 
                      pos.piece_count(Color::Black, piece_type)) as i32 * 2;
    }
    
    let white_pawns = pos.pieces_colored(PieceType::Pawn, Color::White);
    let black_pawns = pos.pieces_colored(PieceType::Pawn, Color::Black);
    
    complexity += count_pawn_islands(white_pawns) * 3;
    complexity += count_pawn_islands(black_pawns) * 3;
    
    if !is_king_safe(pos, Color::White) {
        complexity += 10;
    }
    if !is_king_safe(pos, Color::Black) {
        complexity += 10;
    }
    
    complexity
}

fn count_pawn_islands(pawns: crate::board::bitboard::Bitboard) -> i32 {
    let mut islands = 0;
    let mut in_island = false;
    
    for file in 0..8 {
        let file_pawns = pawns & (0x0101010101010101u64 << file);
        
        if file_pawns != 0 {
            if !in_island {
                islands += 1;
                in_island = true;
            }
        } else {
            in_island = false;
        }
    }
    
    islands
}

fn is_king_safe(pos: &Position, color: crate::board::position::Color) -> bool {
    use crate::movegen::magic::all_attacks;
    
    let king_sq = pos.king_square(color);
    let enemy_attacks = all_attacks(pos, color.opposite());
    
    let king_zone = crate::movegen::magic::get_king_attacks(king_sq);
    let attacked_zone = enemy_attacks & king_zone;
    
    attacked_zone.count_ones() <= 2
}

#[derive(Debug, Clone, Copy)]
pub struct EvalFlags {
    pub is_opening: bool,
    pub is_middlegame: bool,
    pub is_endgame: bool,
    pub is_drawn: bool,
    pub white_ahead: bool,
    pub complex_position: bool,
}

impl EvalFlags {
    pub fn from_position(pos: &Position) -> Self {
        let phase = calculate_phase(pos);
        let white_material = calculate_material_value(pos, crate::board::position::Color::White);
        let black_material = calculate_material_value(pos, crate::board::position::Color::Black);
        let complexity = calculate_complexity(pos);
        
        Self {
            is_opening: is_opening(pos),
            is_middlegame: is_middlegame(phase),
            is_endgame: is_endgame(phase),
            is_drawn: is_likely_draw(pos),
            white_ahead: white_material > black_material + 100,
            complex_position: complexity > 50,
        }
    }
}

#[derive(Debug)]
pub struct EvalDebugInfo {
    pub material: Score,
    pub positional: Score,
    pub mobility: Score,
    pub king_safety: Score,
    pub pawn_structure: Score,
    pub threats: Score,
    pub space: Score,
    pub phase: u32,
    pub flags: EvalFlags,
}

impl EvalDebugInfo {
    pub fn new() -> Self {
        Self {
            material: Score::zero(),
            positional: Score::zero(),
            mobility: Score::zero(),
            king_safety: Score::zero(),
            pawn_structure: Score::zero(),
            threats: Score::zero(),
            space: Score::zero(),
            phase: 0,
            flags: EvalFlags {
                is_opening: false,
                is_middlegame: false,
                is_endgame: false,
                is_drawn: false,
                white_ahead: false,
                complex_position: false,
            },
        }
    }
    
    pub fn total_mg(&self) -> i32 {
        self.material.mg + self.positional.mg + self.mobility.mg + 
        self.king_safety.mg + self.pawn_structure.mg + self.threats.mg + self.space.mg
    }
    
    pub fn total_eg(&self) -> i32 {
        self.material.eg + self.positional.eg + self.mobility.eg + 
        self.king_safety.eg + self.pawn_structure.eg + self.threats.eg + self.space.eg
    }
    
    pub fn interpolated(&self) -> i32 {
        let total = Score::new(self.total_mg(), self.total_eg());
        total.interpolate(self.phase)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::position::Position;
    
    #[test]
    fn test_eval_cache() {
        init_eval_cache();
        
        let hash = 0x1234567890ABCDEFu64;
        let score = Score::new(100, 150);
        let phase = 128;
        
        assert!(EvalCache::probe(hash).is_none());
        
        EvalCache::store(hash, score, phase);
        let result = EvalCache::probe(hash);
        assert!(result.is_some());
        
        let (retrieved_score, retrieved_phase) = result.unwrap();
        assert_eq!(retrieved_score.mg, score.mg);
        assert_eq!(retrieved_score.eg, score.eg);
        assert_eq!(retrieved_phase, phase);
    }
    
    #[test]
    fn test_phase_detection() {
        let pos = Position::startpos();
        assert!(is_opening(&pos));
        assert!(!is_endgame(phase_value(&pos)));
    }
    
    #[test]
    fn test_score_scaling() {
        let pos = Position::startpos();
        let score = 100;
        let scaled = scale_evaluation(score, &pos);
        
        assert!((scaled - score).abs() < 20);
    }
    
    #[test]
    fn test_draw_detection() {
        let pos = Position::startpos();
        assert!(!is_likely_draw(&pos));
        
        let pos = Position::from_fen("8/8/8/8/8/8/8/4k1K1 w - - 0 1").unwrap();
        assert!(is_likely_draw(&pos));
    }
    
    #[test]
    fn test_complexity_calculation() {
        let pos = Position::startpos();
        let complexity = calculate_complexity(&pos);
        
        assert!(complexity > 30);
        assert!(complexity < 100);
    }
}