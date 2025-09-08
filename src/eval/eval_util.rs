use crate::board::position::{Position, Color, PieceType};
use crate::eval::evaluate::Score;
use crate::eval::material::{calculate_phase, MAX_PHASE};
use std::sync::RwLock;
use once_cell::sync::Lazy;

pub struct EvalCache {
    entries: Vec<EvalCacheEntry>,
    size: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct EvalCacheEntry {
    hash: u64,
    score: i32,
    depth: u8,
}

impl EvalCache {
    pub fn new(size: usize) -> Self {
        let actual_size = size.next_power_of_two();
        Self {
            entries: vec![EvalCacheEntry::default(); actual_size],
            size: actual_size,
        }
    }
    
    pub fn probe(&self, hash: u64) -> Option<i32> {
        let index = (hash as usize) & (self.size - 1);
        let entry = &self.entries[index];
        
        if entry.hash == hash {
            Some(entry.score)
        } else {
            None
        }
    }
    
    pub fn store(&mut self, hash: u64, score: i32, depth: u8) {
        let index = (hash as usize) & (self.size - 1);
        let entry = &mut self.entries[index];
        
        if entry.hash != hash || depth >= entry.depth {
            *entry = EvalCacheEntry { hash, score, depth };
        }
    }
    
    pub fn clear(&mut self) {
        for entry in &mut self.entries {
            *entry = EvalCacheEntry::default();
        }
    }
}

static GLOBAL_EVAL_CACHE: Lazy<RwLock<EvalCache>> = Lazy::new(|| {
    RwLock::new(EvalCache::new(1024 * 1024))
});

pub fn init_eval_cache() {
}

pub fn probe_eval_cache(hash: u64) -> Option<i32> {
    GLOBAL_EVAL_CACHE.read().unwrap().probe(hash)
}

pub fn store_eval_cache(hash: u64, score: i32, depth: u8) {
    GLOBAL_EVAL_CACHE.write().unwrap().store(hash, score, depth);
}

pub fn clear_eval_cache() {
    GLOBAL_EVAL_CACHE.write().unwrap().clear();
}

#[inline(always)]
pub fn phase_value(phase: u32) -> f32 {
    phase as f32 / MAX_PHASE as f32
}

#[inline(always)]
pub fn is_endgame(phase: u32) -> bool {
    phase < 64
}

#[inline(always)]
pub fn is_middlegame(phase: u32) -> bool {
    phase >= 128
}

#[inline(always)]
pub fn is_opening(phase: u32) -> bool {
    phase >= 200
}

const CONTEMPT_VALUE: i32 = 12;

pub fn apply_contempt(score: i32, pos: &Position) -> i32 {
    if pos.side_to_move == Color::White {
        if score > 0 {
            score + CONTEMPT_VALUE
        } else {
            score - CONTEMPT_VALUE / 2
        }
    } else {
        if score < 0 {
            score - CONTEMPT_VALUE
        } else {
            score + CONTEMPT_VALUE / 2
        }
    }
}

pub fn scale_evaluation(score: i32, pos: &Position) -> i32 {
    let phase = calculate_phase(pos);
    
    let scale_factor = calculate_scale_factor(pos, phase);
    
    (score as f32 * scale_factor) as i32
}

fn calculate_scale_factor(pos: &Position, phase: u32) -> f32 {
    if !is_endgame(phase) {
        return 1.0;
    }
    
    let white_pawns = pos.piece_count(Color::White, PieceType::Pawn);
    let black_pawns = pos.piece_count(Color::Black, PieceType::Pawn);
    
    let white_bishops = pos.piece_count(Color::White, PieceType::Bishop);
    let black_bishops = pos.piece_count(Color::Black, PieceType::Bishop);
    
    if white_bishops == 1 && black_bishops == 1 {
        if are_opposite_colored_bishops(pos) {
            if white_pawns == 0 && black_pawns == 0 {
                return 0.1;
            }
            return 0.5;
        }
    }
    
    let white_rooks = pos.piece_count(Color::White, PieceType::Rook);
    let black_rooks = pos.piece_count(Color::Black, PieceType::Rook);
    
    if white_rooks == 1 && black_rooks == 1 {
        if white_pawns <= 1 && black_pawns <= 1 {
            return 0.6;
        }
        return 0.85;
    }
    
    let white_queens = pos.piece_count(Color::White, PieceType::Queen);
    let black_queens = pos.piece_count(Color::Black, PieceType::Queen);
    
    if white_queens == 1 && black_queens == 1 {
        if white_pawns + black_pawns <= 2 {
            return 0.5;
        }
        return 0.75;
    }
    
    let white_minors = pos.piece_count(Color::White, PieceType::Knight) + 
                      pos.piece_count(Color::White, PieceType::Bishop);
    let black_minors = pos.piece_count(Color::Black, PieceType::Knight) + 
                      pos.piece_count(Color::Black, PieceType::Bishop);
    
    if white_minors == 1 && black_minors == 1 && white_pawns == 0 && black_pawns == 0 {
        return 0.2;
    }
    
    1.0
}

fn are_opposite_colored_bishops(pos: &Position) -> bool {
    let white_bishops = pos.pieces_colored(PieceType::Bishop, Color::White);
    let black_bishops = pos.pieces_colored(PieceType::Bishop, Color::Black);
    
    if white_bishops.count_ones() != 1 || black_bishops.count_ones() != 1 {
        return false;
    }
    
    let white_square = white_bishops.trailing_zeros() as u8;
    let black_square = black_bishops.trailing_zeros() as u8;
    
    let white_color = (white_square / 8 + white_square % 8) % 2;
    let black_color = (black_square / 8 + black_square % 8) % 2;
    
    white_color != black_color
}

pub fn is_likely_draw(pos: &Position) -> bool {
    if pos.halfmove_clock >= 100 {
        return true;
    }
    
    if is_insufficient_material(pos) {
        return true;
    }
    
    let phase = calculate_phase(pos);
    if is_endgame(phase) {
        if is_drawish_endgame(pos) {
            return true;
        }
    }
    
    if is_fortress(pos) {
        return true;
    }
    
    
    false
}

fn is_insufficient_material(pos: &Position) -> bool {
    let white_pawns = pos.piece_count(Color::White, PieceType::Pawn);
    let black_pawns = pos.piece_count(Color::Black, PieceType::Pawn);
    let white_knights = pos.piece_count(Color::White, PieceType::Knight);
    let black_knights = pos.piece_count(Color::Black, PieceType::Knight);
    let white_bishops = pos.piece_count(Color::White, PieceType::Bishop);
    let black_bishops = pos.piece_count(Color::Black, PieceType::Bishop);
    let white_rooks = pos.piece_count(Color::White, PieceType::Rook);
    let black_rooks = pos.piece_count(Color::Black, PieceType::Rook);
    let white_queens = pos.piece_count(Color::White, PieceType::Queen);
    let black_queens = pos.piece_count(Color::Black, PieceType::Queen);
    
    if white_pawns == 0 && black_pawns == 0 && 
       white_rooks == 0 && black_rooks == 0 && 
       white_queens == 0 && black_queens == 0 {
        
        if white_knights + white_bishops == 0 && black_knights + black_bishops == 0 {
            return true;
        }
        
        if (white_knights + white_bishops == 1 && black_knights + black_bishops == 0) ||
           (black_knights + black_bishops == 1 && white_knights + white_bishops == 0) {
            return true;
        }
        
        if (white_knights == 2 && white_bishops == 0 && black_knights + black_bishops == 0) ||
           (black_knights == 2 && black_bishops == 0 && white_knights + white_bishops == 0) {
            return true;
        }
        
        if white_knights == 0 && black_knights == 0 && 
           white_bishops == 1 && black_bishops == 1 {
            if !are_opposite_colored_bishops(pos) {
                return true;
            }
        }
    }
    
    false
}

fn is_drawish_endgame(pos: &Position) -> bool {
    let white_pawns = pos.piece_count(Color::White, PieceType::Pawn);
    let black_pawns = pos.piece_count(Color::Black, PieceType::Pawn);
    let white_bishops = pos.piece_count(Color::White, PieceType::Bishop);
    let black_bishops = pos.piece_count(Color::Black, PieceType::Bishop);
    let white_knights = pos.piece_count(Color::White, PieceType::Knight);
    let black_knights = pos.piece_count(Color::Black, PieceType::Knight);
    let white_rooks = pos.piece_count(Color::White, PieceType::Rook);
    let black_rooks = pos.piece_count(Color::Black, PieceType::Rook);
    
    if white_bishops == 1 && black_bishops == 1 && are_opposite_colored_bishops(pos) {
        if white_pawns + black_pawns <= 4 {
            if are_pawns_blocked(pos) {
                return true;
            }
        }
    }
    
    if white_rooks == 1 && black_rooks == 1 && white_pawns == 0 && black_pawns == 0 {
        return true;
    }
    
    if white_pawns == 0 && black_pawns == 0 {
        if (white_knights == 1 && white_bishops == 0 && black_knights == 0 && black_bishops == 1) ||
           (black_knights == 1 && black_bishops == 0 && white_knights == 0 && white_bishops == 1) {
            return true;
        }
    }
    
    false
}

fn are_pawns_blocked(pos: &Position) -> bool {
    let white_pawns = pos.pieces_colored(PieceType::Pawn, Color::White);
    let black_pawns = pos.pieces_colored(PieceType::Pawn, Color::Black);
    
    let white_advanced = white_pawns << 8;
    let black_advanced = black_pawns >> 8;
    
    let blocked = (white_advanced & black_pawns) | (black_advanced & white_pawns);
    
    blocked.count_ones() >= (white_pawns | black_pawns).count_ones() / 2
}

pub fn normalize_score(score: i32) -> i32 {
    score.max(-29000).min(29000)
}

pub fn calculate_complexity(pos: &Position) -> i32 {
    let mut complexity = 0;
    
    let white_material = calculate_material_value(pos, Color::White);
    let black_material = calculate_material_value(pos, Color::Black);
    let imbalance = (white_material - black_material).abs();
    
    complexity += imbalance / 100;
    
    let white_pawns = pos.pieces_colored(PieceType::Pawn, Color::White);
    let black_pawns = pos.pieces_colored(PieceType::Pawn, Color::Black);
    
    complexity += count_pawn_islands(white_pawns) * 5;
    complexity += count_pawn_islands(black_pawns) * 5;
    
    complexity += count_passed_pawns(pos, Color::White) * 10;
    complexity += count_passed_pawns(pos, Color::Black) * 10;
    
    let white_pieces = count_active_pieces(pos, Color::White);
    let black_pieces = count_active_pieces(pos, Color::Black);
    complexity += (white_pieces + black_pieces) * 3;
    
    let queens = pos.piece_count(Color::White, PieceType::Queen) + 
                pos.piece_count(Color::Black, PieceType::Queen);
    complexity += queens as i32 * 15;
    
    if is_king_exposed(pos, Color::White) {
        complexity += 20;
    }
    if is_king_exposed(pos, Color::Black) {
        complexity += 20;
    }
    
    complexity
}

fn calculate_material_value(pos: &Position, color: Color) -> i32 {
    pos.piece_count(color, PieceType::Pawn) as i32 * 100 +
    pos.piece_count(color, PieceType::Knight) as i32 * 320 +
    pos.piece_count(color, PieceType::Bishop) as i32 * 330 +
    pos.piece_count(color, PieceType::Rook) as i32 * 500 +
    pos.piece_count(color, PieceType::Queen) as i32 * 900
}

fn count_pawn_islands(pawns: u64) -> i32 {
    let mut islands = 0;
    let mut in_island = false;
    
    for file in 0..8 {
        let file_mask = 0x0101010101010101u64 << file;
        if (pawns & file_mask) != 0 {
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

fn count_passed_pawns(pos: &Position, color: Color) -> i32 {
    let our_pawns = pos.pieces_colored(PieceType::Pawn, color);
    let enemy_pawns = pos.pieces_colored(PieceType::Pawn, color.opposite());
    
    let mut passed = 0;
    let mut pawns_bb = our_pawns;
    
    while pawns_bb != 0 {
        let square = pawns_bb.trailing_zeros() as u8;
        pawns_bb &= pawns_bb - 1;
        
        if is_passed_pawn(square, color, enemy_pawns) {
            passed += 1;
        }
    }
    
    passed
}

fn is_passed_pawn(square: u8, color: Color, enemy_pawns: u64) -> bool {
    let file = square % 8;
    let rank = square / 8;
    
    let mut mask = 0u64;
    
    for f in file.saturating_sub(1)..=(file + 1).min(7) {
        if color == Color::White {
            for r in (rank + 1)..8 {
                mask |= 1u64 << (r * 8 + f);
            }
        } else {
            for r in 0..rank {
                mask |= 1u64 << (r * 8 + f);
            }
        }
    }
    
    (enemy_pawns & mask) == 0
}

fn count_active_pieces(pos: &Position, color: Color) -> i32 {
    let mut active = 0;
    
    let knights = pos.pieces_colored(PieceType::Knight, color);
    let center_extended = 0x00003C3C3C3C0000u64;
    active += (knights & center_extended).count_ones() as i32;
    
    let bishops = pos.pieces_colored(PieceType::Bishop, color);
    let long_diagonals = 0x8142241818244281u64;
    active += (bishops & long_diagonals).count_ones() as i32;
    
    let rooks = pos.pieces_colored(PieceType::Rook, color);
    let seventh_rank = if color == Color::White { 0x00FF000000000000 } else { 0x000000000000FF00 };
    active += (rooks & seventh_rank).count_ones() as i32 * 2;
    
    active
}

fn is_king_exposed(pos: &Position, color: Color) -> bool {
    let king_sq = pos.king_square(color);
    let king_file = king_sq % 8;
    let king_rank = king_sq / 8;
    
    if king_file >= 2 && king_file <= 5 && king_rank >= 2 && king_rank <= 5 {
        return true;
    }
    
    let our_pawns = pos.pieces_colored(PieceType::Pawn, color);
    let shield_mask = get_king_shield_mask(king_sq, color);
    
    let shield_pawns = (our_pawns & shield_mask).count_ones();
    shield_pawns < 2
}

fn get_king_shield_mask(king_sq: u8, color: Color) -> u64 {
    let file = king_sq % 8;
    let rank = king_sq / 8;
    
    let mut mask = 0u64;
    
    let shield_rank = if color == Color::White {
        if rank < 7 { rank + 1 } else { rank }
    } else {
        if rank > 0 { rank - 1 } else { rank }
    };
    
    for f in file.saturating_sub(1)..=(file + 1).min(7) {
        mask |= 1u64 << (shield_rank * 8 + f);
        
        let second_rank = if color == Color::White {
            if shield_rank < 7 { shield_rank + 1 } else { shield_rank }
        } else {
            if shield_rank > 0 { shield_rank - 1 } else { shield_rank }
        };
        
        mask |= 1u64 << (second_rank * 8 + f);
    }
    
    mask
}

pub fn score_to_win_probability(score: i32) -> f32 {
    let k = 4.0;
    let normalized = score as f32 / 400.0;
    
    1.0 / (1.0 + (-k * normalized).exp())
}

pub fn is_fortress(pos: &Position) -> bool {
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
    
    blocked_files >= 5
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_eval_cache() {
        let mut cache = EvalCache::new(16);
        
        cache.store(12345, 100, 5);
        assert_eq!(cache.probe(12345), Some(100));
        assert_eq!(cache.probe(54321), None);
    }
    
    #[test]
    fn test_phase_helpers() {
        assert!(is_endgame(32));
        assert!(!is_endgame(200));
        assert!(is_middlegame(150));
        assert!(!is_middlegame(50));
        assert!(is_opening(220));
        assert!(!is_opening(100));
    }
    
    #[test]
    fn test_win_probability() {
        let prob_even = score_to_win_probability(0);
        assert!((prob_even - 0.5).abs() < 0.01);
        
        let prob_winning = score_to_win_probability(400);
        assert!(prob_winning > 0.8);
        
        let prob_losing = score_to_win_probability(-400);
        assert!(prob_losing < 0.2);
    }
    
    #[test]
    fn test_complexity() {
        let pos = Position::startpos();
        let complexity = calculate_complexity(&pos);
        
        assert!(complexity > 20);
        assert!(complexity < 100);
    }
    
    #[test]
    fn test_is_likely_draw() {
        let pos = Position::from_fen("8/8/8/8/8/8/8/4k1K1 w - - 0 1").unwrap();
        assert!(is_likely_draw(&pos));
        
        let pos = Position::startpos();
        assert!(!is_likely_draw(&pos));
    }
}
