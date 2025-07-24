
use crate::board::position::{Position, Move, Color, PieceType};
use crate::eval::material::calculate_phase;

pub const FUTILITY_MARGINS: [i32; 8] = [0, 100, 150, 200, 300, 400, 550, 700];
pub const REVERSE_FUTILITY_MARGINS: [i32; 8] = [0, 150, 250, 350, 500, 700, 900, 1200];
pub const LMP_COUNTS: [usize; 8] = [0, 4, 8, 12, 18, 26, 36, 48];

pub const SEE_QUIET_THRESHOLD: i32 = -50;
pub const SEE_CAPTURE_THRESHOLD: i32 = -100;

#[inline(always)]
pub fn futility_pruning(
    pos: &Position,
    depth: i32,
    alpha: i32,
    _beta: i32,
    static_eval: i32,
    improving: bool,
) -> Option<i32> {
    if depth >= 6 || depth < 0 || pos.is_in_check(pos.side_to_move) {
        return None;
    }
    
    let king_square = pos.king_square(pos.side_to_move);
    if pos.is_square_attacked(king_square, pos.side_to_move.opposite()) {
        return None;
    }
    
    let base_margin = FUTILITY_MARGINS[depth.min(7) as usize];
    let safety_margin = if improving { 0 } else { 50 };
    let total_margin = base_margin + safety_margin;
    
    if static_eval + total_margin <= alpha {
        Some(static_eval)
    } else {
        None
    }
}

#[inline(always)]
pub fn reverse_futility_pruning(
    pos: &Position,
    depth: i32,
    beta: i32,
    static_eval: i32,
    improving: bool,
) -> Option<i32> {
    if depth >= 7 || depth < 0 || pos.is_in_check(pos.side_to_move) {
        return None;
    }
    
    let king_square = pos.king_square(pos.side_to_move);
    if pos.is_square_attacked(king_square, pos.side_to_move.opposite()) {
        return None;
    }
    
    let base_margin = REVERSE_FUTILITY_MARGINS[depth.min(7) as usize];
    let safety_margin = if improving { 0 } else { 50 };
    let total_margin = base_margin + safety_margin;
    
    if static_eval - total_margin >= beta {
        Some(static_eval - total_margin / 2)
    } else {
        None
    }
}

#[inline(always)]
pub fn razoring(
    pos: &Position,
    depth: i32,
    alpha: i32,
    static_eval: i32,
) -> Option<i32> {
    if depth > 2 || pos.is_in_check(pos.side_to_move) {
        return None;
    }
    
    let razor_margin = 150 + 100 * depth;
    
    if static_eval + razor_margin <= alpha {
        Some(static_eval)
    } else {
        None
    }
}

#[inline(always)]
pub fn late_move_pruning(
    depth: i32,
    moves_searched: usize,
    is_pv: bool,
    improving: bool,
) -> bool {
    if is_pv || depth >= 8 || depth < 0 {
        return false;
    }
    
    let base_count = LMP_COUNTS[depth.min(7) as usize];
    let move_limit = if improving {
        base_count + 4
    } else {
        base_count
    };
    
    moves_searched >= move_limit
}

pub fn see_pruning(
    pos: &Position,
    mv: Move,
    threshold: i32,
) -> bool {
    use crate::search::alphabeta::see;
    
    let see_value = see(pos, mv);
    see_value < threshold
}

fn is_tactical_position(pos: &Position) -> bool {
    let king_square = pos.king_square(pos.side_to_move);
    if pos.is_square_attacked(king_square, pos.side_to_move.opposite()) {
        return true;
    }
    
    has_hanging_valuable_pieces(pos)
}

fn has_hanging_valuable_pieces(pos: &Position) -> bool {
    let colors = [Color::White, Color::Black];
    
    for &color in &colors {
        for piece_type in [PieceType::Rook, PieceType::Queen] {
            let mut pieces = pos.pieces_colored(piece_type, color);
            
            while pieces != 0 {
                let square = pieces.trailing_zeros() as u8;
                pieces &= pieces - 1;
                
                if pos.is_square_attacked(square, color.opposite()) {
                    let pawn_defenders = pos.pieces_colored(PieceType::Pawn, color);
                    let pawn_attacks = get_pawn_attacks_bitboard(pawn_defenders, color);
                    
                    if (pawn_attacks & (1u64 << square)) == 0 {
                        return true;
                    }
                }
            }
        }
    }
    
    false
}

fn get_pawn_attacks_bitboard(pawns: u64, color: Color) -> u64 {
    if color == Color::White {
        let left_attacks = (pawns & !0x0101010101010101) << 7;
        let right_attacks = (pawns & !0x8080808080808080) << 9;
        left_attacks | right_attacks
    } else {
        let left_attacks = (pawns & !0x8080808080808080) >> 7;
        let right_attacks = (pawns & !0x0101010101010101) >> 9;
        left_attacks | right_attacks
    }
}

fn has_material_advantage(pos: &Position) -> bool {
    let side = pos.side_to_move;
    let opponent = side.opposite();
    
    let my_queens = pos.pieces_colored(PieceType::Queen, side).count_ones() as i32;
    let opp_queens = pos.pieces_colored(PieceType::Queen, opponent).count_ones() as i32;
    
    let my_rooks = pos.pieces_colored(PieceType::Rook, side).count_ones() as i32;
    let opp_rooks = pos.pieces_colored(PieceType::Rook, opponent).count_ones() as i32;
    
    let my_minors = (pos.pieces_colored(PieceType::Knight, side) | 
                     pos.pieces_colored(PieceType::Bishop, side)).count_ones() as i32;
    let opp_minors = (pos.pieces_colored(PieceType::Knight, opponent) | 
                      pos.pieces_colored(PieceType::Bishop, opponent)).count_ones() as i32;
    
    let my_pawns = pos.pieces_colored(PieceType::Pawn, side).count_ones() as i32;
    let opp_pawns = pos.pieces_colored(PieceType::Pawn, opponent).count_ones() as i32;
    
    let material_diff = (my_queens - opp_queens) * 900 +
                       (my_rooks - opp_rooks) * 500 +
                       (my_minors - opp_minors) * 325 +
                       (my_pawns - opp_pawns) * 100;
    
    material_diff > 150
}

pub fn enhanced_null_move_conditions(
    pos: &Position,
    depth: i32,
    beta: i32,
    static_eval: i32,
    is_pv: bool,
) -> bool {
    if is_pv || depth < 3 {
        return false;
    }
    
    if pos.is_in_check(pos.side_to_move) {
        return false;
    }
    
    let side = pos.side_to_move;
    let has_pieces = pos.pieces_colored(PieceType::Knight, side) != 0 ||
                    pos.pieces_colored(PieceType::Bishop, side) != 0 ||
                    pos.pieces_colored(PieceType::Rook, side) != 0 ||
                    pos.pieces_colored(PieceType::Queen, side) != 0;
    
    if !has_pieces {
        return false;
    }
    
    static_eval >= beta
}

pub fn null_move_reduction(depth: i32, static_eval: i32, beta: i32) -> i32 {
    let base_reduction = 3;
    let eval_margin = (static_eval - beta) / 200;
    let depth_reduction = depth / 5;
    
    (base_reduction + eval_margin + depth_reduction).min(depth - 1)
}

pub fn history_pruning(
    depth: i32,
    history_score: i32,
    moves_searched: usize,
    is_capture: bool,
) -> bool {
    if is_capture || depth >= 6 || moves_searched < 6 {
        return false;
    }
    
    let threshold = match depth {
        1 => -4000,
        2 => -3000,
        3 => -2000,
        4 => -1000,
        5 => -500,
        _ => 0,
    };
    
    history_score < threshold
}


#[derive(Debug, Clone, Copy)]
pub enum PruningReason {
    Futility,
    ReverseFutility,
    Razoring,
    LateMove,
    SEE,
    History,
    MultiCut,
    ProbCut,
}

impl PruningReason {
    pub fn description(&self) -> &'static str {
        match self {
            PruningReason::Futility => "Futility",
            PruningReason::ReverseFutility => "Reverse Futility",
            PruningReason::Razoring => "Razoring",
            PruningReason::LateMove => "Late Move",
            PruningReason::SEE => "SEE",
            PruningReason::History => "History",
            PruningReason::MultiCut => "Multi-Cut",
            PruningReason::ProbCut => "ProbCut",
        }
    }
}

#[derive(Default)]
pub struct PruningStats {
    pub futility_prunes: std::sync::atomic::AtomicU64,
    pub lmp_prunes: std::sync::atomic::AtomicU64,
    pub see_prunes: std::sync::atomic::AtomicU64,
    pub history_prunes: std::sync::atomic::AtomicU64,
    pub razoring_prunes: std::sync::atomic::AtomicU64,
    pub multi_cut_prunes: std::sync::atomic::AtomicU64,
}

impl PruningStats {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn record_prune(&self, reason: PruningReason) {
        use std::sync::atomic::Ordering;
        match reason {
            PruningReason::Futility => { self.futility_prunes.fetch_add(1, Ordering::Relaxed); },
            PruningReason::LateMove => { self.lmp_prunes.fetch_add(1, Ordering::Relaxed); },
            PruningReason::SEE => { self.see_prunes.fetch_add(1, Ordering::Relaxed); },
            PruningReason::History => { self.history_prunes.fetch_add(1, Ordering::Relaxed); },
            PruningReason::Razoring => { self.razoring_prunes.fetch_add(1, Ordering::Relaxed); },
            PruningReason::MultiCut => { self.multi_cut_prunes.fetch_add(1, Ordering::Relaxed); },
            _ => {},
        }
    }
    
    pub fn clear(&self) {
        use std::sync::atomic::Ordering;
        self.futility_prunes.store(0, Ordering::Relaxed);
        self.lmp_prunes.store(0, Ordering::Relaxed);
        self.see_prunes.store(0, Ordering::Relaxed);
        self.history_prunes.store(0, Ordering::Relaxed);
        self.razoring_prunes.store(0, Ordering::Relaxed);
        self.multi_cut_prunes.store(0, Ordering::Relaxed);
    }
}

#[derive(Default)]
pub struct MultiCutTracker {
    cutoff_count: usize,
    depth_threshold: i32,
    move_threshold: usize,
}

impl MultiCutTracker {
    pub fn new() -> Self {
        Self {
            cutoff_count: 0,
            depth_threshold: 8,
            move_threshold: 6,
        }
    }
    
    pub fn record_cutoff(&mut self) {
        self.cutoff_count += 1;
    }
    
    pub fn should_multi_cut(&self, depth: i32, moves_searched: usize) -> bool {
        depth >= self.depth_threshold &&
        moves_searched >= self.move_threshold &&
        self.cutoff_count >= 3
    }
    
    pub fn reset(&mut self) {
        self.cutoff_count = 0;
    }
}