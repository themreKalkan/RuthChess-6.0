
use crate::{
    board::{
        position::{Position, Move, Color, PieceType, MoveType},
        zobrist::{ZOBRIST},
    },
    eval::{
        evaluate::{evaluate, evaluate_fast, Score, MATE_VALUE, DRAW_VALUE,evaluate_int},
        material::calculate_phase,
        eval_util,
    },
    movegen::{
        legal_moves::{generate_legal_moves, move_to_uci},
        magic::{all_attacks, get_bishop_attacks, get_rook_attacks, get_queen_attacks,
                get_knight_attacks, get_king_attacks, get_pawn_attacks,all_attacks_for_king},
    },
    search::{
        time_management::TimeManager,
        transposition::{TranspositionTable, TT_BOUND_EXACT, TT_BOUND_LOWER, TT_BOUND_UPPER, TTData},
        pruning::*,
    },
};

use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;
use std::cmp::{max, min};
use std::thread;

pub const MAX_PLY: i32 = 128;
const MAX_MOVES: usize = 256;
const INFINITY: i32 = 32000;
const MATE_SCORE: i32 = MATE_VALUE - MAX_PLY;
const DRAW_SCORE: i32 = 0;

const MAX_QPLY: i32 = 16;

const ASPIRATION_WINDOW_INIT: i32 = 25;
const ASPIRATION_WINDOW_MAX: i32 = 500;

const LMR_BASE: i32 = 1;
const LMR_DEPTH_FACTOR: f64 = 0.75;
const LMR_MOVE_FACTOR: f64 = 0.5;

const HISTORY_MAX: i32 = 16384;
const HISTORY_BONUS_MAX: i32 = 2048;
const HISTORY_MALUS_MAX: i32 = -2048;

const NULL_MOVE_REDUCTION: i32 = 3;
const NULL_MOVE_DEPTH_REDUCTION: i32 = 4;

const HASH_MOVE_SCORE: i32 = 1000000;
const WINNING_CAPTURE_SCORE: i32 = 900000;
const EQUAL_CAPTURE_SCORE: i32 = 800000;
const KILLER_MOVE_SCORE: i32 = 700000;
const COUNTER_MOVE_SCORE: i32 = 600000;
const LOSING_CAPTURE_SCORE: i32 = -100000;

const PIECE_VALUES: [i32; 7] = [0, 100, 320, 330, 500, 900, 10000];

#[derive(Default)]
pub struct SearchStats {
    pub nodes: AtomicU64,
    pub qnodes: AtomicU64,
    pub tt_hits: AtomicU64,
    pub tt_cuts: AtomicU64,
    pub null_cuts: AtomicU64,
    pub lmr_reductions: AtomicU64,
    pub pruned_moves: AtomicU64,
}

impl SearchStats {
    fn new() -> Self {
        Self::default()
    }
    
    fn clear(&self) {
        self.nodes.store(0, Ordering::Relaxed);
        self.qnodes.store(0, Ordering::Relaxed);
        self.tt_hits.store(0, Ordering::Relaxed);
        self.tt_cuts.store(0, Ordering::Relaxed);
        self.null_cuts.store(0, Ordering::Relaxed);
        self.lmr_reductions.store(0, Ordering::Relaxed);
        self.pruned_moves.store(0, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub best_move: Move,
    pub score: i32,
    pub depth: i32,
    pub nodes: u64,
    pub time_ms: u64,
    pub pv: Vec<Move>,
    pub hashfull: u32,
}

struct HistoryTables {
    quiet_history: [[[i32; 64]; 64]; 2],
    capture_history: [[[[i32; 7]; 64]; 7]; 2],
    counter_moves: [[[Move; 64]; 7]; 2],
    killers: [[Move; 2]; MAX_PLY as usize],
}

impl HistoryTables {
    fn new() -> Self {
        Self {
            quiet_history: [[[0; 64]; 64]; 2],
            capture_history: [[[[0; 7]; 64]; 7]; 2],
            counter_moves: [[[Move::null(); 64]; 7]; 2],
            killers: [[Move::null(); 2]; MAX_PLY as usize],
        }
    }
    
    fn update_quiet_history(&mut self, color: Color, from: u8, to: u8, bonus: i32) {
        let h = &mut self.quiet_history[color as usize][from as usize][to as usize];
        *h += bonus - (*h * bonus.abs() / HISTORY_MAX);
        *h = (*h).clamp(HISTORY_MALUS_MAX, HISTORY_BONUS_MAX);
    }
    
    fn update_capture_history(&mut self, color: Color, piece: PieceType, to: u8, captured: PieceType, bonus: i32) {
        let h = &mut self.capture_history[color as usize][piece as usize][to as usize][captured as usize];
        *h += bonus - (*h * bonus.abs() / HISTORY_MAX);
        *h = (*h).clamp(HISTORY_MALUS_MAX, HISTORY_BONUS_MAX);
    }
    
    fn update_killers(&mut self, ply: usize, mv: Move) {
        if ply < MAX_PLY as usize && self.killers[ply][0] != mv {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = mv;
        }
    }
    
    fn update_counter_move(&mut self, color: Color, piece: PieceType, to: u8, counter: Move) {
        self.counter_moves[color as usize][piece as usize][to as usize] = counter;
    }
    
    fn get_quiet_history(&self, color: Color, from: u8, to: u8) -> i32 {
        self.quiet_history[color as usize][from as usize][to as usize]
    }
    
    fn get_capture_history(&self, color: Color, piece: PieceType, to: u8, captured: PieceType) -> i32 {
        self.capture_history[color as usize][piece as usize][to as usize][captured as usize]
    }
}

fn lmr_reduction(depth: i32, moves_searched: usize, improving: bool, is_pv: bool) -> i32 {
    if depth < 3 || moves_searched < 2 {
        return 0;
    }
    
    let mut reduction = LMR_BASE + 
        ((depth as f64).ln() * LMR_DEPTH_FACTOR +
         (moves_searched as f64).ln() * LMR_MOVE_FACTOR) as i32;
    
    if improving {
        reduction -= 1;
    }
    if is_pv {
        reduction -= 1;
    }
    
    reduction.max(0).min(depth - 2)
}

pub fn see(pos: &Position, mv: Move) -> i32 {
    let from = mv.from();
    let to = mv.to();
    let (moving_piece, moving_color) = pos.piece_at(from);
    let (captured_piece, _) = pos.piece_at(to);
    
    let captured_value = if mv.move_type() == MoveType::EnPassant {
        PIECE_VALUES[PieceType::Pawn as usize]
    } else {
        PIECE_VALUES[captured_piece as usize]
    };
    
    let promotion_bonus = if mv.is_promotion() {
        PIECE_VALUES[mv.promotion() as usize] - PIECE_VALUES[PieceType::Pawn as usize]
    } else {
        0
    };
    
    if captured_value > PIECE_VALUES[moving_piece as usize] + 200 {
        return captured_value + promotion_bonus;
    }
    
    let mut gain_stack = [0i32; 32];
    let mut depth = 0;
    let mut occupied = pos.all_pieces();
    let mut color = moving_color;
    
    gain_stack[depth] = captured_value + promotion_bonus;
    occupied ^= 1u64 << from;
    
    loop {
        depth += 1;
        
        color = color.opposite();
        
        let attackers = get_attackers_to_square(pos, to, occupied) & pos.pieces(color);
        
        if attackers == 0 {
            break;
        }
        
        let (attacker_square, attacker_value) = find_least_attacker(pos, attackers, color);
        
        if attacker_square == 64 {
            break;
        }
        
        gain_stack[depth] = attacker_value - gain_stack[depth - 1];
        
        if attacker_value >= PIECE_VALUES[PieceType::King as usize] {
            break;
        }
        
        occupied ^= 1u64 << attacker_square;
        
        occupied = update_xray_attackers(pos, to, occupied, attacker_square);

        if(depth == 31){
            break;
        }
    }
    
    while depth > 0 {
        depth -= 1;
        gain_stack[depth] = -(-gain_stack[depth]).max(gain_stack[depth + 1]);
    }
    
    gain_stack[0]
}

fn get_attackers_to_square(pos: &Position, square: u8, occupied: u64) -> u64 {
    let mut attackers = 0u64;
    
    let white_pawns = pos.pieces_colored(PieceType::Pawn, Color::White) & occupied;
    let black_pawns = pos.pieces_colored(PieceType::Pawn, Color::Black) & occupied;
    attackers |= get_pawn_attacks(square, Color::Black) & white_pawns;
    attackers |= get_pawn_attacks(square, Color::White) & black_pawns;
    
    let knights = pos.pieces_of_type(PieceType::Knight) & occupied;
    attackers |= get_knight_attacks(square) & knights;
    
    let bishops_queens = (pos.pieces_of_type(PieceType::Bishop) | pos.pieces_of_type(PieceType::Queen)) & occupied;
    attackers |= get_bishop_attacks(square, occupied) & bishops_queens;
    
    let rooks_queens = (pos.pieces_of_type(PieceType::Rook) | pos.pieces_of_type(PieceType::Queen)) & occupied;
    attackers |= get_rook_attacks(square, occupied) & rooks_queens;
    
    let kings = pos.pieces_of_type(PieceType::King) & occupied;
    attackers |= get_king_attacks(square) & kings;
    
    attackers
}

fn find_least_attacker(pos: &Position, attackers: u64, color: Color) -> (u8, i32) {
    for piece_type in [PieceType::Pawn, PieceType::Knight, PieceType::Bishop, 
                       PieceType::Rook, PieceType::Queen, PieceType::King] {
        let piece_attackers = attackers & pos.pieces_colored(piece_type, color);
        if piece_attackers != 0 {
            let square = piece_attackers.trailing_zeros() as u8;
            return (square, PIECE_VALUES[piece_type as usize]);
        }
    }
    (64, 0)
}

fn update_xray_attackers(pos: &Position, target: u8, mut occupied: u64, removed_square: u8) -> u64 {
    let (removed_piece, _) = pos.piece_at(removed_square);
    
    // X-ray saldırganları bul
    match removed_piece {
        PieceType::Bishop | PieceType::Queen => {
            // Removed square'den target'a doğru yönü bul
            let direction = get_line_direction(removed_square, target);
            if direction.0 != 0 || direction.1 != 0 { // Diagonal
                let behind_attackers = get_xray_attackers_on_line(pos, removed_square, target, occupied, true);
                occupied |= behind_attackers;
            }
        }
        PieceType::Rook => {
            let direction = get_line_direction(removed_square, target);
            if (direction.0 == 0) != (direction.1 == 0) { // Straight line
                let behind_attackers = get_xray_attackers_on_line(pos, removed_square, target, occupied, false);
                occupied |= behind_attackers;
            }
        }
        PieceType::Queen if removed_piece == PieceType::Queen => {
            // Queen için hem diagonal hem straight kontrol et
            let behind_diagonal = get_bishop_attacks(target, occupied) & 
                (pos.pieces_of_type(PieceType::Bishop) | pos.pieces_of_type(PieceType::Queen));
            let behind_straight = get_rook_attacks(target, occupied) & 
                (pos.pieces_of_type(PieceType::Rook) | pos.pieces_of_type(PieceType::Queen));
            occupied |= behind_diagonal | behind_straight;
        }
        _ => {}
    }
    
    occupied
}

// Yardımcı fonksiyon ekle
fn get_xray_attackers_on_line(pos: &Position, from: u8, to: u8, occupied: u64, is_diagonal: bool) -> u64 {
    let mut result = 0u64;
    let direction = get_line_direction(from, to);
    
    let mut current = from as i8;
    loop {
        current -= direction.0 * 8 + direction.1;
        if current < 0 || current >= 64 {
            break;
        }
        
        let sq = current as u8;
        if occupied & (1u64 << sq) != 0 {
            let (piece, _) = pos.piece_at(sq);
            if is_diagonal && (piece == PieceType::Bishop || piece == PieceType::Queen) {
                result |= 1u64 << sq;
            } else if !is_diagonal && (piece == PieceType::Rook || piece == PieceType::Queen) {
                result |= 1u64 << sq;
            }
            break;
        }
    }
    
    result
}

fn get_line_direction(from: u8, to: u8) -> (i8, i8) {
    let from_rank = (from / 8) as i8;
    let from_file = (from % 8) as i8;
    let to_rank = (to / 8) as i8;
    let to_file = (to % 8) as i8;
    
    let rank_diff = to_rank - from_rank;
    let file_diff = to_file - from_file;
    
    (
        if rank_diff == 0 { 0 } else { rank_diff / rank_diff.abs() },
        if file_diff == 0 { 0 } else { file_diff / file_diff.abs() }
    )
}

struct MoveOrder {
    moves: Vec<Move>,
    scores: Vec<i32>,
    hash_move: Move,
    stage: usize,
}

impl MoveOrder {
    fn new(pos: &Position, tt_move: Move, ply: usize, history: &HistoryTables, prev_move: Move) -> Self {
        let moves = generate_legal_moves(pos);
        let mut scores = vec![0; moves.len()];
        
        let valid_tt_move = if tt_move != Move::null() && moves.contains(&tt_move) {
            tt_move
        } else {
            Move::null()
        };
        
        for (i, &mv) in moves.iter().enumerate() {
            scores[i] = Self::score_move(pos, mv, valid_tt_move, ply, history, prev_move);
        }
        
        Self {
            moves,
            scores,
            hash_move: valid_tt_move,
            stage: 0,
        }
    }
    
    fn score_move(
        pos: &Position,
        mv: Move,
        tt_move: Move,
        ply: usize,
        history: &HistoryTables,
        prev_move: Move,
    ) -> i32 {
        if mv == tt_move {
            return HASH_MOVE_SCORE;
        }
        
        let from = mv.from();
        let to = mv.to();
        let (piece, color) = pos.piece_at(from);
        let (captured, _) = pos.piece_at(to);
        
        if captured != PieceType::None || mv.move_type() == MoveType::EnPassant {
            let see_value = see(pos, mv);
            
            let cap_history = history.get_capture_history(color, piece, to, captured);
            
            if see_value > 0 {
                return WINNING_CAPTURE_SCORE + see_value + cap_history / 16;
            } else if see_value == 0 {
                return EQUAL_CAPTURE_SCORE + cap_history / 16;
            } else {
                return LOSING_CAPTURE_SCORE + see_value + cap_history / 16;
            }
        }
        
        if mv.is_promotion() {
            let promo_bonus = match mv.promotion() {
                PieceType::Queen => 800,
                PieceType::Rook => 500,
                PieceType::Bishop => 300,
                PieceType::Knight => 300,
                _ => 0,
            };
            return KILLER_MOVE_SCORE + promo_bonus;
        }
        
        
        if ply < (MAX_PLY - 1) as usize {
            if mv == history.killers[ply][0] {
                return KILLER_MOVE_SCORE;
            }
            if mv == history.killers[ply][1] {
                return KILLER_MOVE_SCORE - 10000;
            }
        }
        
        if prev_move != Move::null() {
            let (prev_piece, prev_color) = pos.piece_at(prev_move.to());
            let counter = history.counter_moves[prev_color as usize][prev_piece as usize][prev_move.to() as usize];
            if mv == counter {
                return COUNTER_MOVE_SCORE;
            }
        }
        
        history.get_quiet_history(color, from, to)
    }
    
    fn next(&mut self) -> Option<Move> {
        if self.stage >= self.moves.len() {
            return None;
        }
        
        let mut best_idx = self.stage;
        let mut best_score = self.scores[self.stage];
        
        for i in (self.stage + 1)..self.moves.len() {
            if self.scores[i] > best_score {
                best_score = self.scores[i];
                best_idx = i;
            }
        }
        
        if best_idx != self.stage {
            self.moves.swap(self.stage, best_idx);
            self.scores.swap(self.stage, best_idx);
        }
        
        let mv = self.moves[self.stage];
        self.stage += 1;
        Some(mv)
    }
}

pub struct SearchContext {
    tt: Arc<TranspositionTable>,
    history: Arc<Mutex<HistoryTables>>,
    stats: Arc<SearchStats>,
    stop_flag: Arc<AtomicBool>,
    root_position: Position,
    start_time: Instant,
    time_manager: TimeManager,
    thread_id: usize,
    pruning_stats: PruningStats,
}

impl SearchContext {
    fn new(
        tt: Arc<TranspositionTable>,
        stop_flag: Arc<AtomicBool>,
        root_position: Position,
        time_manager: TimeManager,
        thread_id: usize,
    ) -> Self {
        Self {
            tt,
            history: Arc::new(Mutex::new(HistoryTables::new())),
            stats: Arc::new(SearchStats::new()),
            stop_flag,
            root_position,
            start_time: Instant::now(),
            time_manager,
            thread_id,
            pruning_stats: PruningStats::new(),
        }
    }
    
    fn should_stop(&self) -> bool {
        if self.stop_flag.load(Ordering::Relaxed) {
            return true;
        }
        
        let nodes = self.stats.nodes.load(Ordering::Relaxed);
        if nodes & 4095 == 0 {
            self.time_manager.should_stop(self.start_time)
        } else {
            false
        }
    }

    fn is_repetition_draw(&self, pos: &Position) -> bool {
        false
    }
    
    fn alpha_beta(
        &self,
        pos: &mut Position,
        mut alpha: i32,
        mut beta: i32,
        mut depth: i32,
        ply: i32,
        is_pv: bool,
        skip_null: bool,
    ) -> i32 {
        if ply >= MAX_PLY - 1 {
            return evaluate_fast(pos);
        }
        
        let alpha_orig = alpha;
        self.stats.nodes.fetch_add(1, Ordering::Relaxed);

        if ply > 0 && (ply & 7) == 0 && self.should_stop() {
            return evaluate_fast(pos);
        }

        if !is_pv && ply > 0 && self.is_repetition_draw(pos) {
            return DRAW_SCORE;
        }

        let mate_alpha = -MATE_VALUE + ply;
        let mate_beta = MATE_VALUE - ply - 1;
        if mate_alpha >= beta {
            return mate_alpha;
        }
        if mate_beta <= alpha {
            return mate_beta;
        }
        alpha = alpha.max(mate_alpha);
        beta = beta.min(mate_beta);

        let root_node = ply == 0;
        let in_check = pos.is_in_check(pos.side_to_move);

        if in_check && ply < 2 * depth {
            depth += 1;
        }

        let tt_entry = self.tt.probe(pos.hash);
        let mut tt_move = Move::null();
        
        if let Some(ref entry) = tt_entry {
            tt_move = entry.best_move;
            self.stats.tt_hits.fetch_add(1, Ordering::Relaxed);
            
            if entry.depth >= depth as u8 && !is_pv && !root_node {
                match entry.bound {
                    TT_BOUND_EXACT => {
                        self.stats.tt_cuts.fetch_add(1, Ordering::Relaxed);
                        return entry.score;
                    }
                    TT_BOUND_LOWER => {
                        if entry.score >= beta {
                            self.stats.tt_cuts.fetch_add(1, Ordering::Relaxed);
                            return entry.score;
                        }
                    }
                    TT_BOUND_UPPER => {
                        if entry.score <= alpha {
                            self.stats.tt_cuts.fetch_add(1, Ordering::Relaxed);
                            return entry.score;
                        }
                    }
                    _ => {}
                }
            }
        }
        
        if depth <= 0 {
            return self.quiescence_search(pos, alpha, beta, ply, ply);
        }
        
        let static_eval = if in_check {
            if let Some(ref entry) = tt_entry {
                entry.static_eval
            } else {
                0
            }
        } else if let Some(ref entry) = tt_entry {
            if entry.static_eval.abs() < MATE_SCORE {
                entry.static_eval
            } else {
                evaluate(pos).score
            }
        } else {
            evaluate(pos).score
        };
        
        let improving = !in_check && ply >= 2 && 
            static_eval > self.get_static_eval_two_plies_ago(ply);
        
        if !is_pv && !in_check && !root_node {
            let is_tactical = is_tactical_position(pos);
            
            if depth <= 5 && !is_tactical {
                if let Some(score) = reverse_futility_pruning(pos, depth, beta, static_eval, improving) {
                    self.pruning_stats.record_prune(PruningReason::ReverseFutility);
                    return score;
                }
            }
            
            if !skip_null && depth >= 3 && static_eval >= beta && !is_tactical {
                let r = 3 + depth / 4 + (static_eval - beta).min(200) / 200;
                
                pos.make_null_move();
                let score = -self.alpha_beta(pos, -beta, -beta + 1, depth - r, ply + 1, false, true);
                pos.unmake_null_move();
                
                if score >= beta && score < MATE_SCORE {
                    self.stats.null_cuts.fetch_add(1, Ordering::Relaxed);
                    return score;
                }
            }
            
            if depth <= 2 && !is_tactical {
                if let Some(score) = razoring(pos, depth, alpha, static_eval) {
                    self.pruning_stats.record_prune(PruningReason::Razoring);
                    return score;
                }
            }
        }

        if tt_move == Move::null() && depth >= 6 {
            let iid_depth = if is_pv { depth - 2 } else { depth / 2 };
            self.alpha_beta(pos, alpha, beta, iid_depth, ply, is_pv, skip_null);
            if let Some(entry) = self.tt.probe(pos.hash) {
                tt_move = entry.best_move;
            }
        }
        
        let history = self.history.lock().unwrap();
        let prev_move = if ply > 0 { self.get_previous_move(ply) } else { Move::null() };
        let mut move_order = MoveOrder::new(pos, tt_move, ply as usize, &history, prev_move);
        drop(history);
        
        let mut best_move: Move = Move::null();
        let mut best_score: i32 = -INFINITY;
        let mut moves_searched: usize = 0;
        let mut quiet_moves: Vec<Move> = Vec::with_capacity(64);
        
        while let Some(mv) = move_order.next() {
            let is_capture = pos.piece_at(mv.to()).0 != PieceType::None || mv.move_type() == MoveType::EnPassant;
            let gives_check = pos.gives_check(mv);
            
            if !is_pv && !root_node && best_score > -MATE_SCORE && depth <= 8 {
                let is_tactical_move = gives_check || mv.is_promotion() || 
                    (is_capture && see(pos, mv) >= 0);
                
                if !is_tactical_move {
                    if !is_capture && late_move_pruning(depth, moves_searched, is_pv, improving) {
                        self.pruning_stats.record_prune(PruningReason::LateMove);
                        self.stats.pruned_moves.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    
                    if is_capture && moves_searched > 0 && see_pruning(pos, mv, -50 * depth) {
                        self.pruning_stats.record_prune(PruningReason::SEE);
                        self.stats.pruned_moves.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                }
            }
            
            if !is_capture && !gives_check {
                quiet_moves.push(mv);
            }
            
            if !pos.make_move(mv) {
                continue;
            }
            
            let mut score;
            let mut extension = 0;
            
            if gives_check && see(pos, mv) >= 0 && ply < 2 * depth {
                extension = 1;
            }
            
            if moves_searched == 0 {
                score = -self.alpha_beta(pos, -beta, -alpha, depth - 1 + extension, ply + 1, is_pv, false);
            } else {
                let mut reduction = 0;
                if depth >= 3 && moves_searched >= 2 && !is_capture && !gives_check && 
                   !in_check && !mv.is_promotion() {
                    
                    reduction = lmr_reduction(depth, moves_searched, improving, is_pv);
                    
                    if !is_capture {
                        let history = self.history.lock().unwrap();
                        let history_score = history.get_quiet_history(pos.side_to_move, mv.from(), mv.to());
                        drop(history);
                        reduction -= (history_score / 8192).clamp(-2, 2);
                    }
                    
                    reduction = reduction.max(0);
                    
                    if reduction > 0 {
                        self.stats.lmr_reductions.fetch_add(1, Ordering::Relaxed);
                    }
                }
                
                score = -self.alpha_beta(pos, -alpha - 1, -alpha, 
                                       depth - 1 - reduction + extension, ply + 1, false, false);
                
                if score > alpha && (reduction > 0 || score < beta) {
                    score = -self.alpha_beta(pos, -beta, -alpha, 
                                           depth - 1 + extension, ply + 1, is_pv, false);
                }
            }
            
            pos.unmake_move(mv);
            
            moves_searched += 1;
            
            if score > best_score {
                best_score = score;
                best_move = mv;
                
                if score > alpha {
                    alpha = score;
                    
                    if score >= beta {
                        if !is_capture && !gives_check {
                            let bonus = (depth * depth).min(400);
                            let mut history = self.history.lock().unwrap();
                            
                            if ply < (MAX_PLY - 1) as i32 {
                                history.update_killers(ply as usize, mv);
                            }
                            
                            let (piece, color) = pos.piece_at(mv.from());
                            history.update_quiet_history(color, mv.from(), mv.to(), bonus);
                            
                            if prev_move != Move::null() {
                                let (prev_piece, prev_color) = pos.piece_at(prev_move.to());
                                history.update_counter_move(prev_color, prev_piece, prev_move.to(), mv);
                            }
                            
                            for &quiet_mv in &quiet_moves {
                                if quiet_mv != mv {
                                    history.update_quiet_history(color, quiet_mv.from(), quiet_mv.to(), -bonus/2);
                                }
                            }
                            
                            drop(history);
                        }
                        
                        break;
                    }
                }
            }
        }
        
        if moves_searched == 0 {
            if in_check {
                return -MATE_VALUE + ply;
            } else {
                return DRAW_SCORE;
            }
        }
        
        let bound = if best_score >= beta {
            TT_BOUND_LOWER
        } else if best_score <= alpha_orig {
            TT_BOUND_UPPER
        } else {
            TT_BOUND_EXACT
        };
        
        self.tt.store(
            pos.hash,
            best_move,
            best_score,
            static_eval,
            depth as u8,
            bound,
            ply as u8,
        );
        
        best_score
    }

    fn quiescence_search(
        &self,
        pos: &mut Position,
        mut alpha: i32,
        beta: i32,
        ply: i32,
        q_start_ply: i32,
    ) -> i32 {
        if ply >= MAX_PLY - 1 {
            return evaluate_fast(pos);
        }
        
        let q_depth = ply - q_start_ply;
        if q_depth >= MAX_QPLY {
            return evaluate_fast(pos);
        }
        
        self.stats.qnodes.fetch_add(1, Ordering::Relaxed);

        if (self.stats.qnodes.load(Ordering::Relaxed) & 1023) == 0 && self.should_stop() {
            return evaluate_fast(pos);
        }

        if pos.is_draw() {
            return DRAW_SCORE;
        }

        let in_check = pos.is_in_check(pos.side_to_move);
        
        let mut stand_pat = -INFINITY;
        if !in_check {
            stand_pat = evaluate_fast(pos);
            
            if stand_pat >= beta {
                return beta;
            }
            
            if stand_pat < alpha - 900 {
                return alpha;
            }
            
            if stand_pat > alpha {
                alpha = stand_pat;
            }
        }

        let all_moves = generate_legal_moves(pos);
        
        if in_check {
            if all_moves.is_empty() {
                return -MATE_VALUE + ply;
            }
            
            let mut move_count = 0;
            let max_moves_in_check = 32;
            
            let mut scored_moves: Vec<(Move, i32)> = all_moves.into_iter()
                .map(|mv| {
                    let mut score = 0;
                    if pos.piece_at(mv.to()).0 != PieceType::None {
                        score = see(pos, mv) + 10000;
                    }
                    if pos.gives_check(mv) {
                        score += 5000;
                    }
                    (mv, score)
                })
                .collect();
            
            scored_moves.sort_unstable_by_key(|&(_, score)| -score);
            
            for (mv, _) in scored_moves.into_iter().take(max_moves_in_check) {
                pos.make_move(mv);
                let score = -self.quiescence_search(pos, -beta, -alpha, ply + 1, q_start_ply);
                pos.unmake_move(mv);
                
                move_count += 1;
                
                if score >= beta {
                    return beta;
                }
                if score > alpha {
                    alpha = score;
                }
            }
            
            return alpha;
        }

        let mut captures: Vec<(Move, i32)> = Vec::new();
        
        for mv in all_moves {
            let is_capture = pos.piece_at(mv.to()).0 != PieceType::None || mv.move_type() == MoveType::EnPassant;
            let is_promotion = mv.is_promotion();
            
            if is_capture || is_promotion {
                let see_value = see(pos, mv);
                
                if !is_promotion && see_value < -50 {
                    continue;
                }
                
                if stand_pat + see_value + 200 < alpha {
                    continue;
                }
                
                captures.push((mv, see_value + if is_promotion { 900 } else { 0 }));
            }
        }
        
        if captures.len() > 32 {
            captures.sort_unstable_by_key(|&(_, see_val)| -see_val);
            captures.truncate(32);
        } else {
            captures.sort_unstable_by_key(|&(_, see_val)| -see_val);
        }
        
        for (mv, see_val) in captures {
            if see_val < 0 && stand_pat + see_val + 100 < alpha {
                continue;
            }
            
            pos.make_move(mv);
            let score = -self.quiescence_search(pos, -beta, -alpha, ply + 1, q_start_ply);
            pos.unmake_move(mv);
            
            if score >= beta {
                return beta;
            }
            if score > alpha {
                alpha = score;
            }
        }
        
        alpha
    }
    
    fn get_static_eval_two_plies_ago(&self, _ply: i32) -> i32 {
        0
    }
    
    fn get_previous_move(&self, _ply: i32) -> Move {
        Move::null()
    }
}

fn is_tactical_position(pos: &Position) -> bool {
    let king_square = pos.king_square(pos.side_to_move);
    if pos.is_square_attacked(king_square, pos.side_to_move.opposite()) {
        return true;
    }
    
    if has_hanging_pieces(pos) {
        return true;
    }
    
    let side = pos.side_to_move;
    let pawns = pos.pieces_colored(PieceType::Pawn, side);
    let seventh_rank = if side == Color::White { 0xFF00000000000000u64 } else { 0x00000000000000FFu64 };
    if (pawns & seventh_rank) != 0 {
        return true;
    }
    
    false
}

fn has_hanging_pieces(pos: &Position) -> bool {
    for &color in &[Color::White, Color::Black] {
        for piece_type in [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
            let mut pieces = pos.pieces_colored(piece_type, color);
            
            while pieces != 0 {
                let square = pieces.trailing_zeros() as u8;
                pieces &= pieces - 1;
                
                if is_piece_hanging(pos, square, PIECE_VALUES[piece_type as usize]) {
                    return true;
                }
            }
        }
    }
    
    false
}

fn is_piece_hanging(pos: &Position, square: u8, piece_value: i32) -> bool {
    let (piece, color) = pos.piece_at(square);
    if piece == PieceType::None {
        return false;
    }
    
    let occupied = pos.all_pieces();
    let attackers = get_attackers_to_square(pos, square, occupied);
    
    let enemy_attackers = attackers & pos.pieces(color.opposite());
    let friendly_defenders = attackers & pos.pieces(color);
    
    if enemy_attackers == 0 {
        return false;
    }
    
    if friendly_defenders == 0 {
        return true;
    }
    
    let (attacker_sq, _) = find_least_attacker(pos, enemy_attackers, color.opposite());
    if attacker_sq == 64 {
        return false;
    }
    
    let capture_move = Move::new(attacker_sq, square, MoveType::Normal, PieceType::None);
    let see_value = see(pos, capture_move);
    
    see_value > 0
}

struct SearchWorker {
    id: usize,
    position: Position,
    tt: Arc<TranspositionTable>,
    history: Arc<Mutex<HistoryTables>>,
    stop_flag: Arc<AtomicBool>,
    depth_offset: i32,
    stats: Arc<SearchStats>,
}

impl SearchWorker {
    fn run(
        &self,
        target_depth: i32,
        time_manager: TimeManager,
        result_sender: std::sync::mpsc::Sender<(i32, i32, Vec<Move>)>,
    ) {
        let mut local_pos = self.position.clone();
        let mut best_score = -INFINITY;
        let mut best_pv = Vec::new();
        
        let start_depth = if self.id == 0 { 1 } else { 1 + (self.id as i32 % 4) };
        
        for depth in start_depth..=target_depth {
            if self.stop_flag.load(Ordering::Relaxed) {
                break;
            }
            
            let ctx = SearchContext {
                tt: Arc::clone(&self.tt),
                history: Arc::clone(&self.history),
                stats: Arc::clone(&self.stats),
                stop_flag: Arc::clone(&self.stop_flag),
                root_position: self.position.clone(),
                start_time: Instant::now(),
                time_manager: time_manager.clone(),
                thread_id: self.id,
                pruning_stats: PruningStats::new(),
            };
            
            let window_offset = if self.id == 0 { 0 } else { (self.id as i32 * 5) % 20 };
            let adjusted_depth = depth + self.depth_offset;
            
            let mut alpha = best_score - ASPIRATION_WINDOW_INIT - window_offset;
            let mut beta = best_score + ASPIRATION_WINDOW_INIT + window_offset;
            let mut delta = ASPIRATION_WINDOW_INIT;
            let mut fail_count = 0;
            
            if depth < 5 || best_score.abs() >= MATE_SCORE - 100 {
                alpha = -INFINITY;
                beta = INFINITY;
            }
            
            loop {
                let score = ctx.alpha_beta(&mut local_pos, alpha, beta, adjusted_depth, 0, true, false);
                
                if self.stop_flag.load(Ordering::Relaxed) {
                    break;
                }
                
                fail_count += 1;
                if fail_count > 10 {
                    alpha = -INFINITY;
                    beta = INFINITY;
                    let score = ctx.alpha_beta(&mut local_pos, alpha, beta, adjusted_depth, 0, true, false);
                    best_score = score;
                    best_pv = self.extract_pv_from_tt(&self.position, adjusted_depth);
                    let _ = result_sender.send((adjusted_depth, score, best_pv.clone()));
                    break;
                }
                
                if score <= alpha {
                    beta = (alpha + beta) / 2;
                    alpha = (score - delta).max(-INFINITY);
                    delta = delta + delta / 4 + 5;
                    
                    if fail_count > 3 {
                        alpha = -INFINITY;
                    }
                } else if score >= beta {
                    alpha = (alpha + beta) / 2;
                    beta = (score + delta).min(INFINITY);
                    delta = delta + delta / 4 + 5;
                    
                    if fail_count > 3 {
                        beta = INFINITY;
                    }
                } else {
                    best_score = score;
                    best_pv = self.extract_pv_from_tt(&self.position, adjusted_depth);
                    
                    let _ = result_sender.send((adjusted_depth, score, best_pv.clone()));
                    break;
                }
                
                if delta > ASPIRATION_WINDOW_MAX {
                    alpha = -INFINITY;
                    beta = INFINITY;
                }
            }
        }
    }
    
    fn extract_pv_from_tt(&self, pos: &Position, max_length: i32) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut pv_pos = pos.clone();
        let mut seen = std::collections::HashSet::new();
        
        for _ in 0..max_length.min(MAX_PLY) {
            if !seen.insert(pv_pos.hash) {
                break;
            }
            
            if let Some(entry) = self.tt.probe(pv_pos.hash) {
                let mv = entry.best_move;
                if mv == Move::null() {
                    break;
                }
                
                let legal_moves = generate_legal_moves(&pv_pos);
                if legal_moves.contains(&mv) {
                    pv.push(mv);
                    if !pv_pos.make_move(mv) {
                        break;
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        
        pv
    }
}

pub struct ParallelSearch {
    tt: Arc<TranspositionTable>,
    pub stop_flag: Arc<AtomicBool>,
    thread_count: usize,
    global_history: Arc<Mutex<HistoryTables>>,
    global_stats: Arc<SearchStats>,
}

impl ParallelSearch {
    pub fn new(thread_count: usize, tt_size_mb: usize) -> Self {
        Self {
            tt: Arc::new(TranspositionTable::new(tt_size_mb)),
            stop_flag: Arc::new(AtomicBool::new(false)),
            thread_count: thread_count.max(1),
            global_history: Arc::new(Mutex::new(HistoryTables::new())),
            global_stats: Arc::new(SearchStats::new()),
        }
    }
    
    pub fn search(&mut self, pos: &Position, max_depth: i32, time_manager: TimeManager) -> SearchResult {
        self.stop_flag.store(false, Ordering::SeqCst);
        self.tt.new_search();
        self.global_stats.clear();
        
        let start_time = Instant::now();
        let (result_sender, result_receiver) = std::sync::mpsc::channel();
        
        let mut worker_handles = Vec::new();
        
        for thread_id in 0..self.thread_count {
            let worker = SearchWorker {
                id: thread_id,
                position: pos.clone(),
                tt: Arc::clone(&self.tt),
                history: Arc::clone(&self.global_history),
                stop_flag: Arc::clone(&self.stop_flag),
                depth_offset: if thread_id < 4 { 0 } else { (thread_id as i32 / 4) },
                stats: Arc::clone(&self.global_stats),
            };
            
            let thread_result_sender = result_sender.clone();
            let thread_time_manager = time_manager.clone();
            let thread_max_depth = max_depth;
            
            let handle = thread::spawn(move || {
                worker.run(thread_max_depth, thread_time_manager, thread_result_sender);
            });
            
            worker_handles.push(handle);
        }
        
        drop(result_sender);
        
        let mut best_result = SearchResult {
            best_move: Move::null(),
            score: 0,
            depth: 0,
            nodes: 0,
            time_ms: 0,
            pv: Vec::new(),
            hashfull: 0,
        };
        
        let mut last_info_time = Instant::now();
        let info_interval = std::time::Duration::from_millis(500);
        
        loop {
            if time_manager.should_stop(start_time) && best_result.depth >= 4 {
                self.stop_flag.store(true, Ordering::SeqCst);
                break;
            }
            
            match result_receiver.recv_timeout(std::time::Duration::from_millis(10)) {
                Ok((depth, score, pv)) => {
                    if depth > best_result.depth || 
                       (depth == best_result.depth && score > best_result.score) {
                        
                        let elapsed = start_time.elapsed().as_millis() as u64;
                        let nodes = self.global_stats.nodes.load(Ordering::Relaxed);
                        let qnodes = self.global_stats.qnodes.load(Ordering::Relaxed);
                        let total_nodes = nodes + qnodes;
                        let nps = if elapsed > 0 { total_nodes * 1000 / elapsed } else { 0 };
                        
                        best_result = SearchResult {
                            best_move: pv.first().copied().unwrap_or(Move::null()),
                            score,
                            depth,
                            nodes: total_nodes,
                            time_ms: elapsed,
                            pv: pv.clone(),
                            hashfull: self.tt.hashfull(),
                        };
                        
                        self.send_uci_info(depth, score, total_nodes, elapsed, nps, &pv, 0);
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if last_info_time.elapsed() > info_interval && best_result.depth > 0 {
                        let elapsed = start_time.elapsed().as_millis() as u64;
                        let nodes = self.global_stats.nodes.load(Ordering::Relaxed);
                        let qnodes = self.global_stats.qnodes.load(Ordering::Relaxed);
                        let total_nodes = nodes + qnodes;
                        let nps = if elapsed > 0 { total_nodes * 1000 / elapsed } else { 0 };
                        
                        self.send_uci_info(
                            best_result.depth,
                            best_result.score,
                            total_nodes,
                            elapsed,
                            nps,
                            &best_result.pv,
                            1,
                        );
                        
                        last_info_time = Instant::now();
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    break;
                }
            }
            
            if self.stop_flag.load(Ordering::Relaxed) {
                break;
            }
        }
        
        self.stop_flag.store(true, Ordering::SeqCst);
        
        for handle in worker_handles {
            let _ = handle.join();
        }
        
        let elapsed = start_time.elapsed().as_millis() as u64;
        let nodes = self.global_stats.nodes.load(Ordering::Relaxed);
        let qnodes = self.global_stats.qnodes.load(Ordering::Relaxed);
        let total_nodes = nodes + qnodes;
        
        best_result.nodes = total_nodes;
        best_result.time_ms = elapsed;
        
        if cfg!(debug_assertions) {
            let nps = if elapsed > 0 { total_nodes * 1000 / elapsed } else { 0 };
            let tt_hits = self.global_stats.tt_hits.load(Ordering::Relaxed);
            let tt_cuts = self.global_stats.tt_cuts.load(Ordering::Relaxed);
            let null_cuts = self.global_stats.null_cuts.load(Ordering::Relaxed);
            let lmr_reductions = self.global_stats.lmr_reductions.load(Ordering::Relaxed);
            
            println!("info string nodes {} qnodes {} time {} nps {} tt_hits {} tt_cuts {} null_cuts {} lmr {}",
                     nodes, qnodes, elapsed, nps, tt_hits, tt_cuts, null_cuts, lmr_reductions);
        }
        
        best_result
    }
    
    fn send_uci_info(&self, depth: i32, score: i32, nodes: u64, time_ms: u64, nps: u64, pv: &[Move], currmovenumber: u32) {
        let score_str = if score.abs() >= MATE_SCORE {
        let mut mate_in = (MATE_VALUE - score.abs() + 1) / 2;
        // Adjust for ply count to show correct mate distance
        //mate_in = (mate_in + 1) / 2; // Convert from plies to moves
        format!("mate {}", if score > 0 { mate_in } else { -mate_in })
    } else {
        format!("cp {}", score)
    };
        
        let pv_str = pv.iter()
            .map(|&mv| move_to_uci(mv))
            .collect::<Vec<_>>()
            .join(" ");
        
        let hashfull = self.tt.hashfull();
        
        if currmovenumber > 0 {
            println!(
                "info depth {} seldepth {} score {} nodes {} nps {} time {} hashfull {} currmovenumber {} pv {}",
                depth, depth + 8, score_str, nodes, nps, time_ms, hashfull, currmovenumber, pv_str
            );
        } else {
            println!(
                "info depth {} seldepth {} score {} nodes {} nps {} time {} hashfull {} pv {}",
                depth, depth + 8, score_str, nodes, nps, time_ms, hashfull, pv_str
            );
        }
        
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }
    
    pub fn clear_hash(&mut self) {
        self.tt.clear();
        *self.global_history.lock().unwrap() = HistoryTables::new();
        self.global_stats.clear();
    }
}


pub fn see_ge_threshold(pos: &Position, mv: Move, threshold: i32) -> bool {
    see(pos, mv) >= threshold
}

pub fn alpha_beta_search(
    pos: &mut Position,
    depth: i32,
    time_manager: TimeManager,
    tt: Arc<TranspositionTable>,
    stop_flag: Arc<AtomicBool>,
) -> SearchResult {
    let mut search = ParallelSearch::new(1, 256);
    search.tt = tt;
    search.stop_flag = stop_flag;
    search.search(pos, depth, time_manager)
}
