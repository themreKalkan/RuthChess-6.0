use crate::{
    board::position::{Position, Move, Color, PieceType},
    eval::evaluate::{evaluate, EvalResult},
    movegen::legal_moves::{generate_legal_moves, is_checkmate},
    search::{
        transposition::{TranspositionTable, TTEntry, TTFlag},
        time_management::TimeManager,
    },
};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, Duration};
use parking_lot::RwLock;
use std::cell::RefCell;



// Constants
const INFINITY: i32 = 32000;
const MATE_VALUE: i32 = 31000;
const MAX_PLY: usize = 128;
const NULL_MV_RED: i32 = 3;  // Daha konservatif
const IID_REDUCTION: i32 = 2;
const ASP_WIN: i32 = 25;   // Daha geniş window
const FUTILITY_MARGIN: [i32; 4] = [0, 150, 250, 400];  
const RAZORM: i32 = 600;//Original 600
const LMR_BASE: f64 = 0.75;
const LMR_FACTOR: f64 = 2.25;
const HISTORY_MAX: i32 = 16384;
const MCUT_MOVES: usize = 2;
const MCUT_DEPTH: i32 = 4;//Original 8

#[derive(Default)]
pub struct SearchStats {
    pub nodes: AtomicU64,
    pub qnodes: AtomicU64,
    pub beta_cutoffs: AtomicU64,
    pub null_cutoffs: AtomicU64,
    pub tt_hits: AtomicU64,
}

impl SearchStats {
    pub fn clear(&self) {
        self.nodes.store(0, Ordering::Relaxed);
        self.qnodes.store(0, Ordering::Relaxed);
        self.beta_cutoffs.store(0, Ordering::Relaxed);
        self.null_cutoffs.store(0, Ordering::Relaxed);
        self.tt_hits.store(0, Ordering::Relaxed);
    }
}


#[derive(Default)]
pub struct SearchData {
    // Killer moves [ply][slot]
    pub killers: Vec<[Move; 2]>,
    // History heuristic [from][to]
    pub history: Vec<Vec<i32>>,
    // Countermove heuristic [piece][to]
    pub countermoves: Vec<Vec<Move>>,
    // Principal variation
    pub pv: Vec<Vec<Move>>,
    // Move ordering scores
    pub move_scores: Vec<i32>,
    // Search stack
    pub stack: Vec<SearchStackEntry>,
}

impl SearchData {
    pub fn new() -> Self {
        let mut history = vec![vec![0; 64]; 64];
        let mut countermoves = vec![vec![Move::null(); 64]; 7];
        let mut killers = vec![[Move::null(); 2]; MAX_PLY];
        let mut pv = vec![Vec::with_capacity(MAX_PLY); MAX_PLY];
        let stack = vec![SearchStackEntry::default(); MAX_PLY + 10];
        
        Self {
            killers,
            history,
            countermoves,
            pv,
            move_scores: Vec::with_capacity(256),
            stack,
        }
    }

    pub fn clear(&mut self) {
        self.killers.iter_mut().for_each(|k| {
            k[0] = Move::null();
            k[1] = Move::null();
        });
        
        self.history.iter_mut().for_each(|row| {
            row.iter_mut().for_each(|val| *val = 0);
        });
        
        self.countermoves.iter_mut().for_each(|row| {
            row.iter_mut().for_each(|m| *m = Move::null());
        });
    }
}

#[derive(Clone, Default)]
pub struct SearchStackEntry {
    pub static_eval: i32,
    pub current_move: Move,
    pub excluded_move: Move,
    pub killers: [Move; 2],
    pub double_extensions: i32,
}

// Main search
pub struct AlphaBetaSearch {
    pub tt: Arc<RwLock<TranspositionTable>>,
    pub time_manager: TimeManager,
    pub stats: Arc<SearchStats>,
    pub stop: Arc<AtomicBool>,
    pub start_time: Instant,
}

impl AlphaBetaSearch {
    pub fn new(tt_size_mb: usize) -> Self {
        Self {
            tt: Arc::new(RwLock::new(TranspositionTable::new(tt_size_mb))),
            time_manager: TimeManager::infinite(),
            stats: Arc::new(SearchStats::default()),
            stop: Arc::new(AtomicBool::new(false)),
            start_time: Instant::now(),
        }
    }

    /// Main iterative deepening search
    /// Main iterative deepening search - DETERMINISTIK VERSIYON
    pub fn search(
        &mut self,
        pos: &Position,
        max_depth: i32,
        time_manager: TimeManager,
    ) -> (Move, i32) {
        let start_time = Instant::now();
        self.start_time = start_time;
        let mut time_manager = time_manager;
        self.stop.store(false, Ordering::Relaxed);
        self.stats.clear();
        
        let mut search_data = SearchData::new();
        search_data.clear();
        
        {
            let mut tt = self.tt.write();
            tt.new_search();
        }

        let mut best_move = Move::null();
        let mut best_score = EvalResult { score: -INFINITY, is_mate: false, mate_distance: INFINITY };
        let mut prev_score = -INFINITY;

        // Root moves - sadece legal moves
        let root_moves = generate_legal_moves(pos);
        if root_moves.is_empty() {
            return (Move::null(), 0);
        }

        // Iterative deepening
        for depth in 1..=max_depth {
            if self.should_stop() {
                break;
            }

            let mut alpha = -INFINITY;
            let mut beta = INFINITY;
            
            // Aspiration window - sadece depth >= 5 için
            if depth >= 5 && best_score.score > -INFINITY + 1000 && best_score.score < INFINITY - 1000 {
                alpha = best_score.score - ASP_WIN;
                beta = best_score.score + ASP_WIN;
            }

            let mut score = best_score;
            let mut fail_high_count = 0;
            let mut fail_low_count = 0;

            // Aspiration window loop - maksimum 4 deneme
            for attempt in 0..4 {
                if self.should_stop() {
                    break;
                }

                score = self.alpha_beta(
                    pos,
                    alpha,
                    beta,
                    depth,
                    0,
                    true,
                    &mut search_data,
                );

                // Aspiration window başarısız oldu mu?
                if score.score <= alpha {
                    fail_low_count += 1;
                    alpha = alpha - ASP_WIN * (1 << fail_low_count);
                    if fail_low_count >= 2 {
                        alpha = -INFINITY;
                    }
                } else if score.score >= beta {
                    fail_high_count += 1;
                    beta = beta + ASP_WIN * (1 << fail_high_count);
                    if fail_high_count >= 2 {
                        beta = INFINITY;
                    }
                } else {
                    // Başarılı arama
                    break;
                }
            }

            if !self.should_stop() {
                best_score = score;
                if let Some(mv) = search_data.pv[0].get(0) {
                    best_move = *mv;
                }
                
                // Sadece valid score için print
                if score.score > -INFINITY && score.score < INFINITY {
                    self.print_info(depth, score, &search_data.pv[0], start_time);
                }
            }

            // Mate bulunduysa dur
            if score.is_mate {
                break;
            }

            // Zaman kontrolü
            if time_manager.should_stop(start_time) {
                break;
            }

            prev_score = score.score;
        }

        (best_move, best_score.score)
    }

    /// Main alpha-beta search 
    fn alpha_beta(
        &self,
        pos: &Position,
        mut alpha: i32,
        mut beta: i32,
        mut depth: i32,
        ply: usize,
        is_pv: bool,
        search_data: &mut SearchData,
    ) -> EvalResult 
    {
        // Ply limiti
        if ply >= MAX_PLY - 1 {
            return evaluate(pos);
        }

        search_data.pv[ply].clear();

        // Zaman kontrolü - daha sık kontrol et
        if self.stats.nodes.load(Ordering::Relaxed) & 511 == 0 {
            if self.should_stop() {
                return EvalResult{score:alpha, is_mate: false, mate_distance: 0};
            }
        }

        // Mate distance pruning
        let mate_score = MATE_VALUE - ply as i32;
        if alpha >= mate_score {
            return EvalResult{score:alpha, is_mate: true, mate_distance: depth - ply as i32};
        }
        if beta <= -mate_score {
            return EvalResult{score:beta, is_mate: true, mate_distance: depth - ply as i32};;
        }
        alpha = alpha.max(-mate_score);
        beta = beta.min(mate_score);

        // Quiescence search
        if depth <= 0 {
            return EvalResult {
                score: self.quiescence(pos, alpha, beta, ply, search_data),
                is_mate: false,
                mate_distance: 0,
            };
        }

        self.stats.nodes.fetch_add(1, Ordering::Relaxed);

        let in_check = pos.is_in_check(pos.side_to_move);
        let mut best_score = EvalResult {
            score: -INFINITY,
            is_mate: false,
            mate_distance: 0,
        };
        let mut best_move = Move::null();

        // Transposition table lookup
        let tt_entry = {
            let tt = self.tt.read();
            tt.probe(pos.hash).cloned()
        };
        
        let mut tt_move = Move::null();
        if let Some(entry) = &tt_entry {
            tt_move = entry.best_move;
            self.stats.tt_hits.fetch_add(1, Ordering::Relaxed);
            
            // TT cutoff - sadece non-PV node'larda
            if !is_pv && entry.depth >= depth as u8 {
                let tt_score = self.score_from_tt(entry.score, ply);
                
                // Mate score'ları dikkatli kullan
                if !tt_score.is_mate || 
                   (tt_score.is_mate && entry.depth >= depth as u8) {
                    match entry.flag {
                        TTFlag::Exact => return tt_score,
                        TTFlag::LowerBound => {
                            if tt_score.score >= beta {
                                return tt_score;
                            }
                        }
                        TTFlag::UpperBound => {
                            if tt_score.score <= alpha {
                                return tt_score;
                            }
                        }
                    }
                }
            }
        }

        // Static evaluation
        let static_eval = if in_check {
            -INFINITY
        } else {
            evaluate(pos).score
        };

        search_data.stack[ply].static_eval = static_eval;

        // Improving flag
        let improving = if ply >= 2 && !in_check {
            static_eval > search_data.stack[ply - 2].static_eval
        } else {
            false
        };

        // Check extension
        if in_check {
            depth += 1;
        }

        // Pruning techniques - sadece non-PV ve non-check
        if !is_pv && !in_check && depth >= 2 {
            // Reverse futility pruning (static eval pruning)
            if depth <= 6 && static_eval - 90 * depth >= beta {
                return EvalResult {
                    score: static_eval,
                    is_mate: false,
                    mate_distance: 0,
                };
            }

            // Null move pruning
            if depth >= 3 && 
               static_eval >= beta &&
               self.has_non_pawn_material(pos, pos.side_to_move) {
                
                // Null move'dan önce pozisyon kontrolü
                if ply > 0 && search_data.stack[ply - 1].current_move != Move::null() {
                    let mut new_pos = pos.clone();
                    new_pos.make_null_move();
                    
                    let reduction = NULL_MV_RED + depth / 4;
                    
                    let null_score = -(self.alpha_beta(
                        &new_pos,
                        -beta,
                        -beta + 1,
                        depth - reduction,
                        ply + 1,
                        false,
                        search_data,
                    ).score);
                    
                    new_pos.unmake_null_move();
                    
                    if null_score >= beta {
                        // Mate score'ları dikkatli döndür
                        if null_score >= MATE_VALUE - MAX_PLY as i32 {
                            return EvalResult{
                                score:beta,
                                is_mate:true,
                                mate_distance: depth - ply as i32,
                            };
                        }
                        return EvalResult {
                            score: null_score,
                            is_mate: false,
                            mate_distance: 0,
                        };
                    }
                }
            }

            // Razoring
            if depth <= 3 && static_eval + RAZORM < alpha {
                let razor_score = self.quiescence(pos, alpha, beta, ply, search_data);
                if razor_score <= alpha {
                    return EvalResult {
                        score: razor_score,
                        is_mate: false,
                        mate_distance: 0,
                    };
                }
            }
        }

        // Internal iterative deepening
        if tt_move == Move::null() && depth >= 2 && is_pv {
            self.alpha_beta(
                pos,
                alpha,
                beta,
                depth - IID_REDUCTION,
                ply,
                true,
                search_data,
            );
            
            // TT'den move al
            let tt_entry = {
                let tt = self.tt.read();
                tt.probe(pos.hash).cloned()
            };
            
            if let Some(entry) = tt_entry {
                tt_move = entry.best_move;
            }
        }

        // Move generation
        let mut moves = generate_legal_moves(pos);
        
        if moves.is_empty() {
            return if in_check {
                EvalResult {
                    score: -MATE_VALUE + ply as i32,
                    is_mate: true,
                    mate_distance: depth - ply as i32,
                }
            } else {
                EvalResult{
                    score: 0,
                    is_mate: false,
                    mate_distance: 0,
                } // Stalemate
            };
        }

        // Move ordering
        let move_scores = self.score_moves(&moves, tt_move, ply, pos, search_data);
        let mut move_indices: Vec<usize> = (0..moves.len()).collect();
        move_indices.sort_unstable_by_key(|&i| -move_scores[i]);

        let mut moves_searched = 0;
        let mut quiet_moves = Vec::new();

        // Move loop
        for &move_idx in &move_indices {
            let mv = moves[move_idx];
            
            // Excluded move için skip
            if mv == search_data.stack[ply].excluded_move {
                continue;
            }

            let mut new_pos = pos.clone();
            if !new_pos.make_move(mv) {
                continue;
            }

            moves_searched += 1;
            let gives_check = new_pos.is_in_check(new_pos.side_to_move);
            let is_capture = self.is_capture(pos, mv);
            let is_quiet = !is_capture && !mv.is_promotion();
            
            if is_quiet {
                quiet_moves.push(mv);
            }

            search_data.stack[ply].current_move = mv;

            // Extensions
            let mut extension = 0;
            
            // Singular extension
            if depth >= 8 && 
               mv == tt_move && 
               !in_check &&
               tt_entry.is_some() {
                let entry = tt_entry.as_ref().unwrap();
                if entry.flag == TTFlag::LowerBound && 
                   entry.depth >= (depth - 3) as u8 {
                    
                    let singular_beta = entry.score.score - 2 * depth;
                    let singular_score = self.singular_search(
                        pos,
                        singular_beta - 1,
                        singular_beta,
                        depth / 2,
                        ply,
                        mv,
                        search_data,
                    );
                    
                    if singular_score < singular_beta {
                        extension = 1;
                    }
                }
            }

            let new_depth = depth - 1 + extension;

            // Futility pruning
            if !is_pv && 
               !in_check && 
               !gives_check &&
               is_quiet &&
               new_depth <= 3 &&
               static_eval + FUTILITY_MARGIN[new_depth as usize] <= alpha &&
               best_score.score > -MATE_VALUE + 100 {
                new_pos.unmake_move(mv);
                continue;
            }

            // SEE pruning
            if !is_pv &&
               !in_check &&
               is_quiet &&
               depth <= 6 &&
               moves_searched > 1 &&
               !pos.see_ge(mv, -30 * depth * depth) {
                new_pos.unmake_move(mv);
                continue;
            }

            let mut score: EvalResult;

            // First move - full window
            if moves_searched == 1 {
                score = self.alpha_beta(
                    &new_pos,
                    -beta,
                    -alpha,
                    new_depth,
                    ply + 1,
                    is_pv,
                    search_data,
                );
                score.score = -score.score;
            } else {
                // LMR
                let mut reduction = 0;
                
                if depth >= 3 && 
                   moves_searched > 1 &&
                   is_quiet &&
                   !in_check &&
                   !gives_check {
                    
                    // Basit LMR formula
                    reduction = if moves_searched > 4 { 2 } else { 1 };
                    
                    // Adjustments
                    if !improving {
                        reduction += 1;
                    }
                    if is_pv {
                        reduction -= 1;
                    }
                    
                    // Killer moves
                    if mv == search_data.killers[ply][0] || 
                       mv == search_data.killers[ply][1] {
                        reduction -= 1;
                    }
                    
                    // History bonus
                    let history_score = search_data.history[mv.from() as usize][mv.to() as usize];
                    if history_score > 0 {
                        reduction -= 1;
                    }
                    
                    reduction = reduction.max(0).min(new_depth - 1);
                }

                // Null window search
                score = self.alpha_beta(
                    &new_pos,
                    -alpha - 1,
                    -alpha,
                    new_depth - reduction,
                    ply + 1,
                    false,
                    search_data,
                );
                score.score = -score.score;

                // Research if necessary
                if score.score > alpha && reduction > 0 {
                    score = self.alpha_beta(
                        &new_pos,
                        -alpha - 1,
                        -alpha,
                        new_depth,
                        ply + 1,
                        false,
                        search_data,
                    );
                    score.score = -score.score;
                }

                // Full window search for PV
                if is_pv && score.score > alpha && score.score < beta {
                    score = self.alpha_beta(
                        &new_pos,
                        -beta,
                        -alpha,
                        new_depth,
                        ply + 1,
                        true,
                        search_data,
                    );
                    score.score = -score.score;
                }
            }

            new_pos.unmake_move(mv);

            if score.score > best_score.score {
                best_score = score;
                best_move = mv;

                if score.score > alpha {
                    alpha = score.score;
                    
                    // PV update
                    search_data.pv[ply].clear();
                    search_data.pv[ply].push(mv);
                    if ply + 1 < MAX_PLY {
let (left, right) = search_data.pv.split_at_mut(ply + 1);
left[ply].extend_from_slice(&right[0]);                    }

                    if score.score >= beta {
                        // Beta cutoff
                        if is_quiet {
                            self.update_killers(ply, mv, search_data);
                            self.update_history(mv, depth, true, search_data);
                            
                            // Penalize quiet moves that didn't cause cutoff
                            for &failed_mv in &quiet_moves {
                                if failed_mv != mv {
                                    self.update_history(failed_mv, depth, false, search_data);
                                }
                            }
                        }
                        
                        break;
                    }
                }
            }
        }

        // TT store
        let flag = if best_score.score >= beta {
            TTFlag::LowerBound
        } else if best_score.score <= alpha {
            TTFlag::UpperBound  
        } else {
            TTFlag::Exact
        };

        {
            let mut tt = self.tt.write();
            tt.store(
                pos.hash,
                depth as u8,
                self.score_to_tt(best_score, ply),
                flag,
                best_move,
                ply as u8,
            );
        }

        best_score
    }

    // Quiescence search
    fn quiescence(
        &self,
        pos: &Position,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        search_data: &mut SearchData,
    ) -> i32 {
        self.stats.qnodes.fetch_add(1, Ordering::Relaxed);

        if ply >= MAX_PLY - 1 {
            return evaluate(pos).score;
        }

        let in_check = pos.is_in_check(pos.side_to_move);
        
        let stand_pat = if in_check {
            -INFINITY
        } else {
            evaluate(pos).score
        };

        // Stand pat cutoff
        if !in_check {
            if stand_pat >= beta {
                return stand_pat;
            }
            
            alpha = alpha.max(stand_pat);
        }

        // Generate moves
        let moves = if in_check {
            generate_legal_moves(pos)
        } else {
            self.generate_captures(pos)
        };

        // Checkmate detection
        if moves.is_empty() && in_check {
            return -MATE_VALUE + ply as i32;
        }

        let mut best_score = stand_pat;

        // Move ordering for captures
        let mut scored_moves: Vec<(Move, i32)> = moves.iter().map(|&mv| {
            let mut score = 0;
            
            if self.is_capture(pos, mv) {
                let capture_value = self.get_capture_value(pos, mv);
                let (attacker, _) = pos.piece_at(mv.from());
                let attacker_value = self.get_piece_value(attacker);
                score = capture_value * 10 - attacker_value;
            }
            
            if mv.is_promotion() {
                score += (mv.promotion() as i32) * 100;
            }
            
            (mv, score)
        }).collect();

        // Sort by score descending
        scored_moves.sort_unstable_by_key(|&(_, score)| -score);

        for (mv, _) in scored_moves {
            // Delta pruning
            if !in_check && !mv.is_promotion() {
                let capture_value = self.get_capture_value(pos, mv);
                if stand_pat + capture_value + 200 < alpha {
                    continue;
                }
            }

            // SEE pruning
            if !in_check && !pos.see_ge(mv, 0) {
                continue;
            }

            let mut new_pos = pos.clone();
            if !new_pos.make_move(mv) {
                continue;
            }

            let score = -self.quiescence(&new_pos, -beta, -alpha, ply + 1, search_data);
            new_pos.unmake_move(mv);

            if score > best_score {
                best_score = score;
                
                if score > alpha {
                    alpha = score;
                    
                    if score >= beta {
                        return score;
                    }
                }
            }
        }

        best_score
    }

    fn get_piece_value(&self, piece: PieceType) -> i32 {
        match piece {
            PieceType::Pawn => 100,
            PieceType::Knight => 320,
            PieceType::Bishop => 330,
            PieceType::Rook => 500,
            PieceType::Queen => 900,
            PieceType::King => 0,
            _ => 0,
        }
    }


    // Singular search
    fn singular_search(
        &self,
        pos: &Position,
        alpha: i32,
        beta: i32,
        depth: i32,
        ply: usize,
        excluded_move: Move,
        search_data: &mut SearchData,
    ) -> i32 {
        let old_excluded = search_data.stack[ply].excluded_move;
        search_data.stack[ply].excluded_move = excluded_move;
        let score = self.alpha_beta(pos, alpha, beta, depth, ply, false, search_data);
        search_data.stack[ply].excluded_move = old_excluded;
        score.score
    }

    ///// Helper Functions

    fn should_stop(&self) -> bool {
        self.stop.load(Ordering::Relaxed) || 
        self.time_manager.should_stop(self.start_time)
    }

    fn score_from_tt(&self, score: EvalResult, ply: usize) -> EvalResult {
        if score.score >= MATE_VALUE - MAX_PLY as i32 {
            EvalResult {
                score: score.score - ply as i32,
                is_mate: true,
                mate_distance: score.mate_distance,
            }
        } else if score.score <= -MATE_VALUE + MAX_PLY as i32 {
            EvalResult {
                score: score.score + ply as i32,
                is_mate: true,
                mate_distance: score.mate_distance,
            }
        } else {
            EvalResult {
                score: score.score,
                is_mate: false,
                mate_distance: score.mate_distance,
            }
        }
    }

    fn score_to_tt(&self, score: EvalResult, ply: usize) -> EvalResult {
        if score.score >= MATE_VALUE - MAX_PLY as i32 {
            EvalResult {
                score: score.score - ply as i32,
                is_mate: true,
                mate_distance: score.mate_distance,
            }
        } else if score.score <= -MATE_VALUE + MAX_PLY as i32 {
           EvalResult {
                score: score.score + ply as i32,
                is_mate: true,
                mate_distance: score.mate_distance,
            }
        } else {
            EvalResult {
                score: score.score,
                is_mate: false,
                mate_distance: score.mate_distance,
            }
        }
    }

    fn has_non_pawn_material(&self, pos: &Position, color: Color) -> bool {
        pos.pieces_colored(PieceType::Knight, color) != 0 ||
        pos.pieces_colored(PieceType::Bishop, color) != 0 ||
        pos.pieces_colored(PieceType::Rook, color) != 0 ||
        pos.pieces_colored(PieceType::Queen, color) != 0
    }

    fn is_capture(&self, pos: &Position, mv: Move) -> bool {
        let (piece_at_to, _) = pos.piece_at(mv.to());
        piece_at_to != PieceType::None || mv.is_en_passant()
    }

    fn is_quiet(&self, pos: &Position, mv: Move) -> bool {
        !self.is_capture(pos, mv) && !mv.is_promotion()
    }

    fn generate_captures(&self, pos: &Position) -> Vec<Move> {
        let moves = generate_legal_moves(pos);
        moves.into_iter()
            .filter(|&mv| self.is_capture(pos, mv) || mv.is_promotion())
            .collect()
    }

    fn get_capture_value(&self, pos: &Position, mv: Move) -> i32 {
        let (captured, _) = pos.piece_at(mv.to());
        match captured {
            PieceType::Pawn => 100,
            PieceType::Knight => 320,
            PieceType::Bishop => 330,
            PieceType::Rook => 500,
            PieceType::Queen => 900,
            _ => 0,
        }
    }

    fn score_moves(
        &self, 
        moves: &[Move], 
        tt_move: Move, 
        ply: usize, 
        pos: &Position,
        search_data: &SearchData,
    ) -> Vec<i32> {
        let mut scores = Vec::with_capacity(moves.len());
        
        for (idx, &mv) in moves.iter().enumerate() {
            let mut score = 0;
            
            if mv == tt_move {
                score = 1_000_000;
            }
            // Better captures
            else if self.is_capture(pos, mv) {
                let capture_value = self.get_capture_value(pos, mv);
                let (attacker, _) = pos.piece_at(mv.from());
                let attacker_value = match attacker {
                    PieceType::Pawn => 100,
                    PieceType::Knight => 320,
                    PieceType::Bishop => 330,
                    PieceType::Rook => 500,
                    PieceType::Queen => 900,
                    PieceType::King => 0,
                    _ => 0,
                };
                
                score = 100_000 + capture_value * 10 - attacker_value;
                
                // SEE ordering
                if !pos.see_ge(mv, 0) {
                    score -= 200_000;
                }
            }
            // Promotions
            else if mv.is_promotion() {
                score = 80_000 + (mv.promotion() as i32) * 1000;
            }
            // Killer moves
            else if mv == search_data.killers[ply][0] {
                score = 70_000;
            }
            else if mv == search_data.killers[ply][1] {
                score = 60_000;
            }
            // History heuristic
            else {
                score = search_data.history[mv.from() as usize][mv.to() as usize];
            }
            
            // DÜZELTME: Stable sort için move index ekle
            score = score * 10000 + (1000 - idx as i32);
            
            scores.push(score);
        }
        
        scores
    }

    fn update_killers(&self, ply: usize, mv: Move, search_data: &mut SearchData) {
        if mv != search_data.killers[ply][0] {
            search_data.killers[ply][1] = search_data.killers[ply][0];
            search_data.killers[ply][0] = mv;
        }
    }

    fn update_history(&self, mv: Move, depth: i32, good: bool, search_data: &mut SearchData) {
        let bonus = depth * depth;
        let from = mv.from() as usize;
        let to = mv.to() as usize;
        
        if good {
            search_data.history[from][to] += bonus;
            if search_data.history[from][to] > HISTORY_MAX {
                search_data.history[from][to] = HISTORY_MAX;
            }
        } else {
            search_data.history[from][to] -= bonus;
            if search_data.history[from][to] < -HISTORY_MAX {
                search_data.history[from][to] = -HISTORY_MAX;
            }
        }
    }

    fn lmr_reduction(&self, depth: i32, move_count: usize, improving: bool) -> f64 {
        // DÜZELTME: Sabit reduction table kullan
        let base_reduction = match (depth, move_count) {
            (d, m) if d >= 3 && m > 3 => 2,
            (d, m) if d >= 3 && m > 1 => 1,
            _ => 0,
        };
        
        let mut reduction = base_reduction;
        
        if !improving {
            reduction += 1;
        }
        
        reduction.max(0) as f64
    }

    fn print_info(&self, depth: i32, score: EvalResult, pv: &[Move], start_time: Instant) {
        let elapsed = start_time.elapsed().as_millis() as u64;
        let nodes = self.stats.nodes.load(Ordering::Relaxed);
        let nps = if elapsed > 0 { nodes * 1000 / elapsed } else { 0 };
        
        print!("info depth {} score ", depth);
        
        if score.is_mate {
            let mate_in = score.mate_distance;
            print!("mate {} ", mate_in);
        } else {
            print!("cp {} ", score.score);
        }
        
        print!("time {} nodes {} nps {} pv", elapsed, nodes, nps);
        
        for mv in pv {
            print!(" {}", self.move_to_uci(*mv));
        }
        println!();
    }

    fn move_to_uci(&self, mv: Move) -> String {
        let from = mv.from();
        let to = mv.to();
        let from_str = format!("{}{}", (b'a' + from % 8) as char, from / 8 + 1);
        let to_str = format!("{}{}", (b'a' + to % 8) as char, to / 8 + 1);
        
        let mut result = format!("{}{}", from_str, to_str);
        
        if mv.is_promotion() {
            let promo_char = match mv.promotion() {
                PieceType::Queen => 'q',
                PieceType::Rook => 'r',
                PieceType::Bishop => 'b',
                PieceType::Knight => 'n',
                _ => 'q',
            };
            result.push(promo_char);
        }
        
        result
    }
}

// SEE helper function for position.rs
pub fn see_ge_threshold(pos: &Position, mv: Move, threshold: i32) -> bool {
    // Simple SEE implementation
    let (captured, _) = pos.piece_at(mv.to());
    let capture_value = match captured {
        PieceType::Pawn => 100,
        PieceType::Knight => 320,
        PieceType::Bishop => 330,
        PieceType::Rook => 500,
        PieceType::Queen => 900,
        _ => 0,
    };
    
    if capture_value == 0 && !mv.is_promotion() {
        return threshold <= 0;
    }
    
    let (attacker, _) = pos.piece_at(mv.from());
    let attacker_value = match attacker {
        PieceType::Pawn => 100,
        PieceType::Knight => 320,
        PieceType::Bishop => 330,
        PieceType::Rook => 500,
        PieceType::Queen => 900,
        _ => 0,
    };
    
    capture_value - attacker_value >= threshold
}

// Parallel search 
use std::thread;
use std::sync::mpsc::{channel, Sender, Receiver};

pub struct ParallelSearch {
    pub thread_count: usize,
    pub shared_tt: Arc<RwLock<TranspositionTable>>,
    pub shared_stop: Arc<AtomicBool>,
    pub shared_stats: Arc<SearchStats>,
}

#[derive(Clone)]
struct ThreadResult {
    
    thread_id: usize,
    depth: i32,
    score: i32,
    best_move: Move,
    pv: Vec<Move>,
}

impl ParallelSearch {
    pub fn new(thread_count: usize, tt_size_mb: usize) -> Self {
        let thread_count = thread_count.max(1).min(256);
        
        Self {
            thread_count,
            shared_tt: Arc::new(RwLock::new(TranspositionTable::new(tt_size_mb))),
            shared_stop: Arc::new(AtomicBool::new(false)),
            shared_stats: Arc::new(SearchStats::default()),
        }
    }
    
    pub fn search(
        &mut self,
        pos: &Position,
        max_depth: i32,
        time_manager: TimeManager,
    ) -> SearchResult {
        // Reset stop flag for new search
        self.shared_stop.store(false, Ordering::SeqCst);
        self.shared_stats.clear();
        
        {
            let mut tt = self.shared_tt.write();
            tt.new_search();
        }
        
        // Channel for thread communication
        let (tx, rx): (Sender<ThreadResult>, Receiver<ThreadResult>) = channel();
        
        // Store thread handles
        let mut handles = Vec::new();
        let start_time = Instant::now();
        
        // begin worker threads
        for thread_id in 0..self.thread_count {
            let pos_clone = pos.clone();
            let tx_clone = tx.clone();
            let tt_clone = Arc::clone(&self.shared_tt);
            let stop_clone = Arc::clone(&self.shared_stop);
            let stats_clone = Arc::clone(&self.shared_stats);
            let time_mgr_clone = time_manager.clone();
            
            let handle = thread::spawn(move || {
                let search = AlphaBetaSearch {
                    tt: tt_clone,
                    time_manager: time_mgr_clone.clone(),
                    stats: stats_clone,
                    stop: stop_clone.clone(),  // Use the cloned stop flag
                    start_time,
                };
                
                let mut search_data = SearchData::new();
                search_data.clear();
                
                let depth_offset = thread_id % 4;
                
                for depth in (1 + depth_offset as i32)..=max_depth {
                    // Check stop flag at start of each iteration
                    if search.stop.load(Ordering::SeqCst) || 
                       search.time_manager.should_stop(start_time) {
                        break;
                    }
                    
                    // Add some randomness to aspiration windows for diversity
                    let window_randomness = if thread_id == 0 { 
                        0 
                    } else { 
                        (thread_id as i32 * 7) % 20 - 10 
                    };
                    
                    let alpha = -INFINITY + window_randomness;
                    let beta = INFINITY + window_randomness;
                    
                    let score = search.alpha_beta(
                        &pos_clone,
                        alpha,
                        beta,
                        depth,
                        0,
                        thread_id == 0, // Only main thread is PV
                        &mut search_data,
                    );
                    
                    // Check stop flag again before sending result
                    if !search.stop.load(Ordering::SeqCst) && !search_data.pv[0].is_empty() {
                        let temp_score = score.score;
                        let result = ThreadResult {
                            thread_id,
                            depth,
                            score:temp_score,
                            best_move: search_data.pv[0][0],
                            pv: search_data.pv[0].clone(),
                        };
                        
                        // Send result, ignore error if receiver dropped
                        let _ = tx_clone.send(result);
                        
                        // Main thread prints info
                        if thread_id == 0 && !search.stop.load(Ordering::SeqCst) {
                            search.print_info(depth, score, &search_data.pv[0], start_time);
                        }
                    }
                }
            });
            
            handles.push(handle);
        }
        
        drop(tx); // Close sender so receiver knows when all threads are done
        
        // Collect results
        let mut best_result = ThreadResult {
            thread_id: 0,
            depth: 0,
            score: -INFINITY,
            best_move: Move::null(),
            pv: Vec::new(),
        };
        
        let timeout_duration = if time_manager.is_infinite() {
            Duration::from_secs(3600) // 1 hour for infinite
        } else {
            Duration::from_millis(time_manager.allocated_ms() + 100)
        };
        
        let deadline = start_time + timeout_duration;
        
        // Process results as they come in
        loop {
            // Check if we should stop
            if self.shared_stop.load(Ordering::SeqCst) || time_manager.should_stop(start_time) {
                // Signal all threads to stop
                self.shared_stop.store(true, Ordering::SeqCst);
                break;
            }
            
            let now = Instant::now();
            if now >= deadline {
                self.shared_stop.store(true, Ordering::SeqCst);
                break;
            }
            
            let timeout = deadline.saturating_duration_since(now).min(Duration::from_millis(50));
            
            match rx.recv_timeout(timeout) {
                Ok(result) => {
                    // Update best result if this is deeper or better at same depth
                    if result.depth > best_result.depth ||
                       (result.depth == best_result.depth && result.score > best_result.score) {
                        best_result = result;
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    // Continue checking
                    continue;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    // All senders dropped, threads finished
                    break;
                }
            }
        }
        
        // Signal all threads to stop
        self.shared_stop.store(true, Ordering::SeqCst);
        
        // Wait for all threads with timeout
        let join_deadline = Instant::now() + Duration::from_secs(2);
        for handle in handles {
            let remaining = join_deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                // Timeout reached, abandon remaining threads
                break;
            }
            
            // Try to join with timeout
            match handle.join() {
                Ok(_) => {},
                Err(_) => {
                    // Thread panicked, continue
                }
            }
        }
        
        // Return results
        let nodes = self.shared_stats.nodes.load(Ordering::Relaxed);
        let elapsed = start_time.elapsed().as_millis() as u64;
        let nps = if elapsed > 0 { nodes * 1000 / elapsed } else { 0 };
        
        SearchResult {
            best_move: best_result.best_move,
            score: best_result.score,
            depth: best_result.depth as u8,
            pv: best_result.pv,
            nodes,
            time_ms: elapsed,
            nps,
        }
    }
    
    pub fn clear_hash(&mut self) {
        let mut tt = self.shared_tt.write();
        tt.clear();
    }

    }


// Search result structure
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub best_move: Move,
    pub score: i32,
    pub depth: u8,
    pub pv: Vec<Move>,
    pub nodes: u64,
    pub time_ms: u64,
    pub nps: u64,
}

// Fast alpha-beta search entry point for UCI compatibility
pub fn alpha_beta_search(
    pos: &Position,
    depth: u8,
    time_mgr: &TimeManager,
    stop: &Arc<AtomicBool>,
) -> SearchResult {
    // Get number of threads from environment or use default
    let thread_count = std::env::var("THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1)
        .min(256);
    
    let tt_size_mb = std::env::var("HASH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(256)
        .max(1)
        .min(32768);
    
    let mut parallel_search = ParallelSearch::new(thread_count, tt_size_mb);
    parallel_search.search(pos, depth as i32, time_mgr.clone())
}