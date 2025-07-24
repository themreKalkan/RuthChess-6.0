
use crate::board::position::{Position, Color};
use crate::eval::material::{ calculate_phase};
use crate::eval::{
    material, pst, pawns, mobility, space, king_safety, 
    imbalance, threats, eval_util
};
const MAX_PHASE: u32 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Score {
    pub mg: i32,
    pub eg: i32,
}

impl Score {
    #[inline(always)]
    pub const fn new(mg: i32, eg: i32) -> Self {
        Self { mg, eg }
    }
    
    #[inline(always)]
    pub const fn zero() -> Self {
        Self { mg: 0, eg: 0 }
    }
    
    #[inline(always)]
    pub const fn add(self, other: Self) -> Self {
        Self {
            mg: self.mg + other.mg,
            eg: self.eg + other.eg,
        }
    }
    
    #[inline(always)]
    pub const fn sub(self, other: Self) -> Self {
        Self {
            mg: self.mg - other.mg,
            eg: self.eg - other.eg,
        }
    }
    
    #[inline(always)]
    pub const fn neg(self) -> Self {
        Self {
            mg: -self.mg,
            eg: -self.eg,
        }
    }
    
    #[inline(always)]
    pub fn interpolate(self, phase: u32) -> i32 {
        ((self.mg as i64 * phase as i64 + self.eg as i64 * (MAX_PHASE as i64 - phase as i64)) / MAX_PHASE as i64) as i32
    }
}

impl std::ops::Add for Score {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        self.add(other)
    }
}

impl std::ops::Sub for Score {
    type Output = Self;
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        self.sub(other)
    }
}

impl std::ops::Neg for Score {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        self.neg()
    }
}

pub const MATE_VALUE: i32 = 32000;
pub const DRAW_VALUE: i32 = 0;
pub const TEMPO_BONUS: i32 = 8;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvalResult {
    pub score: i32,
    pub is_mate: bool,
    pub mate_distance: i32,
}

impl EvalResult {
    pub fn new(score: i32) -> Self {
        let is_mate = score.abs() > MATE_VALUE - 1000;
        let mate_distance = if is_mate {
            MATE_VALUE - score.abs()
        } else {
            0
        };
        
        Self {
            score,
            is_mate,
            mate_distance,
        }
    }
    
    pub fn mate_in(moves: i32) -> Self {
        Self {
            score: MATE_VALUE - moves,
            is_mate: true,
            mate_distance: moves,
        }
    }
    
    pub fn mated_in(moves: i32) -> Self {
        Self {
            score: -MATE_VALUE + moves,
            is_mate: true,
            mate_distance: moves,
        }
    }
    
    pub fn draw() -> Self {
        Self {
            score: DRAW_VALUE,
            is_mate: false,
            mate_distance: 0,
        }
    }
}

#[inline(always)]
pub fn evaluate_int(pos: &Position) -> i32 {
    if let Some((cached_score, cached_phase)) = eval_util::EvalCache::probe(pos.hash) {
        let current_phase = calculate_phase(pos);
        if cached_phase == current_phase {
            let final_score = finalize_evaluation(cached_score.interpolate(current_phase), pos);
            return final_score;
        }
    }
    
    
    let phase = calculate_phase(pos);
    
    let mut total_score = Score::zero();
    
    total_score = total_score.add(material::evaluate_material(pos));
    
    total_score = total_score.add(material::evaluate_bishop_pair(pos));
    total_score = total_score.add(material::evaluate_minor_piece_adjustments(pos));
    
    total_score = total_score.add(pst::evaluate_pst(pos));
    
    total_score = total_score.add(pawns::evaluate_pawns(pos));
    
    total_score = total_score.add(mobility::evaluate_mobility(pos));
    
    total_score = total_score.add(space::evaluate_space(pos));
    
    total_score = total_score.add(king_safety::evaluate_king_safety(pos));
    
    total_score = total_score.add(imbalance::evaluate_imbalance(pos));
    
    total_score = total_score.add(threats::evaluate_threats(pos));
    
    eval_util::EvalCache::store(pos.hash, total_score, phase);
    
    let interpolated_score = total_score.interpolate(phase);
    
    let final_score = finalize_evaluation(interpolated_score, pos);
    
    final_score
}

#[inline(always)]
pub fn evaluate(pos: &Position) -> EvalResult {
    if let Some((cached_score, cached_phase)) = eval_util::EvalCache::probe(pos.hash) {
        let current_phase = calculate_phase(pos);
        if cached_phase == current_phase {
            let final_score = finalize_evaluation(cached_score.interpolate(current_phase), pos);
            return EvalResult::new(final_score);
        }
    }
    
    
    let phase = calculate_phase(pos);
    
    let mut total_score = Score::zero();
    
    total_score = total_score.add(material::evaluate_material(pos));
    
    total_score = total_score.add(material::evaluate_bishop_pair(pos));
    total_score = total_score.add(material::evaluate_minor_piece_adjustments(pos));
    
    total_score = total_score.add(pst::evaluate_pst(pos));
    
    total_score = total_score.add(pawns::evaluate_pawns(pos));
    
    total_score = total_score.add(mobility::evaluate_mobility(pos));
    
    total_score = total_score.add(space::evaluate_space(pos));
    
    total_score = total_score.add(king_safety::evaluate_king_safety(pos));
    
    total_score = total_score.add(imbalance::evaluate_imbalance(pos));
    
    total_score = total_score.add(threats::evaluate_threats(pos));
    
    eval_util::EvalCache::store(pos.hash, total_score, phase);
    
    let interpolated_score = total_score.interpolate(phase);
    
    let final_score = finalize_evaluation(interpolated_score, pos);
    
    EvalResult::new(final_score)
}

pub fn evaluate_detailed(pos: &Position) -> eval_util::EvalDebugInfo {
    let mut debug_info = eval_util::EvalDebugInfo::new();
    debug_info.phase = calculate_phase(pos);
    debug_info.flags = eval_util::EvalFlags::from_position(pos);
    
    debug_info.material = material::evaluate_material(pos)
        .add(material::evaluate_bishop_pair(pos))
        .add(material::evaluate_minor_piece_adjustments(pos));
    
    debug_info.positional = pst::evaluate_pst(pos);
    debug_info.pawn_structure = pawns::evaluate_pawns(pos);
    debug_info.mobility = mobility::evaluate_mobility(pos);
    debug_info.space = space::evaluate_space(pos);
    debug_info.king_safety = king_safety::evaluate_king_safety(pos);
    debug_info.threats = threats::evaluate_threats(pos);
    
    debug_info
}

fn is_immediate_draw(pos: &Position) -> bool {
    if pos.halfmove_clock >= 100 {
        return true;
    }
    
    if material::is_material_draw(pos) {
        return true;
    }
    
    false
}

pub fn evaluate_for_debug(pos: &Position) -> EvalResult {
    if let Some((cached_score, cached_phase)) = eval_util::EvalCache::probe(pos.hash) {
        let current_phase = calculate_phase(pos);
        if cached_phase == current_phase {
            let final_score = finalize_evaluation_debug(cached_score.interpolate(current_phase), pos);
            return EvalResult::new(final_score);
        }
    }
    
    if is_immediate_draw(pos) {
        return EvalResult::draw();
    }
    
    let phase = calculate_phase(pos);
    
    let mut total_score = Score::zero();
    
    total_score = total_score.add(material::evaluate_material(pos));
    
    total_score = total_score.add(material::evaluate_bishop_pair(pos));
    total_score = total_score.add(material::evaluate_minor_piece_adjustments(pos));
    
    total_score = total_score.add(pst::evaluate_pst(pos));
    
    total_score = total_score.add(pawns::evaluate_pawns(pos));
    
    total_score = total_score.add(mobility::evaluate_mobility(pos));
    
    total_score = total_score.add(space::evaluate_space(pos));
    
    total_score = total_score.add(king_safety::evaluate_king_safety(pos));
    
    total_score = total_score.add(imbalance::evaluate_imbalance(pos));
    
    total_score = total_score.add(threats::evaluate_threats(pos));
    
    eval_util::EvalCache::store(pos.hash, total_score, phase);
    
    let interpolated_score = total_score.interpolate(phase);
    
    let final_score = finalize_evaluation_debug(interpolated_score, pos);
    
    EvalResult::new(final_score)
}



fn finalize_evaluation_debug(mut score: i32, pos: &Position) -> i32 {
    score += TEMPO_BONUS;
    
    score = eval_util::apply_contempt(score, pos);
    
    score = eval_util::scale_evaluation(score, pos);
    
    
    eval_util::normalize_score(score)
}

fn finalize_evaluation(mut score: i32, pos: &Position) -> i32 {
    score += TEMPO_BONUS;
    
    score = eval_util::apply_contempt(score, pos);
    
    score = eval_util::scale_evaluation(score, pos);
    
    if pos.side_to_move == Color::Black {
        score = -score;
    }
    
    eval_util::normalize_score(score)
}

pub fn evaluate_fast(pos: &Position) -> i32 {
    let mut score = Score::zero();
    
    score = score.add(material::evaluate_material(pos));
    
    score = score.add(pst::evaluate_pst(pos));
    
    let phase = calculate_phase(pos);
    let mut final_score = score.interpolate(phase);
    
    final_score += TEMPO_BONUS;
    if pos.side_to_move == Color::Black {
        final_score = -final_score;
    }
    
    eval_util::normalize_score(final_score)
}

pub fn evaluate_lazy(pos: &Position, alpha: i32, beta: i32) -> Option<i32> {
    const LAZY_MARGIN: i32 = 400;
    
    let quick_eval = evaluate_fast(pos);
    
    if quick_eval + LAZY_MARGIN < alpha {
        return Some(quick_eval);
    }
    
    if quick_eval - LAZY_MARGIN > beta {
        return Some(quick_eval);
    }
    
    None
}

pub fn evaluate_for_color(pos: &Position, color: Color) -> i32 {
    let eval_result = evaluate(pos);
    let mut score = eval_result.score;
    
    if pos.side_to_move == Color::Black {
        score = -score;
    }
    
    if color == Color::Black {
        score = -score;
    }
    
    score
}

pub fn is_winning(score: i32) -> bool {
    score.abs() > 300
}

pub fn is_drawn(score: i32) -> bool {
    score.abs() < 50
}

pub fn score_to_win_probability(score: i32) -> f32 {
    let normalized = score as f32 / 400.0;
    let sigmoid = 1.0 / (1.0 + (-normalized).exp());
    sigmoid
}

pub fn endgame_scale_factor(pos: &Position) -> f32 {
    use crate::board::position::PieceType;
    
    let phase = calculate_phase(pos);
    if !eval_util::is_endgame(phase) {
        return 1.0;
    }
    
    let mut total_material = 0;
    for color in [Color::White, Color::Black] {
        total_material += pos.piece_count(color, PieceType::Pawn) * 1;
        total_material += pos.piece_count(color, PieceType::Knight) * 3;
        total_material += pos.piece_count(color, PieceType::Bishop) * 3;
        total_material += pos.piece_count(color, PieceType::Rook) * 5;
        total_material += pos.piece_count(color, PieceType::Queen) * 9;
    }
    
    if total_material <= 6 {
        0.5
    } else if total_material <= 12 {
        0.75
    } else {
        1.0
    }
}

pub fn get_eval_components(pos: &Position) -> [i32; 8] {
    let debug_info = evaluate_detailed(pos);
    
    [
        debug_info.material.interpolate(debug_info.phase),
        debug_info.positional.interpolate(debug_info.phase),
        debug_info.pawn_structure.interpolate(debug_info.phase),
        debug_info.mobility.interpolate(debug_info.phase),
        debug_info.space.interpolate(debug_info.phase),
        debug_info.king_safety.interpolate(debug_info.phase),
        threats::evaluate_threats(pos).interpolate(debug_info.phase),
        imbalance::evaluate_imbalance(pos).interpolate(debug_info.phase),
    ]
}

pub fn evaluate_complexity(pos: &Position) -> i32 {
    eval_util::calculate_complexity(pos)
}

pub fn needs_careful_evaluation(pos: &Position) -> bool {
    if threats::has_tactical_threats(pos, Color::White) || 
       threats::has_tactical_threats(pos, Color::Black) {
        return true;
    }
    
    if king_safety::is_king_in_danger(pos, Color::White) ||
       king_safety::is_king_in_danger(pos, Color::Black) {
        return true;
    }
    
    let complexity = evaluate_complexity(pos);
    complexity > 60
}

pub fn static_exchange_estimate(pos: &Position, square: u8) -> i32 {
    let (piece_type, piece_color) = pos.piece_at(square);
    if piece_type == crate::board::position::PieceType::None {
        return 0;
    }
    
    let piece_value = material::PIECE_VALUES[piece_type as usize];
    let base_value = if piece_color == Color::White {
        piece_value.mg
    } else {
        -piece_value.mg
    };
    
    let white_attackers = threats::count_threats(pos, Color::White);
    let black_attackers = threats::count_threats(pos, Color::Black);
    
    let attacker_bonus = (white_attackers - black_attackers) * 10;
    base_value + attacker_bonus
}

pub fn tactical_complexity(pos: &Position) -> i32 {
    let mut complexity = 0;
    
    complexity += threats::count_threats(pos, Color::White);
    complexity += threats::count_threats(pos, Color::Black);
    
    if king_safety::is_king_in_danger(pos, Color::White) {
        complexity += 5;
    }
    if king_safety::is_king_in_danger(pos, Color::Black) {
        complexity += 5;
    }
    
    if imbalance::has_drawing_tendency(pos) {
        complexity -= 3;
    }
    
    complexity
}

pub fn print_evaluation(pos: &Position) {
    let debug_info = evaluate_detailed(pos);
    
    println!("=== Position Evaluation ===");
    println!("Phase: {}/{}", debug_info.phase, material::MAX_PHASE);
    println!("Flags: {:?}", debug_info.flags);
    println!();
    
    println!("Component Scores (MG/EG -> Interpolated):");
    println!("Material:      {:+4}/{:+4} -> {:+4}", 
             debug_info.material.mg, debug_info.material.eg,
             debug_info.material.interpolate(debug_info.phase));
    
    println!("Positional:    {:+4}/{:+4} -> {:+4}", 
             debug_info.positional.mg, debug_info.positional.eg,
             debug_info.positional.interpolate(debug_info.phase));
    
    println!("Pawns:         {:+4}/{:+4} -> {:+4}", 
             debug_info.pawn_structure.mg, debug_info.pawn_structure.eg,
             debug_info.pawn_structure.interpolate(debug_info.phase));
    
    println!("Mobility:      {:+4}/{:+4} -> {:+4}", 
             debug_info.mobility.mg, debug_info.mobility.eg,
             debug_info.mobility.interpolate(debug_info.phase));
    
    println!("Space:         {:+4}/{:+4} -> {:+4}", 
             debug_info.space.mg, debug_info.space.eg,
             debug_info.space.interpolate(debug_info.phase));
    
    println!("King Safety:   {:+4}/{:+4} -> {:+4}", 
             debug_info.king_safety.mg, debug_info.king_safety.eg,
             debug_info.king_safety.interpolate(debug_info.phase));
    
    println!("Threats:       {:+4}/{:+4} -> {:+4}", 
             debug_info.threats.mg, debug_info.threats.eg,
             debug_info.threats.interpolate(debug_info.phase));
    
    println!();
    println!("Total MG: {:+4}", debug_info.total_mg());
    println!("Total EG: {:+4}", debug_info.total_eg());
    println!("Final:    {:+4}", debug_info.interpolated());
    
    let final_eval = evaluate_for_debug(pos);
    println!("With adjustments: {:+4} cp", final_eval.score);
    
    if final_eval.is_mate {
        if final_eval.score > 0 {
            println!("Mate in {} moves", final_eval.mate_distance);
        } else {
            println!("Mated in {} moves", final_eval.mate_distance);
        }
    }
    
    println!("Win probability: {:.1}%", score_to_win_probability(final_eval.score) * 100.0);
}

pub fn init_eval() {
    pst::init_pst_tables();
    mobility::init_mobility_tables();
    pawns::init_pawn_masks();
    eval_util::init_eval_cache();
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::position::Position;
    
    #[test]
    fn test_startpos_evaluation() {
        let pos = Position::startpos();
        let eval_result = evaluate(&pos);
        
        assert!(eval_result.score.abs() < 100);
        assert!(!eval_result.is_mate);
    }
    
    #[test]
    fn test_fast_evaluation() {
        let pos = Position::startpos();
        let fast_eval = evaluate_fast(&pos);
        let full_eval = evaluate(&pos);
        
        assert!((fast_eval - full_eval.score).abs() < 200);
    }
    
    #[test]
    fn test_lazy_evaluation() {
        let pos = Position::startpos();
        let lazy_result = evaluate_lazy(&pos, -1000, 1000);
        
        assert!(lazy_result.is_none());
    }
    
    #[test]
    fn test_eval_result() {
        let result = EvalResult::new(100);
        assert_eq!(result.score, 100);
        assert!(!result.is_mate);
        
        let mate_result = EvalResult::mate_in(5);
        assert!(mate_result.is_mate);
        assert_eq!(mate_result.mate_distance, 5);
    }
    
    #[test]
    fn test_score_to_probability() {
        assert!((score_to_win_probability(0) - 0.5).abs() < 0.01);
        assert!(score_to_win_probability(400) > 0.7);
        assert!(score_to_win_probability(-400) < 0.3);
    }
    
    #[test]
    fn test_draw_detection() {
        let draw_pos = Position::from_fen("8/8/8/8/8/8/8/4k1K1 w - - 99 1").unwrap();
        let eval_result = evaluate(&draw_pos);
        assert_eq!(eval_result.score, DRAW_VALUE);
    }
    
    #[test]
    fn test_detailed_evaluation() {
        let pos = Position::startpos();
        let debug_info = evaluate_detailed(&pos);
        
        assert!(debug_info.material.mg > 0);
        assert!(debug_info.material.eg > 0);
        
        assert!(debug_info.interpolated().abs() < 200);
    }
    
    #[test]
    fn test_complexity_evaluation() {
        let pos = Position::startpos();
        let complexity = evaluate_complexity(&pos);
        
        assert!(complexity > 20);
        assert!(complexity < 100);
    }
}