
pub mod material;
pub mod pst;
pub mod pawns;
pub mod mobility;
pub mod space;
pub mod king_safety;
pub mod imbalance;
pub mod threats;
pub mod eval_util;
pub mod evaluate;
pub mod weights;
pub mod initiative;

pub use evaluate::{Score, EvalResult, evaluate, evaluate_fast, evaluate_detailed, MATE_VALUE, DRAW_VALUE};
pub use eval_util::{EvalCache, phase_value, is_endgame, is_middlegame};
pub use material::{calculate_phase, MAX_PHASE};

pub const TEMPO_BONUS: i32 = 28;
pub const LAZY_EVAL_MARGIN: i32 = 400;

pub const EVAL_CACHE_SIZE: usize = 1024 * 1024;



pub fn quick_eval(pos: &crate::board::position::Position) -> i32 {
    evaluate::evaluate_fast(pos)
}

pub fn needs_deep_eval(pos: &crate::board::position::Position) -> bool {
    evaluate::needs_careful_evaluation(pos)
}

pub fn complexity_score(pos: &crate::board::position::Position) -> i32 {
    evaluate::evaluate_complexity(pos)
}