use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use crate::{
    board::position::{Position, Move},
    eval::evaluate::Score,
    search::alphabeta::alpha_beta_search,
    search::time_management::TimeManager,
};

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub best_move: Move,
    pub score: Score,
    pub depth: u8,
    pub nodes: u64,
    pub best_pv: Vec<Move>,
}





