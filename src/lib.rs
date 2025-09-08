pub mod board {
    pub mod position;
    pub mod bitboard;
    pub mod zobrist;
}

pub mod eval {
    pub mod material;
    pub mod pst;
    pub mod pawns;
    pub mod evaluate;
    pub mod mobility;
    pub mod space;
    pub mod king_safety;
    pub mod imbalance;
    pub mod threats;
    pub mod eval_util;
    pub mod weights;
    pub mod initiative;


}

pub mod movegen;
pub mod uci;
pub mod search;
