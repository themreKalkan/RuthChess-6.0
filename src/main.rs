mod board;
mod eval;
mod movegen;
mod search;
mod uci;

use crate::board::zobrist;
use crate::movegen::magic;
use crate::board::position::init_attack_tables;
use crate::eval::evaluate::init_eval;
use crate::eval::evaluate::evaluate;
use crate::board::position::{Position,Move,PieceType};


use std::io::{self, BufRead, BufReader};
use std::sync::Arc;
use std::thread;


use uci::protocol;
use RuthChessOVI::board::bitboard::square_to_algebraic;



fn main() {

    

    zobrist::init_zobrist();
    magic::init_magics();
    init_attack_tables();
    init_eval();

    
    uci::protocol::run_uci();
}
