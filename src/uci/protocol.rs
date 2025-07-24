
use crate::{
    board::{
        position::{Position, Move, PieceType, MoveType, Color},
        zobrist,
        bitboard::square_to_algebraic,
    },
    movegen::{
        legal_moves::{generate_legal_moves,move_to_uci},
        magic::init_magics,
    },
    search::{
        alphabeta::{SearchResult, ParallelSearch},
        time_management::TimeManager,
        transposition::TranspositionTable,
    },
};

use std::io::{self, BufRead, Write};
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

const ENGINE_NAME: &str = "RuthChess";
const ENGINE_AUTHOR: &str = "EmreKalkan";
const ENGINE_VERSION: &str = "6.0";
const DEFAULT_TT_SIZE_MB: usize = 256;
const MIN_TT_SIZE_MB: usize = 1;
const MAX_TT_SIZE_MB: usize = 32768;
const DEFAULT_THREADS: usize = 1;
const MAX_THREADS: usize = 256;

pub struct ImprovedUciEngine {
    position: Arc<RwLock<Position>>,
    
    search_thread: Arc<Mutex<Option<thread::JoinHandle<()>>>>,
    stop_flag: Arc<AtomicBool>,
    searching: Arc<AtomicBool>,
    pondering: Arc<AtomicBool>,
    
    debug_mode: AtomicBool,
    tt_size_mb: usize,
    thread_count: usize,
    move_overhead: u64,
    ponder_enabled: bool,
    
    parallel_search: Arc<Mutex<ParallelSearch>>,
}

impl ImprovedUciEngine {
    pub fn new() -> Self {
        zobrist::init_zobrist();
        crate::board::position::init_attack_tables();
        init_magics();
        
        let tt_size = DEFAULT_TT_SIZE_MB;
        let threads = DEFAULT_THREADS;
        
        Self {
            position: Arc::new(RwLock::new(Position::startpos())),
            search_thread: Arc::new(Mutex::new(None)),
            stop_flag: Arc::new(AtomicBool::new(false)),
            searching: Arc::new(AtomicBool::new(false)),
            pondering: Arc::new(AtomicBool::new(false)),
            debug_mode: AtomicBool::new(false),
            tt_size_mb: tt_size,
            thread_count: threads,
            move_overhead: 50,
            ponder_enabled: true,
            parallel_search: Arc::new(Mutex::new(ParallelSearch::new(threads, tt_size))),
        }
    }

    pub fn run(&mut self) {
        let stdin = io::stdin();
        
        println!("{} {} by {}", ENGINE_NAME, ENGINE_VERSION, ENGINE_AUTHOR);
        io::stdout().flush().unwrap();
        
        for line in stdin.lock().lines() {
            match line {
                Ok(command) => {
                    let trimmed = command.trim();
                    if !trimmed.is_empty() {
                        self.handle_command(trimmed);
                    }
                }
                Err(e) => {
                    self.debug(&format!("Error reading input: {}", e));
                    break;
                }
            }
            
            io::stdout().flush().unwrap();
        }
    }

    fn handle_command(&mut self, command: &str) {
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return;
        }

        self.debug(&format!("<<< {}", command));

        match parts[0] {
            "uci" => self.uci_command(),
            "debug" => self.debug_command(&parts),
            "isready" => self.isready_command(),
            "setoption" => self.setoption_command(&parts),
            "register" => self.register_command(),
            "ucinewgame" => self.ucinewgame_command(),
            "position" => self.position_command(&parts),
            "go" => self.go_command(&parts),
            "stop" => self.stop_command(),
            "ponderhit" => self.ponderhit_command(),
            "quit" => self.quit_command(),
            "d" | "display" => self.display_command(),
            "eval" => self.eval_command(),
            _ => self.debug(&format!("Unknown command: {}", parts[0])),
        }
    }

    fn uci_command(&self) {
        println!("id name {} {}", ENGINE_NAME, ENGINE_VERSION);
        println!("id author {}", ENGINE_AUTHOR);
        
        println!("option name Hash type spin default {} min {} max {}", 
                 DEFAULT_TT_SIZE_MB, MIN_TT_SIZE_MB, MAX_TT_SIZE_MB);
        println!("option name Threads type spin default {} min 1 max {}", 
                 DEFAULT_THREADS, MAX_THREADS);
        println!("option name Ponder type check default true");
        println!("option name Move Overhead type spin default 50 min 0 max 5000");
        println!("option name Clear Hash type button");
        
        println!("uciok");
    }

    fn debug_command(&mut self, parts: &[&str]) {
        if parts.len() >= 2 {
            let debug_on = parts[1] == "on";
            self.debug_mode.store(debug_on, Ordering::Relaxed);
            self.debug(&format!("Debug mode {}", if debug_on { "on" } else { "off" }));
        }
    }

    fn isready_command(&mut self) {
        self.ensure_search_stopped();
        
        println!("readyok");
    }

    fn setoption_command(&mut self, parts: &[&str]) {
        self.ensure_search_stopped();
        
        if parts.len() < 2 {
            return;
        }

        let mut name_parts = Vec::new();
        let mut value_parts = Vec::new();
        let mut parsing_name = false;
        let mut parsing_value = false;

        for &part in &parts[1..] {
            match part {
                "name" => {
                    parsing_name = true;
                    parsing_value = false;
                }
                "value" => {
                    parsing_name = false;
                    parsing_value = true;
                }
                _ => {
                    if parsing_name {
                        name_parts.push(part);
                    } else if parsing_value {
                        value_parts.push(part);
                    }
                }
            }
        }

        let option_name = name_parts.join(" ");
        let option_value = value_parts.join(" ");

        match option_name.as_str() {
            "Hash" => {
                if let Ok(size) = option_value.parse::<usize>() {
                    self.tt_size_mb = size.clamp(MIN_TT_SIZE_MB, MAX_TT_SIZE_MB);
                    let mut search = self.parallel_search.lock().unwrap();
                    *search = ParallelSearch::new(self.thread_count, self.tt_size_mb);
                    self.debug(&format!("Hash table size set to {} MB", self.tt_size_mb));
                }
            }
            "Threads" => {
                if let Ok(threads) = option_value.parse::<usize>() {
                    self.thread_count = threads.clamp(1, MAX_THREADS);
                    let mut search = self.parallel_search.lock().unwrap();
                    *search = ParallelSearch::new(self.thread_count, self.tt_size_mb);
                    self.debug(&format!("Thread count set to {}", self.thread_count));
                }
            }
            "Ponder" => {
                self.ponder_enabled = option_value.to_lowercase() == "true";
                self.debug(&format!("Pondering {}", if self.ponder_enabled { "enabled" } else { "disabled" }));
            }
            "Move Overhead" => {
                if let Ok(overhead) = option_value.parse::<u64>() {
                    self.move_overhead = overhead.clamp(0, 5000);
                    self.debug(&format!("Move overhead set to {} ms", self.move_overhead));
                }
            }
            "Clear Hash" => {
                let mut search = self.parallel_search.lock().unwrap();
                search.clear_hash();
                self.debug("Hash table cleared");
            }
            _ => {
                self.debug(&format!("Unknown option: {}", option_name));
            }
        }
    }

    fn register_command(&self) {
    }

    fn ucinewgame_command(&mut self) {
        self.debug("New game command received");
        
        self.ensure_search_stopped();
        
        if let Ok(mut pos) = self.position.write() {
            *pos = Position::startpos();
        }
        
        {
            let mut search = self.parallel_search.lock().unwrap();
            search.clear_hash();
        }
        
        crate::board::zobrist::init_zobrist();
        
        self.debug("New game started with fresh hash table");
    }

    fn position_command(&mut self, parts: &[&str]) {
        if parts.len() < 2 {
            return;
        }

        self.ensure_search_stopped();

        let mut new_position = None;
        let mut moves_idx = 0;
        
        match parts[1] {
            "startpos" => {
                new_position = Some(Position::startpos());
                for (i, &part) in parts.iter().enumerate() {
                    if part == "moves" {
                        moves_idx = i + 1;
                        break;
                    }
                }
            }
            "fen" => {
                if parts.len() < 8 {
                    self.debug("Invalid FEN command");
                    return;
                }
                
                let mut fen_parts = Vec::new();
                let mut found_moves = false;
                
                for (i, &part) in parts[2..].iter().enumerate() {
                    if part == "moves" {
                        moves_idx = i + 2 + 1;
                        found_moves = true;
                        break;
                    }
                    fen_parts.push(part);
                }
                
                if fen_parts.len() >= 4 {
                    let fen = fen_parts.join(" ");
                    match Position::from_fen(&fen) {
                        Some(pos) => new_position = Some(pos),
                        None => {
                            self.debug(&format!("Invalid FEN: {}", fen));
                            return;
                        }
                    }
                } else {
                    self.debug("Incomplete FEN");
                    return;
                }
            }
            _ => {
                self.debug("Invalid position command");
                return;
            }
        }

        if let Some(mut pos) = new_position {
            let mut move_history = Vec::new();
            
            if moves_idx > 0 && moves_idx < parts.len() {
                self.debug(&format!("Applying {} moves", parts.len() - moves_idx));
                
                for (i, &move_str) in parts[moves_idx..].iter().enumerate() {
                    self.debug(&format!("Move {}: {}", i + 1, move_str));
                    
                    if let Some(mv) = self.parse_move(&pos, move_str) {
                        let old_hash = pos.hash;
                        if !pos.make_move(mv) {
                            self.debug(&format!("Illegal move: {}", move_str));
                            break;
                        }
                        move_history.push(mv);
                        pos.calculate_hash();
                        let recalc_hash:u64 = pos.hash;
                        if pos.hash != recalc_hash {
                            self.debug(&format!("Hash mismatch after move {}: calc={:016X}, stored={:016X}", 
                                             move_str, recalc_hash, pos.hash));
                            pos.hash = recalc_hash;
                        }
                    } else {
                        self.debug(&format!("Invalid move format: {}", move_str));
                        break;
                    }
                }
            }
            
            let side_to_move = pos.side_to_move;
            let final_hash = pos.hash;
            
            if let Ok(mut current_pos) = self.position.write() {
                *current_pos = pos;
            }
            
            self.debug(&format!("Position updated. Side: {:?}, Hash: {:016X}, Moves: {}", 
                               side_to_move, final_hash, move_history.len()));
        }
    }

    fn go_command(&mut self, parts: &[&str]) {
        self.ensure_search_stopped();

        let params = self.parse_go_params(parts);
        
        let position = match self.position.read() {
            Ok(pos) => pos.clone(),
            Err(_) => {
                self.debug("Failed to read position");
                return;
            }
        };

        let stop_flag = Arc::clone(&self.stop_flag);
        let searching = Arc::clone(&self.searching);
        let pondering = Arc::clone(&self.pondering);
        let parallel_search = Arc::clone(&self.parallel_search);
        let debug_mode = self.debug_mode.load(Ordering::Relaxed);
        
        self.stop_flag.store(false, Ordering::SeqCst);
        self.searching.store(true, Ordering::SeqCst);
        self.pondering.store(params.ponder, Ordering::SeqCst);
        
        self.debug(&format!("Starting search: depth={}, time={}ms, threads={}", 
                           params.depth, params.time_mgr.allocated_ms(), params.thread_count));
        
        let thread_handle = thread::spawn(move || {
            let result = {
                let mut search = match parallel_search.lock() {
                    Ok(s) => s,
                    Err(_) => {
                        if debug_mode {
                            println!("info string Failed to acquire search lock");
                        }
                        return;
                    }
                };
                
                search.stop_flag = Arc::clone(&stop_flag);
                search.search(&position, params.depth, params.time_mgr)
            };
            
            if !stop_flag.load(Ordering::SeqCst) || result.best_move != Move::null() {
                Self::output_bestmove(&result, params.ponder && result.pv.len() > 1);
            }
            
            searching.store(false, Ordering::SeqCst);
            pondering.store(false, Ordering::SeqCst);
            
            if debug_mode {
                println!("info string Search completed");
            }
        });
        
        if let Ok(mut thread_guard) = self.search_thread.lock() {
            *thread_guard = Some(thread_handle);
        }
    }

    fn stop_command(&mut self) {
        self.debug("Stop command received");
        self.stop_flag.store(true, Ordering::SeqCst);
        
        thread::sleep(Duration::from_millis(50));
    }

    fn ponderhit_command(&mut self) {
        self.pondering.store(false, Ordering::Relaxed);
        self.debug("Ponderhit - continuing search");
    }

    fn quit_command(&mut self) {
        self.debug("Quit command received");
        self.ensure_search_stopped();
        std::process::exit(0);
    }

    fn display_command(&self) {
        if let Ok(pos) = self.position.read() {
            pos.print();
            println!("FEN: {}", pos.to_fen());
            println!("Hash: {:016X}", pos.hash);
        }
    }

    fn eval_command(&self) {
        if let Ok(pos) = self.position.read() {
            crate::eval::evaluate::print_evaluation(&pos);
            let moves = generate_legal_moves(&pos);
    let mut counter = 0;
    for move1 in moves{
        counter += 1;
        println!("Move: {:?}",move_to_uci(move1));
    }
    println!("Move Count: {:?}",counter);
        }
    }


    fn ensure_search_stopped(&mut self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        
        let thread_handle = {
            if let Ok(mut guard) = self.search_thread.lock() {
                guard.take()
            } else {
                None
            }
        };
        
        if let Some(handle) = thread_handle {
            self.debug("Waiting for search thread to complete...");
            
            let start = Instant::now();
            while self.searching.load(Ordering::SeqCst) && start.elapsed() < Duration::from_secs(2) {
                thread::sleep(Duration::from_millis(10));
            }
            
            match handle.join() {
                Ok(_) => self.debug("Search thread joined successfully"),
                Err(_) => self.debug("Search thread panicked"),
            }
        }
        
        self.searching.store(false, Ordering::SeqCst);
        self.pondering.store(false, Ordering::SeqCst);
    }

    fn parse_move(&self, pos: &Position, move_str: &str) -> Option<Move> {
        if move_str.len() < 4 {
            return None;
        }

        let from_str = &move_str[0..2];
        let to_str = &move_str[2..4];

        let from = crate::board::bitboard::algebraic_to_square(from_str)?;
        let to = crate::board::bitboard::algebraic_to_square(to_str)?;

        let promotion_char = if move_str.len() > 4 {
            move_str.chars().nth(4)
        } else {
            None
        };

        let legal_moves = generate_legal_moves(pos);

        for &mv in &legal_moves {
            if mv.from() == from && mv.to() == to {
                if mv.is_promotion() {
                    if let Some(promo_char) = promotion_char {
                        let promo_type = match promo_char.to_ascii_lowercase() {
                            'q' => PieceType::Queen,
                            'r' => PieceType::Rook,
                            'b' => PieceType::Bishop,
                            'n' => PieceType::Knight,
                            _ => continue,
                        };
                        if mv.promotion() == promo_type {
                            return Some(mv);
                        }
                    }
                } else if promotion_char.is_none() {
                    return Some(mv);
                }
            }
        }

        None
    }

    fn parse_go_params(&self, parts: &[&str]) -> GoParams {
        let mut params = GoParams::default();
        
        let position = if let Ok(pos) = self.position.read() {
            pos.clone()
        } else {
            Position::startpos()
        };
        
        let is_white_to_move = position.side_to_move == Color::White;
        
        let mut i = 1;
        while i < parts.len() {
            match parts[i] {
                "wtime" => {
                    if i + 1 < parts.len() {
                        params.wtime = parts[i + 1].parse().ok();
                        i += 1;
                    }
                }
                "btime" => {
                    if i + 1 < parts.len() {
                        params.btime = parts[i + 1].parse().ok();
                        i += 1;
                    }
                }
                "winc" => {
                    if i + 1 < parts.len() {
                        params.winc = parts[i + 1].parse().ok();
                        i += 1;
                    }
                }
                "binc" => {
                    if i + 1 < parts.len() {
                        params.binc = parts[i + 1].parse().ok();
                        i += 1;
                    }
                }
                "movestogo" => {
                    if i + 1 < parts.len() {
                        params.movestogo = parts[i + 1].parse().ok();
                        i += 1;
                    }
                }
                "depth" => {
                    if i + 1 < parts.len() {
                        if let Ok(d) = parts[i + 1].parse::<u8>() {
                            params.depth = d.min(128) as i32;
                        }
                        i += 1;
                    }
                }
                "nodes" => {
                    if i + 1 < parts.len() {
                        params.nodes = parts[i + 1].parse().ok();
                        i += 1;
                    }
                }
                "movetime" => {
                    if i + 1 < parts.len() {
                        params.movetime = parts[i + 1].parse().ok();
                        i += 1;
                    }
                }
                "infinite" => params.infinite = true,
                "ponder" => params.ponder = true,
                _ => {}
            }
            i += 1;
        }

        if let Some(mt) = params.movetime {
            params.movetime = Some(mt.saturating_sub(self.move_overhead as u32));
        }
        if let Some(wt) = params.wtime {
            params.wtime = Some(wt.saturating_sub(self.move_overhead as u32));
        }
        if let Some(bt) = params.btime {
            params.btime = Some(bt.saturating_sub(self.move_overhead as u32));
        }

        params.time_mgr = TimeManager::new_with_increment(
            params.wtime,
            params.btime,
            params.movetime,
            params.winc,
            params.binc,
            params.movestogo,
            params.nodes,
            params.infinite,
            is_white_to_move,
        );

        params.thread_count = self.thread_count;
        params
    }

    fn output_bestmove(result: &SearchResult, with_ponder: bool) {
        if result.best_move != Move::null() {
            let move_str = Self::move_to_string(result.best_move);
            
            let ponder_str = if with_ponder && result.pv.len() > 1 {
                format!(" ponder {}", Self::move_to_string(result.pv[1]))
            } else {
                String::new()
            };
            
            println!("bestmove {}{}", move_str, ponder_str);
        } else {
            println!("bestmove 0000");
        }
        io::stdout().flush().unwrap();
    }

    fn move_to_string(mv: Move) -> String {
        let mut result = format!("{}{}", 
            square_to_algebraic(mv.from()),
            square_to_algebraic(mv.to())
        );
        
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

    fn debug(&self, message: &str) {
        if self.debug_mode.load(Ordering::Relaxed) {
            println!("info string {}", message);
            io::stdout().flush().unwrap();
        }
    }
}


struct GoParams {
    wtime: Option<u32>,
    btime: Option<u32>,
    winc: Option<u32>,
    binc: Option<u32>,
    movestogo: Option<u32>,
    depth: i32,
    nodes: Option<u64>,
    movetime: Option<u32>,
    infinite: bool,
    ponder: bool,
    time_mgr: TimeManager,
    thread_count: usize,
}

impl Default for GoParams {
    fn default() -> Self {
        Self {
            wtime: None,
            btime: None,
            winc: None,
            binc: None,
            movestogo: None,
            depth: 128,
            nodes: None,
            movetime: None,
            infinite: false,
            ponder: false,
            time_mgr: TimeManager::infinite(),
            thread_count: 1,
        }
    }
}

pub fn run_uci() {
    let mut engine = ImprovedUciEngine::new();
    engine.run();
}