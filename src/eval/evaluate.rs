use crate::board::position::{Position, Color, PieceType};
use crate::eval::material::{calculate_phase};
use crate::eval::{
    material, pst, pawns, mobility, space, king_safety, 
    imbalance, threats, eval_util, weights, initiative
};

use std::sync::Once;

const MAX_PHASE: u32 = 256;

const EVAL_CACHE_SIZE: usize = 16384;
static mut EVAL_CACHE: [EvalCacheEntry; EVAL_CACHE_SIZE] = [EvalCacheEntry::default(); EVAL_CACHE_SIZE];
static EVAL_CACHE_INIT: Once = Once::new();

#[derive(Debug, Clone, Copy)]
struct EvalCacheEntry {
    hash: u64,
    score: i32,
    depth: u8,
}

impl EvalCacheEntry {
    const fn default() -> Self {
        Self {
            hash: 0,
            score: 0,
            depth: 0,
        }
    }
}

fn init_eval_cache() {
    EVAL_CACHE_INIT.call_once(|| {
        unsafe {
            for i in 0..EVAL_CACHE_SIZE {
                EVAL_CACHE[i] = EvalCacheEntry::default();
            }
        }
    });
}

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

#[derive(Debug, Clone)]
pub struct DetailedEvaluation {
    pub white: DetailedScores,
    pub black: DetailedScores,
    pub phase: u32,
    pub total: Score,
}

#[derive(Debug, Clone)]
pub struct DetailedScores {
    pub material: Score,
    pub imbalance: Score,
    pub initiative: Score,
    pub pawns: Score,
    pub knights: Score,
    pub bishops: Score,
    pub rooks: Score,
    pub queens: Score,
    pub mobility: Score,
    pub king_safety: Score,
    pub threats: Score,
    pub passed: Score,
    pub space: Score,
}

impl DetailedScores {
    pub fn new() -> Self {
        Self {
            material: Score::zero(),
            imbalance: Score::zero(),
            initiative: Score::zero(),
            pawns: Score::zero(),
            knights: Score::zero(),
            bishops: Score::zero(),
            rooks: Score::zero(),
            queens: Score::zero(),
            mobility: Score::zero(),
            king_safety: Score::zero(),
            threats: Score::zero(),
            passed: Score::zero(),
            space: Score::zero(),
        }
    }
    
    pub fn total(&self) -> Score {
        self.material
            .add(self.imbalance)
            .add(self.initiative)
            .add(self.pawns)
            .add(self.knights)
            .add(self.bishops)
            .add(self.rooks)
            .add(self.queens)
            .add(self.mobility)
            .add(self.king_safety)
            .add(self.threats)
            .add(self.passed)
            .add(self.space)
    }
}

#[inline(always)]
pub fn evaluate_int(pos: &Position) -> i32 {
    if pos.halfmove_clock >= 100 || material::is_material_draw(pos) {
        return DRAW_VALUE;
    }
    
    let cache_index = (pos.hash as usize) & (EVAL_CACHE_SIZE - 1);
    unsafe {
        let entry = &EVAL_CACHE[cache_index];
        if entry.hash == pos.hash && entry.depth >= 1 {
            return entry.score;
        }
    }
    
    let phase = calculate_phase(pos);
    let w = weights::get_weights();
    
    let mut score = Score::zero();
    score = score.add(weights::apply_weight(material::evaluate_material(pos), w.material_weight));
    score = score.add(weights::apply_weight(pst::evaluate_pst(pos), w.pst_weight));
    
    let material_score = score.interpolate(phase);
    let color = pos.side_to_move;
     let our_pawns = pos.pieces_colored(PieceType::Pawn, color);
    let enemy_pawns = pos.pieces_colored(PieceType::Pawn, color.opposite());
    
    if material_score.abs() > 800 {
        score = score.add(weights::apply_weight(pawns::    evaluate_pawn_structure(pos, color, our_pawns, enemy_pawns)
, w.pawn_structure_weight));
        let final_score = finalize_evaluation(score.interpolate(phase), pos);
        
        unsafe {
            EVAL_CACHE[cache_index] = EvalCacheEntry {
                hash: pos.hash,
                score: final_score,
                depth: 1,
            };
        }
        return final_score;
    }
    
    
    score = score.add(weights::apply_weight(pawns::    evaluate_pawn_structure(pos, color, our_pawns, enemy_pawns)
, w.pawn_structure_weight));
    
    score = score.add(evaluate_pieces_optimized(pos));
    
    score = score.add(weights::apply_weight(mobility::evaluate_mobility(pos), w.mobility_weight));
    
    if phase > 32 {
        score = score.add(weights::apply_weight(king_safety::evaluate_king_safety(pos), w.king_safety_weight));
    }
    
    if is_likely_tactical(pos, phase) {
        score = score.add(weights::apply_weight(threats::evaluate_threats(pos), w.threats_weight));
    } else {
        score = score.add(weights::apply_weight(evaluate_simple_threats(pos), w.threats_weight));
    }
    
    if phase > 100 && pos.all_pieces().count_ones() > 20 {
        score = score.add(weights::apply_weight(space::evaluate_space(pos), w.space_weight));
    }
    
    if should_evaluate_imbalance(pos) {
        score = score.add(weights::apply_weight(imbalance::evaluate_imbalance(pos), w.imbalance_weight));
    }
    
    score = score.add(weights::apply_weight(initiative::evaluate_initiative_simple(pos), w.tempo_bonus_weight));
    
    let final_score = finalize_evaluation(score.interpolate(phase), pos);
    
    unsafe {
        EVAL_CACHE[cache_index] = EvalCacheEntry {
            hash: pos.hash,
            score: final_score,
            depth: 2,
        };
    }
    
    final_score
}

#[inline(always)]
pub fn evaluate(pos: &Position) -> EvalResult {
    let score = evaluate_int(pos);
    EvalResult::new(score)
}

#[inline(always)]
fn evaluate_pieces_optimized(pos: &Position) -> Score {
    use crate::movegen::magic::{get_knight_attacks, get_bishop_attacks, get_rook_attacks, get_queen_attacks};
    
    let mut score = Score::zero();
    let all_pieces = pos.all_pieces();
    
    for color in [Color::White, Color::Black] {
        let sign = if color == Color::White { 1 } else { -1 };
        let our_pieces = pos.pieces(color);
        let our_pawns = pos.pieces_colored(PieceType::Pawn, color);
        let enemy_pawns = pos.pieces_colored(PieceType::Pawn, color.opposite());
        
        let mut knights = pos.pieces_colored(PieceType::Knight, color);
        while knights != 0 {
            let sq = knights.trailing_zeros() as u8;
            knights &= knights - 1;
            
            if is_outpost_square(pos, sq, color) {
                score = score.add(Score::new(sign * 25, sign * 20));
            }
            
            let file = sq % 8;
            let rank = sq / 8;
            if file == 0 || file == 7 || rank == 0 || rank == 7 {
                score = score.add(Score::new(sign * -10, sign * -15));
            }
        }
        
        let bishops = pos.pieces_colored(PieceType::Bishop, color);
        if bishops.count_ones() >= 2 {
            score = score.add(Score::new(sign * 50, sign * 60));
        }
        
        let mut bishops_bb = bishops;
        while bishops_bb != 0 {
            let sq = bishops_bb.trailing_zeros() as u8;
            bishops_bb &= bishops_bb - 1;
            
            let square_color = ((sq / 8 + sq % 8) % 2) as u64;
            let pawns_on_color = our_pawns & (if square_color == 0 { 
                0xAA55AA55AA55AA55u64 
            } else { 
                0x55AA55AA55AA55AAu64 
            });
            
            let blocked_pawns = pawns_on_color.count_ones() as i32;
            score = score.add(Score::new(sign * blocked_pawns * -2, sign * blocked_pawns * -4));
        }
        
        let mut rooks = pos.pieces_colored(PieceType::Rook, color);
        while rooks != 0 {
            let sq = rooks.trailing_zeros() as u8;
            rooks &= rooks - 1;
            
            let file = sq % 8;
            let file_mask = 0x0101010101010101u64 << file;
            
            if (our_pawns & file_mask) == 0 {
                if (enemy_pawns & file_mask) == 0 {
                    score = score.add(Score::new(sign * 35, sign * 20));
                } else {
                    score = score.add(Score::new(sign * 15, sign * 8));
                }
            }
            
            let rank = sq / 8;
            let seventh_rank = if color == Color::White { 6 } else { 1 };
            if rank == seventh_rank {
                score = score.add(Score::new(sign * 30, sign * 25));
            }
        }
        
        if rooks.count_ones() >= 2 {
            score = score.add(Score::new(sign * 15, sign * 10));
        }
        
        let phase = calculate_phase(pos);
        if phase > 200 {
            let queens = pos.pieces_colored(PieceType::Queen, color);
            if queens != 0 {
                let queen_sq = queens.trailing_zeros() as u8;
                let starting_square = if color == Color::White { 3 } else { 59 };
                
                if queen_sq != starting_square {
                    let rank = queen_sq / 8;
                    let development_rank = if color == Color::White { rank } else { 7 - rank };
                    
                    if development_rank > 1 {
                        score = score.add(Score::new(sign * -10, 0));
                    }
                }
            }
        }
    }
    
    score
}

#[inline(always)]
fn evaluate_simple_threats(pos: &Position) -> Score {
    let mut score = Score::zero();
    
    for color in [Color::White, Color::Black] {
        let sign = if color == Color::White { 1 } else { -1 };
        let our_pawns = pos.pieces_colored(PieceType::Pawn, color);
        let enemy_pieces = pos.pieces(color.opposite());
        let enemy_valuable = pos.pieces_colored(PieceType::Queen, color.opposite()) |
                           pos.pieces_colored(PieceType::Rook, color.opposite());
        
        let pawn_attacks = if color == Color::White {
            let left = (our_pawns & !0x0101010101010101u64) << 7;
            let right = (our_pawns & !0x8080808080808080u64) << 9;
            left | right
        } else {
            let left = (our_pawns & !0x8080808080808080u64) >> 7;
            let right = (our_pawns & !0x0101010101010101u64) >> 9;
            left | right
        };
        
        let attacked_valuable = (pawn_attacks & enemy_valuable).count_ones() as i32;
        score = score.add(Score::new(sign * attacked_valuable * 25, sign * attacked_valuable * 30));
        
        let attacked_pieces = (pawn_attacks & enemy_pieces).count_ones() as i32;
        score = score.add(Score::new(sign * attacked_pieces * 5, sign * attacked_pieces * 8));
    }
    
    score
}

#[inline(always)]
fn is_likely_tactical(pos: &Position, phase: u32) -> bool {
    if phase < 100 {
        return true;
    }
    
    let white_pawns = pos.pieces_colored(PieceType::Pawn, Color::White);
    let black_pawns = pos.pieces_colored(PieceType::Pawn, Color::Black);
    
    let white_advanced = white_pawns & 0x00FF000000000000u64;
    let black_advanced = black_pawns & 0x000000000000FF00u64;
    
    if white_advanced != 0 || black_advanced != 0 {
        return true;
    }
    
    pos.halfmove_clock == 0
}

#[inline(always)]
fn should_evaluate_imbalance(pos: &Position) -> bool {
    let total_pieces = pos.all_pieces().count_ones();
    
    if total_pieces < 8 || total_pieces > 30 {
        return false;
    }
    
    let white_minors = pos.pieces_colored(PieceType::Knight, Color::White).count_ones() +
                       pos.pieces_colored(PieceType::Bishop, Color::White).count_ones();
    let black_minors = pos.pieces_colored(PieceType::Knight, Color::Black).count_ones() +
                       pos.pieces_colored(PieceType::Bishop, Color::Black).count_ones();
    
    let white_majors = pos.pieces_colored(PieceType::Rook, Color::White).count_ones() +
                       pos.pieces_colored(PieceType::Queen, Color::White).count_ones();
    let black_majors = pos.pieces_colored(PieceType::Rook, Color::Black).count_ones() +
                       pos.pieces_colored(PieceType::Queen, Color::Black).count_ones();
    
    (white_minors as i32 - black_minors as i32).abs() > 1 ||
    (white_majors as i32 - black_majors as i32).abs() > 0
}

#[inline(always)]
fn is_outpost_square(pos: &Position, square: u8, color: Color) -> bool {
    let file = square % 8;
    let rank = square / 8;
    
    let in_enemy_territory = match color {
        Color::White => rank >= 4,
        Color::Black => rank <= 3,
    };
    
    if !in_enemy_territory {
        return false;
    }
    
    let enemy_pawns = pos.pieces_colored(PieceType::Pawn, color.opposite());
    
    let mut can_be_attacked = false;
    
    if file > 0 {
        let left_file_mask = 0x0101010101010101u64 << (file - 1);
        let left_pawns = enemy_pawns & left_file_mask;
        
        if color == Color::White {
            can_be_attacked |= (left_pawns & ((1u64 << square) >> 9)) != 0;
        } else {
            can_be_attacked |= (left_pawns & ((1u64 << square) << 7)) != 0;
        }
    }
    
    if file < 7 {
        let right_file_mask = 0x0101010101010101u64 << (file + 1);
        let right_pawns = enemy_pawns & right_file_mask;
        
        if color == Color::White {
            can_be_attacked |= (right_pawns & ((1u64 << square) >> 7)) != 0;
        } else {
            can_be_attacked |= (right_pawns & ((1u64 << square) << 9)) != 0;
        }
    }
    
    !can_be_attacked
}

pub fn evaluate_detailed(pos: &Position) -> DetailedEvaluation {
    evaluate_detailed_internal(pos)
}

fn evaluate_detailed_internal(pos: &Position) -> DetailedEvaluation {
    let mut white_scores = DetailedScores::new();
    let mut black_scores = DetailedScores::new();
    
    let phase = calculate_phase(pos);
    let w = weights::get_weights();
    
    white_scores.material = evaluate_material_for_color(pos, Color::White);
    black_scores.material = evaluate_material_for_color(pos, Color::Black);
    
    let imbalance = weights::apply_weight(imbalance::evaluate_imbalance(pos), w.imbalance_weight);
    if imbalance.mg >= 0 && imbalance.eg >= 0 {
        white_scores.imbalance = imbalance;
        black_scores.imbalance = Score::zero();
    } else {
        white_scores.imbalance = Score::zero();
        black_scores.imbalance = imbalance.neg();
    }
    
    white_scores.initiative = weights::apply_weight(
        initiative::evaluate_initiative(pos, Color::White), 
        w.tempo_bonus_weight
    );
    black_scores.initiative = weights::apply_weight(
        initiative::evaluate_initiative(pos, Color::Black), 
        w.tempo_bonus_weight
    );
    
    white_scores.pawns = weights::apply_weight(
        pawns::evaluate_pawn_structure_for_color(pos, Color::White),
        w.pawn_structure_weight
    );
    black_scores.pawns = weights::apply_weight(
        pawns::evaluate_pawn_structure_for_color(pos, Color::Black),
        w.pawn_structure_weight
    );
    
    white_scores.mobility = weights::apply_weight(
        mobility::evaluate_mobility_for_color(pos, Color::White),
        w.mobility_weight
    );
    black_scores.mobility = weights::apply_weight(
        mobility::evaluate_mobility_for_color(pos, Color::Black),
        w.mobility_weight
    );
    
    white_scores.king_safety = weights::apply_weight(
        king_safety::evaluate_king_safety_for_color(pos, Color::White),
        w.king_safety_weight
    );
    black_scores.king_safety = weights::apply_weight(
        king_safety::evaluate_king_safety_for_color(pos, Color::Black),
        w.king_safety_weight
    );
    
    white_scores.threats = weights::apply_weight(
        threats::evaluate_threats_for_color(pos, Color::White),
        w.threats_weight
    );
    black_scores.threats = weights::apply_weight(
        threats::evaluate_threats_for_color(pos, Color::Black),
        w.threats_weight
    );
    
    white_scores.space = weights::apply_weight(
        space::evaluate_space_for_color(pos, Color::White),
        w.space_weight
    );
    black_scores.space = weights::apply_weight(
        space::evaluate_space_for_color(pos, Color::Black),
        w.space_weight
    );
    
    let total = white_scores.total().sub(black_scores.total());
    
    DetailedEvaluation {
        white: white_scores,
        black: black_scores,
        phase,
        total,
    }
}

fn evaluate_material_for_color(pos: &Position, color: Color) -> Score {
    let mut score = Score::zero();
    
    score = score.add(Score::new(
        pos.piece_count(color, PieceType::Pawn) as i32 * material::PIECE_VALUES[1].mg,
        pos.piece_count(color, PieceType::Pawn) as i32 * material::PIECE_VALUES[1].eg,
    ));
    
    score = score.add(Score::new(
        pos.piece_count(color, PieceType::Knight) as i32 * material::PIECE_VALUES[2].mg,
        pos.piece_count(color, PieceType::Knight) as i32 * material::PIECE_VALUES[2].eg,
    ));
    
    score = score.add(Score::new(
        pos.piece_count(color, PieceType::Bishop) as i32 * material::PIECE_VALUES[3].mg,
        pos.piece_count(color, PieceType::Bishop) as i32 * material::PIECE_VALUES[3].eg,
    ));
    
    score = score.add(Score::new(
        pos.piece_count(color, PieceType::Rook) as i32 * material::PIECE_VALUES[4].mg,
        pos.piece_count(color, PieceType::Rook) as i32 * material::PIECE_VALUES[4].eg,
    ));
    
    score = score.add(Score::new(
        pos.piece_count(color, PieceType::Queen) as i32 * material::PIECE_VALUES[5].mg,
        pos.piece_count(color, PieceType::Queen) as i32 * material::PIECE_VALUES[5].eg,
    ));
    
    score
}

pub fn print_evaluation_table(pos: &Position) {
    let eval = evaluate_detailed(pos);
    
    println!("     Term    |    White    |    Black    |    Total");
    println!("             |   MG    EG  |   MG    EG  |   MG    EG");
    println!(" ------------+-------------+-------------+------------");
    
    print_eval_row("Material", eval.white.material, eval.black.material);
    print_eval_row("Imbalance", eval.white.imbalance, eval.black.imbalance);
    print_eval_row("Initiative", eval.white.initiative, eval.black.initiative);
    print_eval_row("Pawns", eval.white.pawns, eval.black.pawns);
    print_eval_row("Knights", eval.white.knights, eval.black.knights);
    print_eval_row("Bishops", eval.white.bishops, eval.black.bishops);
    print_eval_row("Rooks", eval.white.rooks, eval.black.rooks);
    print_eval_row("Queens", eval.white.queens, eval.black.queens);
    print_eval_row("Mobility", eval.white.mobility, eval.black.mobility);
    print_eval_row("King safety", eval.white.king_safety, eval.black.king_safety);
    print_eval_row("Threats", eval.white.threats, eval.black.threats);
    print_eval_row("Passed", eval.white.passed, eval.black.passed);
    print_eval_row("Space", eval.white.space, eval.black.space);
    
    println!(" ------------+-------------+-------------+------------");
    
    let white_total = eval.white.total();
    let black_total = eval.black.total();
    let total = white_total.sub(black_total);
    
    println!("       Total | {:5.2} {:5.2} | {:5.2} {:5.2} | {:5.2} {:5.2}",
             white_total.mg as f32 / 100.0, white_total.eg as f32 / 100.0,
             black_total.mg as f32 / 100.0, black_total.eg as f32 / 100.0,
             total.mg as f32 / 100.0, total.eg as f32 / 100.0);
    
    let final_eval = total.interpolate(eval.phase);
    let adjusted_eval = if pos.side_to_move == Color::Black { -final_eval } else { final_eval };
    
    println!("\nTotal evaluation: {:.2} (white side)", adjusted_eval as f32 / 100.0);
}

fn print_eval_row(name: &str, white: Score, black: Score) {
    let diff = white.sub(black);
    
    println!("{:>12} | {:5.2} {:5.2} | {:5.2} {:5.2} | {:5.2} {:5.2}",
             name,
             white.mg as f32 / 100.0, white.eg as f32 / 100.0,
             black.mg as f32 / 100.0, black.eg as f32 / 100.0,
             diff.mg as f32 / 100.0, diff.eg as f32 / 100.0);
}

#[inline(always)]
fn finalize_evaluation(mut score: i32, pos: &Position) -> i32 {
    let w = weights::get_weights();
    score += weights::apply_weight_i32(TEMPO_BONUS, w.tempo_bonus_weight);
    
    score = eval_util::apply_contempt(score, pos);
    score = eval_util::scale_evaluation(score, pos);
    
    if pos.side_to_move == Color::Black {
        score = -score;
    }
    
    eval_util::normalize_score(score)
}

pub fn evaluate_for_debug(pos: &Position) -> EvalResult {
    let score = evaluate_int(pos);
    EvalResult::new(score)
}

pub fn evaluate_fast(pos: &Position) -> i32 {
    let cache_index = (pos.hash as usize) & (EVAL_CACHE_SIZE - 1);
    unsafe {
        let entry = &EVAL_CACHE[cache_index];
        if entry.hash == pos.hash {
            return entry.score;
        }
    }
    
    let mut score = Score::zero();
    let w = weights::get_weights();
    
    score = score.add(weights::apply_weight(material::evaluate_material(pos), w.material_weight));
    score = score.add(weights::apply_weight(pst::evaluate_pst(pos), w.pst_weight));
    
    let phase = calculate_phase(pos);
    let mut final_score = score.interpolate(phase);
    
    final_score += weights::apply_weight_i32(TEMPO_BONUS, w.tempo_bonus_weight);
    if pos.side_to_move == Color::Black {
        final_score = -final_score;
    }
    
    let result = eval_util::normalize_score(final_score);
    
    unsafe {
        EVAL_CACHE[cache_index] = EvalCacheEntry {
            hash: pos.hash,
            score: result,
            depth: 0,
        };
    }
    
    result
}

pub fn evaluate_lazy(pos: &Position, alpha: i32, beta: i32) -> Option<i32> {
    const LAZY_MARGIN: i32 = 250;
    
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
    let eval = evaluate_detailed(pos);
    let phase = eval.phase;
    
    [
        eval.white.material.sub(eval.black.material).interpolate(phase),
        eval.white.pawns.sub(eval.black.pawns).interpolate(phase),
        eval.white.knights.sub(eval.black.knights).interpolate(phase),
        eval.white.bishops.sub(eval.black.bishops).interpolate(phase),
        eval.white.rooks.sub(eval.black.rooks).interpolate(phase),
        eval.white.queens.sub(eval.black.queens).interpolate(phase),
        eval.white.king_safety.sub(eval.black.king_safety).interpolate(phase),
        eval.white.mobility.sub(eval.black.mobility).interpolate(phase),
    ]
}

pub fn evaluate_complexity(pos: &Position) -> i32 {
    eval_util::calculate_complexity(pos)
}

pub fn needs_careful_evaluation(pos: &Position) -> bool {
    if pos.halfmove_clock == 0 {
        return true;
    }
    
    let white_pawns = pos.pieces_colored(PieceType::Pawn, Color::White);
    let black_pawns = pos.pieces_colored(PieceType::Pawn, Color::Black);
    
    let white_7th = white_pawns & 0x00FF000000000000u64;
    let black_2nd = black_pawns & 0x000000000000FF00u64;
    
    if white_7th != 0 || black_2nd != 0 {
        return true;
    }
    
    use crate::movegen::magic::all_attacks;
    let white_attacks = all_attacks(pos, Color::White);
    let black_attacks = all_attacks(pos, Color::Black);
    
    for color in [Color::White, Color::Black] {
        let our_attacks = if color == Color::White { white_attacks } else { black_attacks };
        let enemy_attacks = if color == Color::White { black_attacks } else { white_attacks };
        
        let valuable = pos.pieces_colored(PieceType::Queen, color) |
                      pos.pieces_colored(PieceType::Rook, color);
        
        let undefended_valuable = valuable & !our_attacks & enemy_attacks;
        if undefended_valuable != 0 {
            return true;
        }
    }
    
    false
}

pub fn static_exchange_estimate(pos: &Position, square: u8) -> i32 {
    let (piece_type, piece_color) = pos.piece_at(square);
    if piece_type == PieceType::None {
        return 0;
    }
    
    let piece_value = material::PIECE_VALUES[piece_type as usize];
    if piece_color == Color::White {
        piece_value.mg
    } else {
        -piece_value.mg
    }
}

pub fn tactical_complexity(pos: &Position) -> i32 {
    let mut complexity = 0;
    
    let white_pawns = pos.pieces_colored(PieceType::Pawn, Color::White);
    let black_pawns = pos.pieces_colored(PieceType::Pawn, Color::Black);
    
    let white_advanced = (white_pawns & 0x00FF000000000000u64).count_ones() as i32;
    let black_advanced = (black_pawns & 0x000000000000FF00u64).count_ones() as i32;
    
    complexity += white_advanced * 3 + black_advanced * 3;
    
    if pos.pieces_colored(PieceType::Queen, Color::White) != 0 {
        complexity += 5;
    }
    if pos.pieces_colored(PieceType::Queen, Color::Black) != 0 {
        complexity += 5;
    }
    
    if king_safety::is_king_in_danger(pos, Color::White) {
        complexity += 8;
    }
    if king_safety::is_king_in_danger(pos, Color::Black) {
        complexity += 8;
    }
    
    complexity
}

pub fn print_evaluation(pos: &Position) {
    print_evaluation_table(pos);
}

pub fn init_eval() {
    init_eval_cache();
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
        init_eval();
        let pos = Position::startpos();
        let eval_result = evaluate(&pos);
        
        assert!(eval_result.score.abs() < 100);
        assert!(!eval_result.is_mate);
    }
    
    #[test]
    fn test_fast_evaluation() {
        init_eval();
        let pos = Position::startpos();
        let fast_eval = evaluate_fast(&pos);
        let full_eval = evaluate(&pos);
        
        assert!((fast_eval - full_eval.score).abs() < 200);
    }
    
    #[test]
    fn test_lazy_evaluation() {
        init_eval();
        let pos = Position::startpos();
        let lazy_result = evaluate_lazy(&pos, -1000, 1000);
        
        assert!(lazy_result.is_none());
    }
    
    #[test]
    fn test_eval_cache() {
        init_eval();
        let pos = Position::startpos();
        
        let eval1 = evaluate_int(&pos);
        
        let eval2 = evaluate_int(&pos);
        
        assert_eq!(eval1, eval2);
    }
}
