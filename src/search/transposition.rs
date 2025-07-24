use std::sync::atomic::{AtomicU64, Ordering};
use std::mem;
use crate::board::position::{Position, Move,MoveType,PieceType};
use crate::eval::evaluate::EvalResult;

const SCORE_NONE: i32 = -32768;
const SCORE_MATE: i32 = 31000;
const MAX_PLY: i32 = 128;

const BOUND_NONE: u8 = 0;
const BOUND_EXACT: u8 = 1;
const BOUND_LOWER: u8 = 2;
const BOUND_UPPER: u8 = 3;

#[repr(align(16))]
pub struct TTEntry {
    key_xor: AtomicU64,
    data: AtomicU64,
}

impl TTEntry {
    const fn new() -> Self {
        Self {
            key_xor: AtomicU64::new(0),
            data: AtomicU64::new(0),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TTData {
    pub best_move: Move,
    pub score: i32,
    pub static_eval: i32,
    pub depth: u8,
    pub bound: u8,
    pub age: u8,
}

impl TTData {
    fn pack(&self) -> u64 {
        let mut packed = 0u64;
        
        packed |= self.best_move.as_u16() as u64;
        
        let score_bits = if self.score.abs() >= SCORE_MATE - MAX_PLY {
            (self.score + 32768) as u16
        } else {
            (self.score.max(-32767).min(32767) + 32768) as u16
        };
        packed |= (score_bits as u64) << 16;
        
        let eval_bits = (self.static_eval.max(-32767).min(32767) + 32768) as u16;
        packed |= (eval_bits as u64) << 32;
        
        packed |= (self.depth as u64) << 48;
        
        packed |= ((self.bound & 0x3) as u64) << 56;
        
        packed |= ((self.age & 0x3F) as u64) << 58;
        
        packed
    }
    
    fn unpack(packed: u64) -> Self {
        let best_move = Move::from_u16(packed as u16);
        
        let score_bits = ((packed >> 16) & 0xFFFF) as u16;
        let score = (score_bits as i32) - 32768;
        
        let eval_bits = ((packed >> 32) & 0xFFFF) as u16;
        let static_eval = (eval_bits as i32) - 32768;
        
        let depth = ((packed >> 48) & 0xFF) as u8;
        
        let bound = ((packed >> 56) & 0x3) as u8;
        
        let age = ((packed >> 58) & 0x3F) as u8;
        
        Self {
            best_move,
            score,
            static_eval,
            depth,
            bound,
            age,
        }
    }
}

pub struct TranspositionTable {
    table: Box<[TTEntry]>,
    size: usize,
    size_mask: usize,
    age: AtomicU64,
}

impl TranspositionTable {
    pub fn new(size_mb: usize) -> Self {
        let size_bytes = size_mb * 1024 * 1024;
        let entry_size = mem::size_of::<TTEntry>();
        let num_entries = size_bytes / entry_size;
        
        let size = num_entries.next_power_of_two() / 2;
        let size_mask = size - 1;
        
        let mut table = Vec::with_capacity(size);
        for _ in 0..size {
            table.push(TTEntry::new());
        }
        
        Self {
            table: table.into_boxed_slice(),
            size,
            size_mask,
            age: AtomicU64::new(0),
        }
    }
    
    #[inline(always)]
    pub fn probe(&self, hash: u64) -> Option<TTData> {
        let idx = (hash as usize) & self.size_mask;
        let entry = &self.table[idx];
        
        let key_xor = entry.key_xor.load(Ordering::Relaxed);
        let data = entry.data.load(Ordering::Relaxed);
        
        if (key_xor ^ data) == hash {
            Some(TTData::unpack(data))
        } else {
            None
        }
    }
    
    #[inline(always)]
    fn should_replace(&self, entry: &TTEntry, new_depth: u8, new_age: u8, new_bound: u8) -> bool {
        let key_xor = entry.key_xor.load(Ordering::Relaxed);
        let data = entry.data.load(Ordering::Relaxed);
        
        if key_xor == 0 {
            return true;
        }
        
        if key_xor ^ data == 0 {
            return true;
        }
        
        let existing = TTData::unpack(data);
        let current_age = self.get_age() as u8;
        
        let age_diff = current_age.wrapping_sub(existing.age) & 0x3F;
        if age_diff > 3 {
            return true;
        }
        
        let depth_advantage = new_depth as i32 - existing.depth as i32;
        
        if depth_advantage >= 4 {
            return true;
        }
        
        let bound_advantage = if new_bound == BOUND_EXACT {
            if existing.bound == BOUND_EXACT { 0 } else { 2 }
        } else {
            0
        };
        
        let replacement_score = depth_advantage + bound_advantage + (age_diff as i32);
        replacement_score >= 0
    }

    #[inline(always)]
    pub fn store(
        &self,
        hash: u64,
        best_move: Move,
        score: i32,
        static_eval: i32,
        depth: u8,
        bound: u8,
        ply: u8,
    ) {
        let idx = (hash as usize) & self.size_mask;
        let entry = &self.table[idx];
        
        let adjusted_score = if score >= SCORE_MATE - MAX_PLY {
            score + ply as i32
        } else if score <= -SCORE_MATE + MAX_PLY {
            score - ply as i32
        } else {
            score
        };
        
        let current_age = self.get_age() as u8;
        
        if !self.should_replace(entry, depth, current_age, bound) {
            return;
        }
        
        let tt_data = TTData {
            best_move,
            score: adjusted_score,
            static_eval,
            depth,
            bound,
            age: current_age,
        };
        
        let packed_data = tt_data.pack();
        let key_xor = hash ^ packed_data;
        
        entry.data.store(packed_data, Ordering::Relaxed);
        entry.key_xor.store(key_xor, Ordering::Relaxed);
    }
    
    pub fn store_eval(
        &self,
        hash: u64,
        best_move: Move,
        eval_result: EvalResult,
        depth: u8,
        bound: u8,
        ply: u8,
    ) {
        self.store(
            hash,
            best_move,
            eval_result.score,
            eval_result.score,
            depth,
            bound,
            ply,
        );
    }
    
    pub fn clear(&self) {
        for entry in self.table.iter() {
            entry.key_xor.store(0, Ordering::Relaxed);
            entry.data.store(0, Ordering::Relaxed);
        }
    }
    
    pub fn new_search(&self) {
        self.age.fetch_add(1, Ordering::Relaxed);
    }
    
    #[inline(always)]
    fn get_age(&self) -> u64 {
        self.age.load(Ordering::Relaxed) & 0x3F
    }
    
    pub fn hashfull(&self) -> u32 {
        let sample_size = 1000.min(self.size);
        let mut filled = 0;
        let current_age = self.get_age() as u8;
        
        for i in 0..sample_size {
            let entry = &self.table[i];
            let key_xor = entry.key_xor.load(Ordering::Relaxed);
            if key_xor != 0 {
                let data = entry.data.load(Ordering::Relaxed);
                let existing = TTData::unpack(data);
                let age_diff = current_age.wrapping_sub(existing.age) & 0x3F;
                if age_diff <= 2 {
                    filled += 1;
                }
            }
        }
        
        (filled * 1000 / sample_size) as u32
    }
    
    #[inline(always)]
    pub fn prefetch(&self, hash: u64) {
        let idx = (hash as usize) & self.size_mask;
        let entry_ptr = &self.table[idx] as *const TTEntry;
        
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(
                entry_ptr as *const i8,
                std::arch::x86_64::_MM_HINT_T0
            );
        }
    }
}

impl TranspositionTable {
    #[inline(always)]
    pub fn score_from_tt(score: i32, ply: u8) -> i32 {
        if score >= SCORE_MATE - MAX_PLY {
            score - ply as i32
        } else if score <= -SCORE_MATE + MAX_PLY {
            score + ply as i32
        } else {
            score
        }
    }
    
    #[inline(always)]
    pub fn score_to_tt(score: i32, ply: u8) -> i32 {
        if score >= SCORE_MATE - MAX_PLY {
            score + ply as i32
        } else if score <= -SCORE_MATE + MAX_PLY {
            score - ply as i32
        } else {
            score
        }
    }
}

#[inline(always)]
pub fn tt_score_from_tt(score: i32, ply: u8) -> i32 {
    if score >= SCORE_MATE - MAX_PLY {
        score - ply as i32
    } else if score <= -SCORE_MATE + MAX_PLY {
        score + ply as i32
    } else {
        score
    }
}

#[inline(always)]
pub fn tt_score_to_tt(score: i32, ply: u8) -> i32 {
    if score >= SCORE_MATE - MAX_PLY {
        score + ply as i32
    } else if score <= -SCORE_MATE + MAX_PLY {
        score - ply as i32
    } else {
        score
    }
}

impl Move {
    #[inline(always)]
    pub fn as_u16(self) -> u16 {
        let from = self.from() as u16;
        let to = self.to() as u16;
        let move_type = self.move_type() as u16;
        
        from | (to << 6) | (move_type << 12)
    }
    
    #[inline(always)]
    pub fn from_u16(bits: u16) -> Self {
        let from = (bits & 0x3F) as u8;
        let to = ((bits >> 6) & 0x3F) as u8;
        let move_type_raw = ((bits >> 12) & 0xF) as u8;
        
        let move_type = match move_type_raw {
            0 => MoveType::Normal,
            1 => MoveType::EnPassant,
            2 => MoveType::Castle,
            3 => MoveType::Promotion,
            _ => MoveType::Normal,
        };
        
        let promotion = if move_type_raw == 3 {
            PieceType::Queen
        } else {
            PieceType::None
        };
        
        Move::new(from, to, move_type, promotion)
    }
}

impl Move {
    #[inline(always)]
    pub fn is_capture_on(self, pos: &Position) -> bool {
        let (piece, _color) = pos.piece_at(self.to());
        piece != PieceType::None || self.move_type() == MoveType::EnPassant
    }
    
    #[inline(always)]
    pub fn is_quiet_on(self, pos: &Position) -> bool {
        !self.is_capture_on(pos) && !self.is_promotion()
    }
    
    #[inline(always)]
    pub fn captured_piece_on(self, pos: &Position) -> PieceType {
        if self.move_type() == MoveType::EnPassant {
            PieceType::Pawn
        } else {
            pos.piece_at(self.to()).0
        }
    }
}

pub const TT_BOUND_NONE: u8 = BOUND_NONE;
pub const TT_BOUND_EXACT: u8 = BOUND_EXACT;
pub const TT_BOUND_LOWER: u8 = BOUND_LOWER;
pub const TT_BOUND_UPPER: u8 = BOUND_UPPER;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pack_unpack() {
        let original = TTData {
            best_move: Move::null(),
            score: 150,
            static_eval: 100,
            depth: 10,
            bound: TT_BOUND_EXACT,
            age: 5,
        };
        
        let packed = original.pack();
        let unpacked = TTData::unpack(packed);
        
        assert_eq!(unpacked.score, original.score);
        assert_eq!(unpacked.static_eval, original.static_eval);
        assert_eq!(unpacked.depth, original.depth);
        assert_eq!(unpacked.bound, original.bound);
        assert_eq!(unpacked.age, original.age);
    }
    
    #[test]
    fn test_store_probe() {
        let tt = TranspositionTable::new(1);
        
        let hash = 0x123456789ABCDEF0;
        let best_move = Move::null();
        
        tt.store(hash, best_move, 100, 90, 8, TT_BOUND_EXACT, 0);
        
        let result = tt.probe(hash);
        assert!(result.is_some());
        
        let data = result.unwrap();
        assert_eq!(data.score, 100);
        assert_eq!(data.depth, 8);
        assert_eq!(data.bound, TT_BOUND_EXACT);
    }
    
    #[test]
    fn test_replacement_scheme() {
        let tt = TranspositionTable::new(1);
        let hash = 0x123456789ABCDEF0;
        let best_move = Move::null();
        
        tt.store(hash, best_move, 50, 50, 4, TT_BOUND_EXACT, 0);
        
        tt.store(hash, best_move, 100, 100, 8, TT_BOUND_EXACT, 0);
        
        let result = tt.probe(hash).unwrap();
        assert_eq!(result.depth, 8);
        assert_eq!(result.score, 100);
    }
    
    #[test]
    fn test_age_replacement() {
        let tt = TranspositionTable::new(1);
        let hash = 0x123456789ABCDEF0;
        let best_move = Move::null();
        
        tt.store(hash, best_move, 100, 100, 8, TT_BOUND_EXACT, 0);
        
        for _ in 0..5 {
            tt.new_search();
        }
        
        tt.store(hash, best_move, 200, 200, 6, TT_BOUND_EXACT, 0);
        
        let result = tt.probe(hash).unwrap();
        assert_eq!(result.score, 200);
    }
}