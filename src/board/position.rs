
use super::zobrist::{self, toggle_piece_hash, toggle_en_passant_hash, toggle_castling_hash, toggle_side_to_move_hash};
use super::bitboard::*;
use crate::eval::evaluate::Score;
use crate::search::alphabeta::{see_ge_threshold};
use crate::movegen::magic::{all_attacks_for_king,get_bishop_attacks,get_king_attacks,get_pawn_attacks,get_rook_attacks,get_queen_attacks,get_knight_attacks};

#[repr(align(64))]
pub struct AlignedAttackTable<const N: usize> {
    pub data: [Bitboard; N],
}
const MAX_PLY:usize = 128;

pub static mut KNIGHT_ATTACKS: AlignedAttackTable<64> = AlignedAttackTable { data: [0; 64] };
pub static mut KING_ATTACKS: AlignedAttackTable<64> = AlignedAttackTable { data: [0; 64] };
pub static mut PAWN_ATTACKS: [AlignedAttackTable<64>; 2] = [
    AlignedAttackTable { data: [0; 64] },
    AlignedAttackTable { data: [0; 64] }
];

static INIT: std::sync::Once = std::sync::Once::new();

pub fn init_attack_tables() {
    INIT.call_once(|| {
        unsafe {
            for sq in 0..64 {
                KNIGHT_ATTACKS.data[sq] = compute_knight_attacks(sq as u8);
                KING_ATTACKS.data[sq] = compute_king_attacks(sq as u8);
                PAWN_ATTACKS[0].data[sq] = compute_pawn_attacks(sq as u8, Color::White);
                PAWN_ATTACKS[1].data[sq] = compute_pawn_attacks(sq as u8, Color::Black);
            }
        }
    });
}

#[inline(always)]
pub fn compute_knight_attacks(square: u8) -> Bitboard {
    const KNIGHT_DELTAS: [(i8, i8); 8] = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1)
    ];
    
    let mut attacks = EMPTY;
    let rank = (square >> 3) as i8;
    let file = (square & 7) as i8;
    
    for &(dr, df) in &KNIGHT_DELTAS {
        let new_rank = rank + dr;
        let new_file = file + df;
        
        if ((new_rank | new_file) & !7) == 0 {
            attacks |= 1u64 << ((new_rank << 3) | new_file);
        }
    }
    
    attacks
}

#[inline(always)]
pub fn compute_king_attacks(square: u8) -> Bitboard {
    let mut attacks = EMPTY;
    let sq_bb = square_mask(square);
    
    attacks |= shift_north(sq_bb) | shift_south(sq_bb);
    attacks |= shift_east(sq_bb) | shift_west(sq_bb);
    attacks |= shift_northeast(sq_bb) | shift_northwest(sq_bb);
    attacks |= shift_southeast(sq_bb) | shift_southwest(sq_bb);
    
    attacks
}

#[inline(always)]
pub fn compute_pawn_attacks(square: u8, color: Color) -> Bitboard {
    let sq_bb = square_mask(square);
    
    match color {
        Color::White => shift_northwest(sq_bb) | shift_northeast(sq_bb),
        Color::Black => shift_southwest(sq_bb) | shift_southeast(sq_bb),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PieceType {
    None = 0,
    Pawn = 1,
    Knight = 2,
    Bishop = 3,
    Rook = 4,
    Queen = 5,
    King = 6,
}

impl From<u8> for PieceType {
    fn from(n: u8) -> Self {
        match n {
            1 => PieceType::Pawn,
            2 => PieceType::Knight,
            3 => PieceType::Bishop,
            4 => PieceType::Rook,
            5 => PieceType::Queen,
            6 => PieceType::King,
            _ => PieceType::None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    #[inline(always)]
    pub const fn opposite(self) -> Color {
        unsafe { std::mem::transmute::<u8, Color>((self as u8) ^ 1) }
    }
}

pub const CASTLE_WK: u8 = 1;
pub const CASTLE_WQ: u8 = 2;
pub const CASTLE_BK: u8 = 4;
pub const CASTLE_BQ: u8 = 8;
pub const CASTLE_WHITE: u8 = CASTLE_WK | CASTLE_WQ;
pub const CASTLE_BLACK: u8 = CASTLE_BK | CASTLE_BQ;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MoveType {
    Normal = 0,
    EnPassant = 1,
    Castle = 2,
    Promotion = 3,
}

#[repr(align(64))]
pub struct CastleData {
    pub king_to: u8,
    pub rook_from: u8,
    pub rook_to: u8,
    pub castling_rights_mask: u8,
}

pub static CASTLE_LOOKUP: [CastleData; 4] = [
    CastleData { king_to: 6, rook_from: 7, rook_to: 5, castling_rights_mask: !CASTLE_WK },
    CastleData { king_to: 2, rook_from: 0, rook_to: 3, castling_rights_mask: !CASTLE_WQ },
    CastleData { king_to: 62, rook_from: 63, rook_to: 61, castling_rights_mask: !CASTLE_BK },
    CastleData { king_to: 58, rook_from: 56, rook_to: 59, castling_rights_mask: !CASTLE_BQ },
];

pub static CASTLING_MASKS: [u8; 64] = {
    let mut masks = [0xFF; 64];
    masks[0] = !(CASTLE_WQ);
    masks[4] = !(CASTLE_WHITE);
    masks[7] = !(CASTLE_WK);
    masks[56] = !(CASTLE_BQ);
    masks[60] = !(CASTLE_BLACK);
    masks[63] = !(CASTLE_BK);
    masks
};

static PIECE_HASH_LOOKUP: [[usize; 2]; 7] = [
    [0, 0],
    [0, 6],
    [1, 7],
    [2, 8],
    [3, 9],
    [4, 10],
    [5, 11],
];

static CASTLE_MOVES: [[(u8, u8, u8); 2]; 2] = [
    [(6, 7, 5), (2, 0, 3)],
    [(62, 63, 61), (58, 56, 59)],
];

static CASTLE_TYPE: [Option<(Color, usize)>; 64] = {
    let mut table = [None; 64];
    table[2] = Some((Color::White, 1));
    table[6] = Some((Color::White, 0));
    table[58] = Some((Color::Black, 1));
    table[62] = Some((Color::Black, 0));
    table
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Move(pub u32);

impl Move {
    #[inline(always)]
    pub const fn new(from: u8, to: u8, move_type: MoveType, promotion: PieceType) -> Self {
        Move(
            (from as u32) |
            ((to as u32) << 6) |
            ((move_type as u32) << 12) |
            ((promotion as u32) << 16)
        )
    }

    pub const fn null() -> Self {
        Move(0)
    }

    #[inline(always)]
    pub const fn from(self) -> u8 { (self.0 & 0x3F) as u8 }

    #[inline(always)]
    pub const fn from_u32(val:u32) -> Self { Self(val) }
    
    #[inline(always)]
    pub const fn to(self) -> u8 { ((self.0 >> 6) & 0x3F) as u8 }

    pub const fn to_u32(self) -> u32 { self.0}
    
    #[inline(always)]
    pub const fn move_type(self) -> MoveType {
        unsafe { std::mem::transmute(((self.0 >> 12) & 0xF) as u8) }
    }
    
    #[inline(always)]
    pub const fn promotion(self) -> PieceType {
        unsafe { std::mem::transmute(((self.0 >> 16) & 0xFF) as u8) }
    }

    

    pub const fn is_pawn_move(&self) -> bool {
        let move_type = (self.0 >> 12) & 0xF;
        move_type == 0 || move_type == 1
    }
    
    #[inline(always)]
    pub const fn is_en_passant(&self) -> bool {
        ((self.0 >> 12) & 0xF) == 1
    }
    
    #[inline(always)]
    pub const fn is_promotion(&self) -> bool {
        ((self.0 >> 12) & 0xF) == 3
    }

    #[inline(always)]
    pub const fn is_castling(&self) -> bool {
        ((self.0 >> 12) & 0xF) == 2
    }    

    pub const NULL: Move = Move(0);
}

impl Default for Move {
    #[inline(always)]
    fn default() -> Self {
        Self::NULL
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct UndoInfo {
    pub captured_piece: u8,
    pub castling_rights: u8,
    pub en_passant_square: u8,
    pub halfmove_clock: u8,
    pub hash_xor: u64,
    pub move_raw: u32,
}

impl Default for UndoInfo {
    fn default() -> Self {
        Self {
            captured_piece: 0,
            castling_rights: 0,
            en_passant_square: 64,
            halfmove_clock: 0,
            hash_xor: 0,
            move_raw: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Position {
    pub piece_bb: [Bitboard; 7],
    pub hash: u64,
    
    pub color_bb: [Bitboard; 2],
    pub occupied: Bitboard,
    pub side_to_move: Color,
    pub castling_rights: u8,
    pub en_passant_square: u8,
    pub halfmove_clock: u8,
    pub fullmove_number: u16,
    pub _padding1: [u8; 2],
    
    pub squares: [u8; 64],
    
    pub undo_stack: [UndoInfo; MAX_PLY],
    pub undo_i: u16,
}

#[repr(align(64))]
struct MoveHashData {
    piece_from_hash: u64,
    piece_to_hash: u64,
    capture_hash: u64,
    total_hash: u64,
}

impl Position {
    pub fn new() -> Self {
        let mut pos = Position {
            piece_bb: [EMPTY; 7],
            color_bb: [EMPTY; 2],
            occupied: EMPTY,
            squares: [0; 64],
            side_to_move: Color::White,
            castling_rights: CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ,
            en_passant_square: 64,
            halfmove_clock: 0,
            fullmove_number: 1,
            hash: 0,
            undo_stack: [UndoInfo::default(); MAX_PLY],
            undo_i: 0,
            _padding1: [0; 2],
        };
        
        pos.setup_startpos();
        pos.calculate_hash();
        pos
    }
    
    pub fn startpos() -> Self {
        let mut pos = Position {
            piece_bb: [EMPTY; 7],
            color_bb: [EMPTY; 2],
            occupied: EMPTY,
            squares: [0; 64],
            side_to_move: Color::White,
            castling_rights: CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ,
            en_passant_square: 64,
            halfmove_clock: 0,
            fullmove_number: 1,
            hash: 0,
            undo_stack: [UndoInfo::default(); MAX_PLY],
            undo_i: 0,
            _padding1: [0; 2],
        };
        
        pos.setup_startpos();
        pos.calculate_hash();
        pos
    }

    pub fn see_ge(&self, mv: Move, threshold: i32) -> bool {
        see_ge_threshold(self, mv, threshold)
    }
    
    fn setup_startpos(&mut self) {
        const WHITE_PIECES: [(u8, PieceType); 16] = [
            (0, PieceType::Rook), (1, PieceType::Knight), (2, PieceType::Bishop), (3, PieceType::Queen),
            (4, PieceType::King), (5, PieceType::Bishop), (6, PieceType::Knight), (7, PieceType::Rook),
            (8, PieceType::Pawn), (9, PieceType::Pawn), (10, PieceType::Pawn), (11, PieceType::Pawn),
            (12, PieceType::Pawn), (13, PieceType::Pawn), (14, PieceType::Pawn), (15, PieceType::Pawn),
        ];
        
        const BLACK_PIECES: [(u8, PieceType); 16] = [
            (48, PieceType::Pawn), (49, PieceType::Pawn), (50, PieceType::Pawn), (51, PieceType::Pawn),
            (52, PieceType::Pawn), (53, PieceType::Pawn), (54, PieceType::Pawn), (55, PieceType::Pawn),
            (56, PieceType::Rook), (57, PieceType::Knight), (58, PieceType::Bishop), (59, PieceType::Queen),
            (60, PieceType::King), (61, PieceType::Bishop), (62, PieceType::Knight), (63, PieceType::Rook),
        ];
        
        for &(sq, piece) in &WHITE_PIECES {
            self.place_piece(sq, piece, Color::White);
        }
        
        for &(sq, piece) in &BLACK_PIECES {
            self.place_piece(sq, piece, Color::Black);
        }
    }

    pub fn from_fen(fen: &str) -> Option<Self> {
        let mut pos = Position {
            piece_bb: [EMPTY; 7],
            color_bb: [EMPTY; 2],
            occupied: EMPTY,
            squares: [0; 64],
            side_to_move: Color::White,
            castling_rights: 0,
            en_passant_square: 64,
            halfmove_clock: 0,
            fullmove_number: 1,
            hash: 0,
            undo_stack: [UndoInfo::default(); MAX_PLY],
            _padding1: [0; 2],
            undo_i: 0,
        };

        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            return None;
        }

        let mut square = 56;
        for ch in parts[0].chars() {
            match ch {
                '/' => square -= 16,
                '1'..='8' => square += ch as u8 - b'0',
                _ => {
                    let (color, piece_type) = match ch {
                        'P' => (Color::White, PieceType::Pawn),
                        'N' => (Color::White, PieceType::Knight),
                        'B' => (Color::White, PieceType::Bishop),
                        'R' => (Color::White, PieceType::Rook),
                        'Q' => (Color::White, PieceType::Queen),
                        'K' => (Color::White, PieceType::King),
                        'p' => (Color::Black, PieceType::Pawn),
                        'n' => (Color::Black, PieceType::Knight),
                        'b' => (Color::Black, PieceType::Bishop),
                        'r' => (Color::Black, PieceType::Rook),
                        'q' => (Color::Black, PieceType::Queen),
                        'k' => (Color::Black, PieceType::King),
                        _ => return None,
                    };
                    
                    if square >= 64 {
                        return None;
                    }
                    
                    pos.place_piece(square, piece_type, color);
                    square += 1;
                }
            }
        }

        pos.side_to_move = match parts.get(1) {
            Some(&"w") => Color::White,
            Some(&"b") => Color::Black,
            _ => return None,
        };

        if let Some(&castle_str) = parts.get(2) {
            if castle_str != "-" {
                for ch in castle_str.chars() {
                    pos.castling_rights |= match ch {
                        'K' => CASTLE_WK,
                        'Q' => CASTLE_WQ,
                        'k' => CASTLE_BK,
                        'q' => CASTLE_BQ,
                        _ => return None,
                    };
                }
            }
        }

        if let Some(&ep_str) = parts.get(3) {
            pos.en_passant_square = if ep_str == "-" {
                64
            } else {
                algebraic_to_square(ep_str)?
            };
        }

        if let Some(&hm) = parts.get(4) {
            pos.halfmove_clock = hm.parse().ok()?;
        }

        if let Some(&fm) = parts.get(5) {
            pos.fullmove_number = fm.parse().ok()?;
        }

        pos.calculate_hash();
        Some(pos)
    }

    pub fn is_capture(&self, mv: Move) -> bool {
        let (piece_at_to, _) = self.piece_at(mv.to());
        piece_at_to != PieceType::None || mv.is_en_passant()
    }

    pub fn piece_value(&self, piece: PieceType) -> i32 {
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

    pub fn to_fen(&self) -> String {
        let mut fen = String::with_capacity(87);

        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let square = rank * 8 + file;
                let packed = self.squares[square as usize];
                
                if packed == 0 {
                    empty += 1;
                } else {
                    if empty > 0 {
                        fen.push((b'0' + empty) as char);
                        empty = 0;
                    }
                    
                    let piece_type = packed & 7;
                    let is_white = (packed >> 3) & 1;
                    
                    let ch = match piece_type {
                        1 => 'p',
                        2 => 'n',
                        3 => 'b',
                        4 => 'r',
                        5 => 'q',
                        6 => 'k',
                        _ => unreachable!(),
                    };
                    
                    fen.push(if is_white == 1 { ch.to_ascii_uppercase() } else { ch });
                }
            }
            
            if empty > 0 {
                fen.push((b'0' + empty) as char);
            }
            
            if rank > 0 {
                fen.push('/');
            }
        }

        fen.push(' ');
        fen.push(if self.side_to_move == Color::White { 'w' } else { 'b' });
        
        fen.push(' ');
        let castle_count = fen.len();
        if self.castling_rights & CASTLE_WK != 0 { fen.push('K'); }
        if self.castling_rights & CASTLE_WQ != 0 { fen.push('Q'); }
        if self.castling_rights & CASTLE_BK != 0 { fen.push('k'); }
        if self.castling_rights & CASTLE_BQ != 0 { fen.push('q'); }
        if fen.len() == castle_count { fen.push('-'); }
        
        fen.push(' ');
        if self.en_passant_square < 64 {
            fen.push_str(&square_to_algebraic(self.en_passant_square));
        } else {
            fen.push('-');
        }
        
        fen.push(' ');
        fen.push_str(&self.halfmove_clock.to_string());
        
        fen.push(' ');
        fen.push_str(&self.fullmove_number.to_string());
        
        fen
    }
    
    #[inline(always)]
    pub fn place_piece(&mut self, square: u8, piece_type: PieceType, color: Color) {
        let mask = square_mask(square);
        
        self.piece_bb[piece_type as usize] |= mask;
        self.color_bb[color as usize] |= mask;
        self.occupied |= mask;
        
        self.squares[square as usize] = ((color as u8) << 3) | piece_type as u8;
    }

    pub fn captured_piece(&self, mv: Move) -> PieceType {
        let (captured, _) = self.piece_at(mv.to());
        captured
    }

    pub fn is_pinned(&self, square: u8, color: Color) -> bool {
        let king_square = self.king_square(color);
        if king_square == 64 {
            return false;
        }

        if square == king_square {
            return false;
        }

        let king_rank = king_square / 8;
        let king_file = king_square % 8;
        let piece_rank = square / 8;
        let piece_file = square % 8;

        let (rank_dir, file_dir) = {
            let rank_diff = king_rank as i8 - piece_rank as i8;
            let file_diff = king_file as i8 - piece_file as i8;

            if rank_diff != 0 && file_diff != 0 && rank_diff.abs() != file_diff.abs() {
                return false;
            }

            (
                if rank_diff == 0 { 0 } else { rank_diff / rank_diff.abs() },
                if file_diff == 0 { 0 } else { file_diff / file_diff.abs() },
            )
        };

        let mut current_square = square as i8;
        let mut between_empty = true;

        loop {
            current_square += rank_dir * 8 + file_dir;
            let current_sq = current_square as u8;

            if current_sq == king_square {
                break;
            }

            if self.squares[current_sq as usize] != 0 {
                between_empty = false;
                break;
            }
        }

        if !between_empty {
            return false;
        }

        let opponent_color = color.opposite();
        let mut current_square = square as i8;

        loop {
            current_square -= rank_dir * 8 + file_dir;
            if current_square < 0 || current_square >= 64 {
                break;
            }

            let current_sq = current_square as u8;
            if self.squares[current_sq as usize] != 0 {
                let (piece_type, piece_color) = self.piece_at(current_sq);
                
                if piece_color == opponent_color {
                    match piece_type {
                        PieceType::Queen => return true,
                        PieceType::Rook if rank_dir == 0 || file_dir == 0 => return true,
                        PieceType::Bishop if rank_dir != 0 && file_dir != 0 => return true,
                        _ => (),
                    }
                }
                break;
            }
        }

        false
    }

    pub fn is_block_check(&self, mv: &Move,atts:Bitboard) -> bool {
        let from = mv.from();
        let to = mv.to();
        let color = self.side_to_move;
        let king_square = self.king_square(color);

        if !self.is_in_check_upg(color,atts) {
            return false;
        }

        if from == king_square {
            return !self.is_square_attacked_ovi_upg(to, color.opposite(),atts);
        }

        let checkers = self.find_all_checking_pieces(king_square, color);
        if checkers.is_empty() {
            return false;
        }

        if checkers.len() > 1 {
            return false;
        }

        let (attacker_square, attacker_piece) = checkers[0];

        if to == attacker_square {
            return true;
        }

        match attacker_piece {
            PieceType::Pawn | PieceType::Knight | PieceType::King => {
                false
            },
            _ => {
                let attack_path = self.get_attack_path(king_square, attacker_square, attacker_piece);
                attack_path & square_mask(to) != 0
            }
        }
    }

    pub fn get_pin_direction(&self, square: u8, color: Color) -> Option<(i8, i8)> {
        let king_square = self.king_square(color);
        if king_square == 64 || square == king_square {
            return None;
        }

        let king_rank = king_square / 8;
        let king_file = king_square % 8;
        let piece_rank = square / 8;
        let piece_file = square % 8;

        let rank_diff = king_rank as i8 - piece_rank as i8;
        let file_diff = king_file as i8 - piece_file as i8;

        if rank_diff != 0 && file_diff != 0 && rank_diff.abs() != file_diff.abs() {
            return None;
        }

        let (rank_dir, file_dir) = (
            if rank_diff == 0 { 0 } else { rank_diff / rank_diff.abs() },
            if file_diff == 0 { 0 } else { file_diff / file_diff.abs() },
        );

        let mut current_square = square as i8;
        let mut between_empty = true;

        loop {
            current_square += rank_dir * 8 + file_dir;
            let current_sq = current_square as u8;

            if current_sq == king_square {
                break;
            }

            if self.squares[current_sq as usize] != 0 {
                between_empty = false;
                break;
            }
        }

        if !between_empty {
            return None;
        }

        let opponent_color = color.opposite();
        let mut current_square = square as i8;

        loop {
            current_square -= rank_dir * 8 + file_dir;
            if current_square < 0 || current_square >= 64 {
                break;
            }

            let current_sq = current_square as u8;
            if self.squares[current_sq as usize] != 0 {
                let (piece_type, piece_color) = self.piece_at(current_sq);
                
                if piece_color == opponent_color {
                    match piece_type {
                        PieceType::Queen => return Some((rank_dir, file_dir)),
                        PieceType::Rook if rank_dir == 0 || file_dir == 0 => return Some((rank_dir, file_dir)),
                        PieceType::Bishop if rank_dir != 0 && file_dir != 0 => return Some((rank_dir, file_dir)),
                        _ => (),
                    }
                }
                break;
            }
        }

        None
    }

    pub fn is_move_along_pin(&self, mv: Move, pin_direction: (i8, i8)) -> bool {
        let from = mv.from();
        let to = mv.to();
        
        if mv.is_en_passant() {
            return self.is_en_passant_move_legal_for_pinned_piece(mv, pin_direction);
        }

        let from_rank = from / 8;
        let from_file = from % 8;
        let to_rank = to / 8;
        let to_file = to % 8;

        let move_rank_diff = to_rank as i8 - from_rank as i8;
        let move_file_diff = to_file as i8 - from_file as i8;

        let (pin_rank_dir, pin_file_dir) = pin_direction;

        if move_rank_diff == 0 && move_file_diff == 0 {
            return false;
        }

        let move_rank_dir = if move_rank_diff == 0 { 0 } else { move_rank_diff.signum() };
        let move_file_dir = if move_file_diff == 0 { 0 } else { move_file_diff.signum() };

        (move_rank_dir == pin_rank_dir && move_file_dir == pin_file_dir) ||
        (move_rank_dir == -pin_rank_dir && move_file_dir == -pin_file_dir)
    }

    fn is_en_passant_move_legal_for_pinned_piece(&self, mv: Move, pin_direction: (i8, i8)) -> bool {
        let from = mv.from();
        let to = mv.to();
        let captured_pawn_square = to ^ 8;
        
        let king_square = self.king_square(self.side_to_move);
        
        
        let from_rank = from / 8;
        let from_file = from % 8;
        let to_rank = to / 8;
        let to_file = to % 8;

        let move_rank_diff = to_rank as i8 - from_rank as i8;
        let move_file_diff = to_file as i8 - from_file as i8;

        let (pin_rank_dir, pin_file_dir) = pin_direction;

        let move_rank_dir = if move_rank_diff == 0 { 0 } else { move_rank_diff.signum() };
        let move_file_dir = if move_file_diff == 0 { 0 } else { move_file_diff.signum() };

        if (move_rank_dir == pin_rank_dir && move_file_dir == pin_file_dir) ||
           (move_rank_dir == -pin_rank_dir && move_file_dir == -pin_file_dir) {
            return true;
        }

        false
    }

    pub fn get_pinned_piece_legal_squares(&self, square: u8, color: Color) -> Bitboard {
        if let Some((rank_dir, file_dir)) = self.get_pin_direction(square, color) {
            let mut legal_squares = 0u64;
            let king_square = self.king_square(color);
            
            let mut current = square as i8;
            
            loop {
                current += rank_dir * 8 + file_dir;
                if current < 0 || current >= 64 {
                    break;
                }
                let current_sq = current as u8;
                
                if current_sq == king_square {
                    break;
                }
                
                if self.squares[current_sq as usize] == 0 {
                    legal_squares |= 1u64 << current_sq;
                } else {
                    let (_, piece_color) = self.piece_at(current_sq);
                    if piece_color != color {
                        legal_squares |= 1u64 << current_sq;
                    }
                    break;
                }
            }
            
            current = square as i8;
            loop {
                current -= rank_dir * 8 + file_dir;
                if current < 0 || current >= 64 {
                    break;
                }
                let current_sq = current as u8;
                
                if self.squares[current_sq as usize] == 0 {
                    legal_squares |= 1u64 << current_sq;
                } else {
                    let (_, piece_color) = self.piece_at(current_sq);
                    if piece_color != color {
                        legal_squares |= 1u64 << current_sq;
                    }
                    break;
                }
            }
            
            legal_squares
        } else {
            0xFFFFFFFFFFFFFFFFu64
        }
    }

    pub fn find_all_checking_pieces(&self, king_square: u8, color: Color) -> Vec<(u8, PieceType)> {
        let mut checkers = Vec::new();
        let opponent = color.opposite();

        unsafe {
            let pawn_attackers = PAWN_ATTACKS[color as usize].data[king_square as usize] & 
                                self.pieces_colored(PieceType::Pawn, opponent);
            let mut pawns = pawn_attackers;
            while pawns != 0 {
                let sq = pawns.trailing_zeros() as u8;
                pawns &= pawns - 1;
                checkers.push((sq, PieceType::Pawn));
            }

            let knight_attackers = KNIGHT_ATTACKS.data[king_square as usize] & 
                                  self.pieces_colored(PieceType::Knight, opponent);
            let mut knights = knight_attackers;
            while knights != 0 {
                let sq = knights.trailing_zeros() as u8;
                knights &= knights - 1;
                checkers.push((sq, PieceType::Knight));
            }

            let king_attackers = KING_ATTACKS.data[king_square as usize] & 
                                self.pieces_colored(PieceType::King, opponent);
            let mut kings = king_attackers;
            while kings != 0 {
                let sq = kings.trailing_zeros() as u8;
                kings &= kings - 1;
                checkers.push((sq, PieceType::King));
            }
        }

        let bishops = self.pieces_colored(PieceType::Bishop, opponent);
        let rooks = self.pieces_colored(PieceType::Rook, opponent);
        let queens = self.pieces_colored(PieceType::Queen, opponent);

        let diagonal_attackers = self.get_diagonal_attacks(king_square) & (bishops | queens);
        let mut diag_attackers = diagonal_attackers;
        while diag_attackers != 0 {
            let sq = diag_attackers.trailing_zeros() as u8;
            diag_attackers &= diag_attackers - 1;
            
            if (bishops & square_mask(sq)) != 0 {
                checkers.push((sq, PieceType::Bishop));
            } else {
                checkers.push((sq, PieceType::Queen));
            }
        }

        let straight_attackers = self.get_straight_attacks(king_square) & (rooks | queens);
        let mut str_attackers = straight_attackers;
        while str_attackers != 0 {
            let sq = str_attackers.trailing_zeros() as u8;
            str_attackers &= str_attackers - 1;
            
            if (rooks & square_mask(sq)) != 0 {
                checkers.push((sq, PieceType::Rook));
            } else {
                checkers.push((sq, PieceType::Queen));
            }
        }

        checkers
    }

    fn find_checking_piece(&self, king_square: u8, color: Color) -> Option<(u8, PieceType)> {
        let checkers = self.find_all_checking_pieces(king_square, color);
        checkers.first().copied()
    }

    fn get_attack_path(&self, king_sq: u8, attacker_sq: u8, attacker_piece: PieceType) -> Bitboard {
        match attacker_piece {
            PieceType::Pawn | PieceType::Knight | PieceType::King => {
                EMPTY
            },
            PieceType::Rook => {
                self.get_straight_attack_path(king_sq, attacker_sq)
            },
            PieceType::Bishop => {
                self.get_diagonal_attack_path(king_sq, attacker_sq)
            },
            PieceType::Queen => {
                if self.is_same_rank_or_file(king_sq, attacker_sq) {
                    self.get_straight_attack_path(king_sq, attacker_sq)
                } else {
                    self.get_diagonal_attack_path(king_sq, attacker_sq)
                }
            },
            _ => EMPTY,
        }
    }

    fn get_straight_attack_path(&self, king_sq: u8, attacker_sq: u8) -> Bitboard {
        if king_sq / 8 == attacker_sq / 8 {
            let start = std::cmp::min(king_sq, attacker_sq);
            let end = std::cmp::max(king_sq, attacker_sq);
            ((start + 1)..end).fold(EMPTY, |acc, sq| acc | square_mask(sq))
        } else if king_sq % 8 == attacker_sq % 8 {
            let start = std::cmp::min(king_sq, attacker_sq);
            let end = std::cmp::max(king_sq, attacker_sq);
            ((start + 8)..end).step_by(8).fold(EMPTY, |acc, sq| acc | square_mask(sq))
        } else {
            EMPTY
        }
    }

    fn get_diagonal_attack_path(&self, king_sq: u8, attacker_sq: u8) -> Bitboard {
        let king_rank = (king_sq / 8) as i8;
        let king_file = (king_sq % 8) as i8;
        let attacker_rank = (attacker_sq / 8) as i8;
        let attacker_file = (attacker_sq % 8) as i8;

        let rank_diff = king_rank - attacker_rank;
        let file_diff = king_file - attacker_file;

        if rank_diff.abs() != file_diff.abs() {
            return EMPTY;
        }

        let rank_dir = if rank_diff > 0 { 1 } else { -1 };
        let file_dir = if file_diff > 0 { 1 } else { -1 };

        let mut path = EMPTY;
        let mut r = attacker_rank + rank_dir;
        let mut f = attacker_file + file_dir;

        while r != king_rank && f != king_file {
            if r < 0 || r >= 8 || f < 0 || f >= 8 {
                break;
            }
            let sq = (r * 8 + f) as u8;
            path |= square_mask(sq);
            r += rank_dir;
            f += file_dir;
        }

        path
    }

    fn is_same_rank_or_file(&self, sq1: u8, sq2: u8) -> bool {
        sq1 / 8 == sq2 / 8 || sq1 % 8 == sq2 % 8
    }

    fn count_checks(&self, color: Color) -> u32 {
        let king_sq = self.king_square(color);
        if king_sq == 64 {
            return 0;
        }

        self.find_all_checking_pieces(king_sq, color).len() as u32
    }

    #[inline(always)]
    pub fn make_null_move(&mut self) {
        self.side_to_move = match self.side_to_move {
            Color::White => Color::Black,
            Color::Black => Color::White,
        };
        self.hash ^= 0x9E3779B97F4A7C15u64;
    }
    
    #[inline(always)]
    pub fn unmake_null_move(&mut self) {
        self.side_to_move = match self.side_to_move {
            Color::White => Color::Black,
            Color::Black => Color::White,
        };
        self.hash ^= 0x9E3779B97F4A7C15u64;
    }
    
    pub fn gives_check(&self, mv: Move) -> bool {
        let to_sq = mv.to() as usize;
        let piece_type = self.squares[mv.from() as usize] & 7;
        let enemy_king_sq = self.find_king(self.side_to_move.opposite());
        
        if enemy_king_sq == 64 {
            return false;
        }
        
        match piece_type {
            1 => {
                let rank_diff = ((enemy_king_sq / 8) as i32 - (to_sq / 8) as i32).abs();
                let file_diff = ((enemy_king_sq % 8) as i32 - (to_sq % 8) as i32).abs();
                rank_diff == 1 && file_diff == 1
            }
            2 => {
                let rank_diff = ((enemy_king_sq / 8) as i32 - (to_sq / 8) as i32).abs();
                let file_diff = ((enemy_king_sq % 8) as i32 - (to_sq % 8) as i32).abs();
                (rank_diff == 2 && file_diff == 1) || (rank_diff == 1 && file_diff == 2)
            }
            3 => {
                let rank_diff = (enemy_king_sq / 8) as i32 - (to_sq / 8) as i32;
                let file_diff = (enemy_king_sq % 8) as i32 - (to_sq % 8) as i32;
                rank_diff.abs() == file_diff.abs() && self.is_diagonal_clear(to_sq, enemy_king_sq)
            }
            4 => {
                let same_rank = enemy_king_sq / 8 == to_sq / 8;
                let same_file = enemy_king_sq % 8 == to_sq % 8;
                (same_rank && self.is_rank_clear(to_sq, enemy_king_sq)) ||
                (same_file && self.is_file_clear(to_sq, enemy_king_sq))
            }
            5 => {
                let rank_diff = (enemy_king_sq / 8) as i32 - (to_sq / 8) as i32;
                let file_diff = (enemy_king_sq % 8) as i32 - (to_sq % 8) as i32;
                let diagonal = rank_diff.abs() == file_diff.abs();
                let straight = enemy_king_sq / 8 == to_sq / 8 || enemy_king_sq % 8 == to_sq % 8;
                
                (diagonal && self.is_diagonal_clear(to_sq, enemy_king_sq)) ||
                (straight && (self.is_rank_clear(to_sq, enemy_king_sq) || 
                             self.is_file_clear(to_sq, enemy_king_sq)))
            }
            _ => false,
        }
    }
    
    fn find_king(&self, color: Color) -> usize {
        let king_piece = 6 | (if color == Color::Black { 8 } else { 0 });
        for sq in 0..64 {
            if self.squares[sq] == king_piece {
                return sq;
            }
        }
        64
    }
    
    fn is_diagonal_clear(&self, from: usize, to: usize) -> bool {
        let rank_dir = if to / 8 > from / 8 { 8 } else { -8 };
        let file_dir = if to % 8 > from % 8 { 1 } else { -1 };
        let dir = rank_dir + file_dir;
        
        let mut sq = from as i32 + dir;
        let to_i32 = to as i32;
        
        while sq != to_i32 {
            if sq < 0 || sq >= 64 || self.squares[sq as usize] != 0 {
                return false;
            }
            sq += dir;
        }
        true
    }
    
    fn is_rank_clear(&self, from: usize, to: usize) -> bool {
        let start = from.min(to) + 1;
        let end = from.max(to);
        for sq in start..end {
            if sq / 8 == from / 8 && self.squares[sq] != 0 {
                return false;
            }
        }
        true
    }
    
    fn is_file_clear(&self, from: usize, to: usize) -> bool {
        let step = 8;
        let start = from.min(to) + step;
        let end = from.max(to);
        let mut sq = start;
        while sq < end {
            if self.squares[sq] != 0 {
                return false;
            }
            sq += step;
        }
        true
    }
    
    #[inline(always)]
    pub fn remove_piece(&mut self, square: u8) {
        let mask = !square_mask(square);
        let packed = self.squares[square as usize];
        let piece_type = packed & 7;
        let color = (packed >> 3) & 1;
        
        self.piece_bb[piece_type as usize] &= mask;
        self.color_bb[color as usize] &= mask;
        self.occupied &= mask;
        self.squares[square as usize] = 0;
    }

    pub fn occupied_without_king(&self, color: Color) -> Bitboard {
        let king_square = self.king_square(color);
        if king_square == 64 {
            return self.occupied;
        }
        
        self.occupied & !(1u64 << king_square)
    }
    
    #[inline(always)]
    pub fn piece_at(&self, square: u8) -> (PieceType, Color) {
        let packed = self.squares[square as usize];
        let piece_type = unsafe { std::mem::transmute(packed & 7) };
        let color = unsafe { std::mem::transmute((packed >> 3) & 1) };
        (piece_type, color)
    }
    
    #[inline(always)]
    pub fn pieces(&self, color: Color) -> Bitboard {
        self.color_bb[color as usize]
    }

    #[inline(always)]
    pub fn all_pieces(&self) -> Bitboard {
        self.occupied
    }
    
    #[inline(always)]
    pub fn pieces_of_type(&self, piece_type: PieceType) -> Bitboard {
        self.piece_bb[piece_type as usize]
    }
    
    #[inline(always)]
    pub fn pieces_colored(&self, piece_type: PieceType, color: Color) -> Bitboard {
        self.piece_bb[piece_type as usize] & self.color_bb[color as usize]
    }

    #[inline(always)]
    pub fn piece_count(&self, color: Color, piece_type: PieceType) -> u32 {
        (self.piece_bb[piece_type as usize] & self.color_bb[color as usize]).count_ones()
    }
    
    #[inline(always)]
    pub fn king_square(&self, color: Color) -> u8 {
        (self.pieces_colored(PieceType::King, color)).trailing_zeros() as u8
    }
    
    pub fn king_check(&self, color: Color) -> bool {
        self.pieces_colored(PieceType::King, color).count_ones() == 1
    }
    
    pub fn calculate_hash(&mut self) {
        self.hash = 0;
        
        for color in 0..2 {
            let color_bb = self.color_bb[color];
            if color_bb == 0 { continue; }
            
            for piece_type in 1..7 {
                let mut pieces = self.piece_bb[piece_type] & color_bb;
                while pieces != 0 {
                    let square = pop_lsb(&mut pieces);
                    let zobrist_index = zobrist::piece_to_zobrist_index(piece_type as u8, color == 0);
                    toggle_piece_hash(&mut self.hash, zobrist_index, square);
                }
            }
        }
        
        if self.side_to_move == Color::Black {
            toggle_side_to_move_hash(&mut self.hash);
        }
        
        if self.castling_rights != 0 {
            toggle_castling_hash(&mut self.hash, 0, self.castling_rights);
        }
        
        if self.en_passant_square < 64 {
            toggle_en_passant_hash(&mut self.hash, self.en_passant_square & 7);
        }
    }

    pub fn push_undo(self, undo: UndoInfo) {
    }

    pub fn is_draw(&self) -> bool {
        crate::eval::eval_util::is_likely_draw(self)
    }

    #[inline(always)]
    pub fn make_move(&mut self, mv: Move) -> bool {
        let move_data = mv.0;
        let from = (move_data & 0x3F) as u8;
        let to = ((move_data >> 6) & 0x3F) as u8;
        let move_type = ((move_data >> 12) & 0xF) as u8;
        
        let piece_data = unsafe { *self.squares.get_unchecked(from as usize) };
        let piece_type = piece_data & 7;
        let color = piece_data >> 3;
        
        let mut undo = UndoInfo {
            captured_piece: 0,
            castling_rights: self.castling_rights,
            en_passant_square: self.en_passant_square,
            halfmove_clock: self.halfmove_clock,
            hash_xor: 0,
            move_raw: move_data,
        };
        
        let mut hash_delta = 0u64;
        
        let from_mask = 1u64 << from;
        let to_mask = 1u64 << to;
        
        match move_type {
            0 => {
                let captured = self.squares[to as usize];
                let captured_type = captured & 7;
                
                if captured != 0 {
                    let captured_color = captured >> 3;
                    self.piece_bb[captured_type as usize] &= !to_mask;
                    self.color_bb[captured_color as usize] &= !to_mask;
                    
                    unsafe {
                        let idx = PIECE_HASH_LOOKUP[captured_type as usize][captured_color as usize];
                        hash_delta ^= zobrist::ZOBRIST.pieces[idx][to as usize];
                    }
                }
                
                let move_mask = from_mask ^ to_mask;
                self.piece_bb[piece_type as usize] ^= move_mask;
                self.color_bb[color as usize] ^= move_mask;
                self.occupied = (self.occupied & !to_mask) ^ move_mask;
                
                self.squares[from as usize] = 0;
                self.squares[to as usize] = piece_data;
                
                unsafe {
                    let idx = PIECE_HASH_LOOKUP[piece_type as usize][color as usize];
                    hash_delta ^= zobrist::ZOBRIST.pieces[idx][from as usize];
                    hash_delta ^= zobrist::ZOBRIST.pieces[idx][to as usize];
                }
                
                undo.captured_piece = captured;
                self.undo_stack[self.undo_i as usize] = undo;
            },
            
            1 => {
                let captured_sq = to ^ 8;
                let captured_mask = 1u64 << captured_sq;
                
                let move_mask = from_mask ^ to_mask;
                self.piece_bb[1] ^= move_mask;
                self.color_bb[color as usize] ^= move_mask;
                
                self.piece_bb[1] &= !captured_mask;
                self.color_bb[(color ^ 1) as usize] &= !captured_mask;
                self.occupied ^= move_mask ^ captured_mask;
                
                self.squares[from as usize] = 0;
                self.squares[to as usize] = piece_data;
                self.squares[captured_sq as usize] = 0;
                
                unsafe {
                    let idx = PIECE_HASH_LOOKUP[1][color as usize];
                    hash_delta ^= zobrist::ZOBRIST.pieces[idx][from as usize];
                    hash_delta ^= zobrist::ZOBRIST.pieces[idx][to as usize];
                    
                    let cap_idx = PIECE_HASH_LOOKUP[1][(color ^ 1) as usize];
                    hash_delta ^= zobrist::ZOBRIST.pieces[cap_idx][captured_sq as usize];
                }
                
                undo.captured_piece = 1 | ((captured_sq as u8) << 3);
                self.undo_stack[self.undo_i as usize] = undo;
            },
            
            2 => {
                let castle_idx = match to {
                    6 => 0,
                    2 => 1,
                    62 => 2,
                    58 => 3,
                    _ => return false,
                };
                
                let castle_data = &CASTLE_LOOKUP[castle_idx];
                let rook_from = castle_data.rook_from;
                let rook_to = castle_data.rook_to;
                
                let king_mask = from_mask ^ to_mask;
                self.piece_bb[6] ^= king_mask;
                self.color_bb[color as usize] ^= king_mask;
                
                let rook_from_mask = 1u64 << rook_from;
                let rook_to_mask = 1u64 << rook_to;
                let rook_mask = rook_from_mask ^ rook_to_mask;
                self.piece_bb[4] ^= rook_mask;
                self.color_bb[color as usize] ^= rook_mask;
                
                self.occupied ^= king_mask ^ rook_mask;
                
                self.squares[from as usize] = 0;
                self.squares[to as usize] = piece_data;
                self.squares[rook_from as usize] = 0;
                self.squares[rook_to as usize] = 4 | (color << 3);
                
                unsafe {
                    let king_idx = PIECE_HASH_LOOKUP[6][color as usize];
                    hash_delta ^= zobrist::ZOBRIST.pieces[king_idx][from as usize];
                    hash_delta ^= zobrist::ZOBRIST.pieces[king_idx][to as usize];
                    
                    let rook_idx = PIECE_HASH_LOOKUP[4][color as usize];
                    hash_delta ^= zobrist::ZOBRIST.pieces[rook_idx][rook_from as usize];
                    hash_delta ^= zobrist::ZOBRIST.pieces[rook_idx][rook_to as usize];
                }
                
                self.undo_stack[self.undo_i as usize] = undo;
            },
            
            3 => {
                let promoted_type = ((move_data >> 16) & 7) as u8;
                
                let captured = self.squares[to as usize];
                let captured_type = captured & 7;
                
                if captured != 0 {
                    let captured_color = captured >> 3;
                    self.piece_bb[captured_type as usize] &= !to_mask;
                    self.color_bb[captured_color as usize] &= !to_mask;
                    
                    unsafe {
                        let idx = PIECE_HASH_LOOKUP[captured_type as usize][captured_color as usize];
                        hash_delta ^= zobrist::ZOBRIST.pieces[idx][to as usize];
                    }
                }
                
                self.piece_bb[1] &= !from_mask;
                self.piece_bb[promoted_type as usize] |= to_mask;
                self.color_bb[color as usize] ^= from_mask ^ to_mask;
                self.occupied = (self.occupied & !to_mask) ^ from_mask ^ to_mask;
                
                self.squares[from as usize] = 0;
                self.squares[to as usize] = promoted_type | (color << 3);
                
                unsafe {
                    let pawn_idx = PIECE_HASH_LOOKUP[1][color as usize];
                    hash_delta ^= zobrist::ZOBRIST.pieces[pawn_idx][from as usize];
                    
                    let promo_idx = PIECE_HASH_LOOKUP[promoted_type as usize][color as usize];
                    hash_delta ^= zobrist::ZOBRIST.pieces[promo_idx][to as usize];
                }
                
                undo.captured_piece = captured;
                self.undo_stack[self.undo_i as usize] = undo;
            },
            
            _ => return false,
        }
        
        if self.en_passant_square < 64 {
            unsafe {
                hash_delta ^= zobrist::ZOBRIST.en_passant[(self.en_passant_square & 7) as usize];
            }
        }
        
let is_double_pawn = ((piece_type == 1) && ((to ^ from) == 16)) as u8;
        self.en_passant_square = if is_double_pawn != 0 { from ^ 8 } else { 64 };
        
        if self.en_passant_square < 64 {
            unsafe {
                hash_delta ^= zobrist::ZOBRIST.en_passant[(self.en_passant_square & 7) as usize];
            }
        }
        
        let new_castling = self.castling_rights & CASTLING_MASKS[from as usize] & CASTLING_MASKS[to as usize];
        if new_castling != self.castling_rights {
            unsafe {
                hash_delta ^= zobrist::ZOBRIST.castling[self.castling_rights as usize];
                hash_delta ^= zobrist::ZOBRIST.castling[new_castling as usize];
            }
            self.castling_rights = new_castling;
        }
        
        let reset_clock = (piece_type == 1) | (self.undo_stack[self.undo_i as usize].captured_piece != 0);
        self.halfmove_clock = if reset_clock { 0 } else { self.halfmove_clock + 1 };
        
        self.side_to_move = unsafe { std::mem::transmute::<u8, Color>(color ^ 1) };
        self.fullmove_number += color as u16;
        
        unsafe {
            hash_delta ^= zobrist::ZOBRIST.side_to_move;
        }
        
        self.undo_stack[self.undo_i as usize].hash_xor = hash_delta;
        self.hash ^= hash_delta;
        
        self.undo_i += 1;
        
        true
    }
    
    #[inline(always)]
    pub fn unmake_move(&mut self, mv: Move) {
        self.undo_i -= 1;
        let undo = self.undo_stack[self.undo_i as usize];
        
        let move_data = mv.0;
        let from = (move_data & 0x3F) as u8;
        let to = ((move_data >> 6) & 0x3F) as u8;
        let move_type = ((move_data >> 12) & 0xF) as u8;
        
        self.hash ^= undo.hash_xor;
        
        self.castling_rights = undo.castling_rights;
        self.en_passant_square = undo.en_passant_square;
        self.halfmove_clock = undo.halfmove_clock;
        
        self.side_to_move = self.side_to_move.opposite();
        self.fullmove_number -= self.side_to_move as u16;
        
        let piece_data = self.squares[to as usize];
        let color = piece_data >> 3;
        
        let from_mask = 1u64 << from;
        let to_mask = 1u64 << to;
        
        match move_type {
            0 => {
                let piece_type = piece_data & 7;
                
                let move_mask = from_mask ^ to_mask;
                self.piece_bb[piece_type as usize] ^= move_mask;
                self.color_bb[color as usize] ^= move_mask;
                
                self.squares[from as usize] = piece_data;
                self.squares[to as usize] = 0;
                
                let captured = undo.captured_piece;
                if captured != 0 {
                    let cap_type = captured & 7;
                    let cap_color = captured >> 3;
                    
                    self.piece_bb[cap_type as usize] |= to_mask;
                    self.color_bb[cap_color as usize] |= to_mask;
                    self.squares[to as usize] = captured;
                    self.occupied |= from_mask;
                } else {
                    self.occupied ^= move_mask;
                }
            },
            
            1 => {
                let move_mask = from_mask ^ to_mask;
                self.piece_bb[1] ^= move_mask;
                self.color_bb[color as usize] ^= move_mask;
                
                let captured_sq = (undo.captured_piece >> 3) as u8;
                let captured_mask = 1u64 << captured_sq;
                
                self.piece_bb[1] |= captured_mask;
                self.color_bb[(color ^ 1) as usize] |= captured_mask;
                self.occupied ^= move_mask ^ captured_mask;
                
                self.squares[from as usize] = piece_data;
                self.squares[to as usize] = 0;
                self.squares[captured_sq as usize] = 1 | ((color ^ 1) << 3);
            },
            
            2 => {
                let castle_idx = match to {
                    6 => 0,
                    2 => 1,
                    62 => 2,
                    58 => 3,
                    _ => return,
                };
                
                let castle_data = &CASTLE_LOOKUP[castle_idx];
                let rook_from = castle_data.rook_from;
                let rook_to = castle_data.rook_to;
                
                let king_mask = from_mask ^ to_mask;
                self.piece_bb[6] ^= king_mask;
                self.color_bb[color as usize] ^= king_mask;
                
                let rook_from_mask = 1u64 << rook_from;
                let rook_to_mask = 1u64 << rook_to;
                let rook_mask = rook_from_mask ^ rook_to_mask;
                self.piece_bb[4] ^= rook_mask;
                self.color_bb[color as usize] ^= rook_mask;
                
                self.occupied ^= king_mask ^ rook_mask;
                
                self.squares[from as usize] = 6 | (color << 3);
                self.squares[to as usize] = 0;
                self.squares[rook_from as usize] = 4 | (color << 3);
                self.squares[rook_to as usize] = 0;
            },
            
            3 => {
                let promoted_type = piece_data & 7;
                
                self.piece_bb[promoted_type as usize] &= !to_mask;
                self.piece_bb[1] |= from_mask;
                self.color_bb[color as usize] ^= from_mask ^ to_mask;
                
                self.squares[from as usize] = 1 | (color << 3);
                self.squares[to as usize] = 0;
                
                let captured = undo.captured_piece;
                if captured != 0 {
                    let cap_type = captured & 7;
                    let cap_color = captured >> 3;
                    
                    self.piece_bb[cap_type as usize] |= to_mask;
                    self.color_bb[cap_color as usize] |= to_mask;
                    self.squares[to as usize] = captured;
                    self.occupied |= from_mask;
                } else {
                    self.occupied ^= from_mask ^ to_mask;
                }
            },
            
            _ => {},
        }
    }
    
    #[inline(always)]
    pub fn is_in_check(&self, color: Color) -> bool {
        let king_bb = self.pieces_colored(PieceType::King, color);
        if king_bb == EMPTY {
            return false;
        }
        let king_square = king_bb.trailing_zeros() as u8;
        self.is_square_attacked_ovi(king_square, color.opposite())
    }

    pub fn is_square_attacked_after_move(&self, square: u8, by_color: Color, occupied: u64) -> bool {
        
        let pawn_attacks = get_pawn_attacks(square, by_color.opposite());
        if pawn_attacks & self.pieces_colored(PieceType::Pawn, by_color) != 0 {
            return true;
        }
        
        let knight_attacks = get_knight_attacks(square);
        if knight_attacks & self.pieces_colored(PieceType::Knight, by_color) != 0 {
            return true;
        }
        
        let bishop_attacks = get_bishop_attacks(square, occupied);
        if bishop_attacks & (self.pieces_colored(PieceType::Bishop, by_color) | 
                           self.pieces_colored(PieceType::Queen, by_color)) != 0 {
            return true;
        }
        
        let rook_attacks = get_rook_attacks(square, occupied);
        if rook_attacks & (self.pieces_colored(PieceType::Rook, by_color) | 
                         self.pieces_colored(PieceType::Queen, by_color)) != 0 {
            return true;
        }
        
        let king_attacks = get_king_attacks(square);
        if king_attacks & self.pieces_colored(PieceType::King, by_color) != 0 {
            return true;
        }
        
        false
    }

    #[inline(always)]
    pub fn is_in_check_upg(&self, color: Color,atts:Bitboard) -> bool {
        let king_bb = self.pieces_colored(PieceType::King, color);
        if king_bb == EMPTY {
            return false;
        }
        let king_square = king_bb.trailing_zeros() as u8;
        self.is_square_attacked_ovi_upg(king_square, color.opposite(),atts)
    }

    pub fn is_in_check_with_all_attackers(&self, color: Color) -> (bool, Vec<PieceType>) {
        let king_bb = self.pieces_colored(PieceType::King, color);
        if king_bb == EMPTY {
            return (false, vec![]);
        }

        let king_square = king_bb.trailing_zeros() as u8;
        let attacker_color = color.opposite();
        let mut attackers = Vec::new();

        unsafe {
            let pawns = self.pieces_colored(PieceType::Pawn, attacker_color);
            if (PAWN_ATTACKS[attacker_color as usize].data[king_square as usize] & pawns) != EMPTY {
                attackers.push(PieceType::Pawn);
            }

            let knights = self.pieces_colored(PieceType::Knight, attacker_color);
            if (KNIGHT_ATTACKS.data[king_square as usize] & knights) != EMPTY {
                attackers.push(PieceType::Knight);
            }

            let kings = self.pieces_colored(PieceType::King, attacker_color);
            if (KING_ATTACKS.data[king_square as usize] & kings) != EMPTY {
                attackers.push(PieceType::King);
            }
        }

        let queens = self.pieces_colored(PieceType::Queen, attacker_color);
        let bishops = self.pieces_colored(PieceType::Bishop, attacker_color);
        let rooks = self.pieces_colored(PieceType::Rook, attacker_color);

        let diagonal_attackers = bishops | queens;
        if diagonal_attackers != EMPTY && self.has_diagonal_attack(king_square, diagonal_attackers) {
            if self.has_diagonal_attack(king_square, bishops) {
                attackers.push(PieceType::Bishop);
            }
            if self.has_diagonal_attack(king_square, queens) {
                attackers.push(PieceType::Queen);
            }
        }

        let straight_attackers = rooks | queens;
        if straight_attackers != EMPTY && self.has_straight_attack(king_square, straight_attackers) {
            if self.has_straight_attack(king_square, rooks) {
                attackers.push(PieceType::Rook);
            }
            if self.has_straight_attack(king_square, queens) {
                attackers.push(PieceType::Queen);
            }
        }

        let is_check = !attackers.is_empty();
        (is_check, attackers)
    }

    #[inline(always)]
    pub fn square_attacked_by(&self, square: u8, by_color: Color) -> Option<PieceType> {
        unsafe {
            let pawns = self.pieces_colored(PieceType::Pawn, by_color);
            if (PAWN_ATTACKS[by_color as usize].data[square as usize] & pawns) != EMPTY {
                return Some(PieceType::Pawn);
            }

            let knights = self.pieces_colored(PieceType::Knight, by_color);
            if (KNIGHT_ATTACKS.data[square as usize] & knights) != EMPTY {
                return Some(PieceType::Knight);
            }

            let kings = self.pieces_colored(PieceType::King, by_color);
            if (KING_ATTACKS.data[square as usize] & kings) != EMPTY {
                return Some(PieceType::King);
            }
        }

        let queens = self.pieces_colored(PieceType::Queen, by_color);
        let bishops = self.pieces_colored(PieceType::Bishop, by_color);
        let rooks = self.pieces_colored(PieceType::Rook, by_color);

        let diagonal_attackers = bishops | queens;
        if diagonal_attackers != EMPTY {
            if self.has_diagonal_attack(square, diagonal_attackers) {
                if (self.has_diagonal_attack(square, bishops)) {
                    return Some(PieceType::Bishop);
                }
                return Some(PieceType::Queen);
            }
        }

        let straight_attackers = rooks | queens;
        if straight_attackers != EMPTY {
            if self.has_straight_attack(square, straight_attackers) {
                if (self.has_straight_attack(square, rooks)) {
                    return Some(PieceType::Rook);
                }
                return Some(PieceType::Queen);
            }
        }

        None
    }
    
    #[inline(always)]
    pub fn is_square_attacked_ovi(&self, square: u8, by_color: Color) -> bool {
        unsafe {
            let all_att = all_attacks_for_king(self, by_color);

            if (all_att & 
                (1u64 << square)) != EMPTY {
                    
                return true;
            }
            false
            
    }}

    #[inline(always)]
    pub fn is_square_attacked_ovi_upg(&self, square: u8, by_color: Color,atts:Bitboard) -> bool {
        unsafe {

            if (atts & 
                (1u64 << square)) != EMPTY {
                    
                return true;
            }
            false
            
    }}

    #[inline(always)]
    pub fn is_square_attacked(&self, square: u8, by_color: Color) -> bool {
        unsafe {
            if (get_pawn_attacks(square, by_color) & 
                (1u64 << square)) != EMPTY {
                    println!("piyon tarafından tehdit");
                return true;
            }
            
            if (get_knight_attacks(square) & 
                self.pieces_colored(PieceType::Knight, by_color)) != EMPTY {
                return true;
            }
            
            if (get_king_attacks(square)& 
                self.pieces_colored(PieceType::King, by_color)) != EMPTY {
                return true;
            }
        }
        
        let queens = self.pieces_colored(PieceType::Queen, by_color);
        
        if (self.pieces_colored(PieceType::Bishop, by_color) | queens) != EMPTY {
            if self.has_diagonal_attack(square, self.pieces_colored(PieceType::Bishop, by_color) | queens) {
                return true;
            }
        }
        
        if (self.pieces_colored(PieceType::Rook, by_color) | queens) != EMPTY {
            if self.has_straight_attack(square, self.pieces_colored(PieceType::Rook, by_color) | queens) {
                return true;
            }
        }
        
        false
    }

    #[inline(always)]
    fn has_diagonal_attack(&self, square: u8, attackers: Bitboard) -> bool {
        let mut mask = square_mask(square);
        loop {
            mask = shift_northeast(mask);
            if mask == 0 { break; }
            if (mask & self.occupied) != 0 {
                if (mask & attackers) != 0 { return true; }
                break;
            }
        }
        
        mask = square_mask(square);
        loop {
            mask = shift_northwest(mask);
            if mask == 0 { break; }
            if (mask & self.occupied) != 0 {
                if (mask & attackers) != 0 { return true; }
                break;
            }
        }
        
        mask = square_mask(square);
        loop {
            mask = shift_southeast(mask);
            if mask == 0 { break; }
            if (mask & self.occupied) != 0 {
                if (mask & attackers) != 0 { return true; }
                break;
            }
        }
        
        mask = square_mask(square);
        loop {
            mask = shift_southwest(mask);
            if mask == 0 { break; }
            if (mask & self.occupied) != 0 {
                if (mask & attackers) != 0 { return true; }
                break;
            }
        }
        
        false
    }
    
    #[inline(always)]
    fn has_straight_attack(&self, square: u8, attackers: Bitboard) -> bool {
        let mut mask = square_mask(square);
        loop {
            mask = shift_north(mask);
            if mask == 0 { break; }
            if (mask & self.occupied) != 0 {
                if (mask & attackers) != 0 { return true; }
                break;
            }
        }
        
        mask = square_mask(square);
        loop {
            mask = shift_south(mask);
            if mask == 0 { break; }
            if (mask & self.occupied) != 0 {
                if (mask & attackers) != 0 { return true; }
                break;
            }
        }
        
        mask = square_mask(square);
        loop {
            mask = shift_east(mask);
            if mask == 0 { break; }
            if (mask & self.occupied) != 0 {
                if (mask & attackers) != 0 { return true; }
                break;
            }
        }
        
        mask = square_mask(square);
        loop {
            mask = shift_west(mask);
            if mask == 0 { break; }
            if (mask & self.occupied) != 0 {
                if (mask & attackers) != 0 { return true; }
                break;
            }
        }
        
        false
    }
    
    #[inline(always)]
    pub fn get_knight_attacks(&self, square: u8) -> Bitboard {
        unsafe { KNIGHT_ATTACKS.data[square as usize] }
    }
    
    #[inline(always)]
    pub fn get_king_attacks(&self, square: u8) -> Bitboard {
        unsafe { KING_ATTACKS.data[square as usize] }
    }

    pub fn copy_from(&mut self, other: &Position) {
        self.piece_bb = other.piece_bb;
        self.hash = other.hash;
        self.color_bb = other.color_bb;
        self.occupied = other.occupied;
        
        self.side_to_move = other.side_to_move;
        self.castling_rights = other.castling_rights;
        self.en_passant_square = other.en_passant_square;
        self.halfmove_clock = other.halfmove_clock;
        self.fullmove_number = other.fullmove_number;
        
        self.squares = other.squares;
        
        self.undo_stack = [UndoInfo::default(); MAX_PLY];
        self.undo_i = 0;
    }
    
    pub fn create_temp_copy(&self) -> Position {
        Position {
            piece_bb: self.piece_bb,
            color_bb: self.color_bb,
            occupied: self.occupied,
            squares: self.squares,
            side_to_move: self.side_to_move,
            castling_rights: self.castling_rights,
            en_passant_square: self.en_passant_square,
            halfmove_clock: self.halfmove_clock,
            fullmove_number: self.fullmove_number,
            hash: self.hash,
            undo_stack: [UndoInfo::default(); MAX_PLY],
            _padding1: [0; 2],
            undo_i: 0,
        }
    }
    
    pub fn get_diagonal_attacks(&self, square: u8) -> Bitboard {
        let mut attacks = EMPTY;
        let sq_bb = square_mask(square);
        
        let mut ray = sq_bb;
        loop {
            ray = shift_northeast(ray);
            if ray == 0 { break; }
            attacks |= ray;
            if (ray & self.occupied) != 0 { break; }
        }
        
        ray = sq_bb;
        loop {
            ray = shift_northwest(ray);
            if ray == 0 { break; }
            attacks |= ray;
            if (ray & self.occupied) != 0 { break; }
        }
        
        ray = sq_bb;
        loop {
            ray = shift_southeast(ray);
            if ray == 0 { break; }
            attacks |= ray;
            if (ray & self.occupied) != 0 { break; }
        }
        
        ray = sq_bb;
        loop {
            ray = shift_southwest(ray);
            if ray == 0 { break; }
            attacks |= ray;
            if (ray & self.occupied) != 0 { break; }
        }
        
        attacks
    }
    
    fn get_straight_attacks(&self, square: u8) -> Bitboard {
        let mut attacks = EMPTY;
        let sq_bb = square_mask(square);
        
        let mut ray = sq_bb;
        loop {
            ray = shift_north(ray);
            if ray == 0 { break; }
            attacks |= ray;
            if (ray & self.occupied) != 0 { break; }
        }
        
        ray = sq_bb;
        loop {
            ray = shift_south(ray);
            if ray == 0 { break; }
            attacks |= ray;
            if (ray & self.occupied) != 0 { break; }
        }
        
        ray = sq_bb;
        loop {
            ray = shift_east(ray);
            if ray == 0 { break; }
            attacks |= ray;
            if (ray & self.occupied) != 0 { break; }
        }
        
        ray = sq_bb;
        loop {
            ray = shift_west(ray);
            if ray == 0 { break; }
            attacks |= ray;
            if (ray & self.occupied) != 0 { break; }
        }
        
        attacks
    }
    
    pub fn print(&self) {
        println!("   +-----------------+");
        for rank in (0..8).rev() {
            print!(" {} |", rank + 1);
            for file in 0..8 {
                let square = rank * 8 + file;
                let packed = self.squares[square as usize];
                
                let symbol = if packed == 0 {
                    '.'
                } else {
                    let piece_type = packed & 7;
                    let is_white = (packed >> 3) & 1;
                    
                    let ch = match piece_type {
                        1 => 'p',
                        2 => 'n',
                        3 => 'b',
                        4 => 'r',
                        5 => 'q',
                        6 => 'k',
                        _ => '?',
                    };
                    
                    if is_white == 1 { ch.to_ascii_uppercase() } else { ch }
                };
                
                print!(" {}", symbol);
            }
            println!(" |");
        }
        println!("   +-----------------+");
        println!("     a b c d e f g h");
        println!("Hamle sırası: {:?}", self.side_to_move);
        println!("Rok hakları: {:04b}", self.castling_rights);
        if self.en_passant_square < 64 {
            println!("En passant: {}", square_to_algebraic(self.en_passant_square));
        }
        println!("Hash: {:016X}", self.hash);
    }
    
    pub fn material_count(&self, color: Color) -> i32 {
        const PIECE_VALUES: [i32; 7] = [0, 100, 320, 330, 500, 900, 0];
        
        let mut total = 0;
        for piece_type in 1..6 {
            total += self.piece_count(color, PieceType::from(piece_type)) as i32
                     * PIECE_VALUES[piece_type as usize];
        }
        total
    }
    
    pub fn is_valid(&self) -> bool {
        if self.pieces_colored(PieceType::King, Color::White).count_ones() != 1 ||
           self.pieces_colored(PieceType::King, Color::Black).count_ones() != 1 {
            return false;
        }
        
        if self.is_in_check(self.side_to_move.opposite()) {
            return false;
        }
        
        let white_pawns = self.pieces_colored(PieceType::Pawn, Color::White);
        let black_pawns = self.pieces_colored(PieceType::Pawn, Color::Black);
        
        if white_pawns.count_ones() > 8 || black_pawns.count_ones() > 8 {
            return false;
        }
        
        if (white_pawns | black_pawns) & (RANK_1 | RANK_8) != EMPTY {
            return false;
        }
        
        true
    }
}

pub fn square_to_algebraic(square: u8) -> String {
    super::bitboard::square_to_algebraic(square)
}

pub fn algebraic_to_square(algebraic: &str) -> Option<u8> {
    super::bitboard::algebraic_to_square(algebraic)
}

impl Default for Position {
    fn default() -> Self {
        Self::startpos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::zobrist;

    #[test]
    fn test_startpos() {
        zobrist::init_zobrist();
        init_attack_tables();
        let pos = Position::startpos();
        
        assert_eq!(pos.side_to_move, Color::White);
        assert_eq!(pos.castling_rights, CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ);
        assert_eq!(pos.en_passant_square, 64);
        assert_eq!(pos.halfmove_clock, 0);
        assert_eq!(pos.fullmove_number, 1);
        
        assert_eq!(pos.king_square(Color::White), 4);
        assert_eq!(pos.king_square(Color::Black), 60);
        
        assert!(pos.is_valid());
    }
    
    #[test]
    fn test_piece_placement() {
        zobrist::init_zobrist();
        init_attack_tables();
        let pos = Position::startpos();
        
        assert_eq!(pos.piece_at(0), (PieceType::Rook, Color::White));
        assert_eq!(pos.piece_at(4), (PieceType::King, Color::White));
        assert_eq!(pos.piece_at(60), (PieceType::King, Color::Black));
        assert_eq!(pos.piece_at(63), (PieceType::Rook, Color::Black));
    }
    
    #[test]
    fn test_algebraic_conversion() {
        assert_eq!(square_to_algebraic(0), "a1");
        assert_eq!(square_to_algebraic(7), "h1");
        assert_eq!(square_to_algebraic(56), "a8");
        assert_eq!(square_to_algebraic(63), "h8");
        
        assert_eq!(algebraic_to_square("a1"), Some(0));
        assert_eq!(algebraic_to_square("h1"), Some(7));
        assert_eq!(algebraic_to_square("a8"), Some(56));
        assert_eq!(algebraic_to_square("h8"), Some(63));
        assert_eq!(algebraic_to_square("z9"), None);
    }
    
    #[test]
    fn test_make_unmake_move() {
        zobrist::init_zobrist();
        init_attack_tables();
        let mut pos = Position::startpos();
        let original_hash = pos.hash;
        
        let mv = Move::new(12, 28, MoveType::Normal, PieceType::None);
        
        pos.make_move(mv);
        assert_eq!(pos.side_to_move, Color::Black);
        assert_eq!(pos.piece_at(12), (PieceType::None, Color::White));
        assert_eq!(pos.piece_at(28), (PieceType::Pawn, Color::White));
        assert_eq!(pos.en_passant_square, 20);
        
        pos.unmake_move(mv);
        assert_eq!(pos.side_to_move, Color::White);
        assert_eq!(pos.piece_at(12), (PieceType::Pawn, Color::White));
        assert_eq!(pos.piece_at(28), (PieceType::None, Color::White));
        assert_eq!(pos.en_passant_square, 64);
        assert_eq!(pos.hash, original_hash);
    }
}