use crate::board::bitboard::{Bitboard, square_mask, has_bit};
use crate::board::position::{Position, Color, PieceType, Move, MoveType};
use crate::board::zobrist::{CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ};
use crate::movegen::magic::all_attacks_for_king;
use super::magic::{get_rook_attacks, get_bishop_attacks, get_queen_attacks, 
                   get_knight_attacks, get_king_attacks, get_pawn_attacks,
                   add_moves_from_bitboard,all_attacks};

const PROMOTION_PIECES: [PieceType; 4] = [
    PieceType::Queen,
    PieceType::Rook,
    PieceType::Bishop,
    PieceType::Knight,
];

#[inline(always)]
pub fn generate_moves(pos: &Position) -> Vec<Move> {
    let mut moves = Vec::with_capacity(256);
    
    generate_pawn_moves(pos, &mut moves);
    generate_knight_moves(pos, &mut moves);
    generate_bishop_moves(pos, &mut moves);
    generate_rook_moves(pos, &mut moves);
    generate_queen_moves(pos, &mut moves);
    generate_king_moves(pos, &mut moves);
    
    moves
}
use std::arch::x86_64::*;


fn generate_pawn_moves(pos: &Position, moves: &mut Vec<Move>) {
    let color = pos.side_to_move;
    let opponent = color.opposite();
    let pawns = pos.pieces_colored(PieceType::Pawn, color);
    let occupancy = pos.all_pieces();
    let opponent_pieces = pos.pieces(opponent);
    
    let (push_offset, double_push_offset, start_rank, promo_rank, ep_capture_rank) = match color {
        Color::White => (8i8, 16i8, 1u8, 7u8, 4u8),
        Color::Black => (-8i8, -16i8, 6u8, 0u8, 3u8),
    };
    
    let mut bb = pawns;
    while bb != 0 {
        let from = bb.trailing_zeros() as u8;
        
        bb &= bb - 1;
        
        let from_rank = from / 8;
        let from_file = from % 8;
        
        let to = (from as i8 + push_offset) as u8;
        if to < 64 && (occupancy & (1 << to)) == 0 {
            if (to / 8) == promo_rank {
                for &promo in &PROMOTION_PIECES {
                    moves.push(Move::new(from, to, MoveType::Promotion, promo));
                }
            } else {
                moves.push(Move::new(from, to, MoveType::Normal, PieceType::None));
                
                if from_rank == start_rank {
                    let to2 = (from as i8 + double_push_offset) as u8;
                    if (occupancy & (1 << to2)) == 0 {
                        moves.push(Move::new(from, to2, MoveType::Normal, PieceType::None));
                    }
                }
            }
        }
        
        let attacks = get_pawn_attacks(from, color) & opponent_pieces;
        let mut attack_bb = attacks;
        while attack_bb != 0 {
            let to = attack_bb.trailing_zeros() as u8;
            attack_bb &= attack_bb - 1;
            
            if (to / 8) == promo_rank {
                for &promo in &PROMOTION_PIECES {
                    moves.push(Move::new(from, to, MoveType::Promotion, promo));
                }
            } else {
                moves.push(Move::new(from, to, MoveType::Normal, PieceType::None));
            }
        }
        
        if pos.en_passant_square < 64 && from_rank == ep_capture_rank {
            let ep = pos.en_passant_square;
            let ep_file = ep % 8;
            
            if ((from_file > 0 && ep_file == from_file - 1) ||
                (from_file < 7 && ep_file == from_file + 1)) &&
               (ep / 8) == (ep_capture_rank as i8 + push_offset / 8) as u8 {
                moves.push(Move::new(from, ep, MoveType::EnPassant, PieceType::None));
            }
        }
    }
}

fn generate_knight_moves(pos: &Position, moves: &mut Vec<Move>) {
    let color = pos.side_to_move;
    let knights = pos.pieces_colored(PieceType::Knight, color);
    let our_pieces = pos.pieces(color);
    
    let mut knights_bb = knights;
    while knights_bb != 0 {
        let from = knights_bb.trailing_zeros() as u8;
        knights_bb &= knights_bb - 1;
        
        let attacks = get_knight_attacks(from) & !our_pieces;
        add_moves_from_bitboard(from, attacks, moves);
    }
}

fn generate_bishop_moves(pos: &Position, moves: &mut Vec<Move>) {
    let color = pos.side_to_move;
    let bishops = pos.pieces_colored(PieceType::Bishop, color);
    let occupancy = pos.all_pieces();
    let our_pieces = pos.pieces(color);
    
    let mut bishops_bb = bishops;
    while bishops_bb != 0 {
        let from = bishops_bb.trailing_zeros() as u8;
        bishops_bb &= bishops_bb - 1;
        
        let attacks = get_bishop_attacks(from, occupancy) & !our_pieces;
        add_moves_from_bitboard(from, attacks, moves);
    }
}

fn generate_rook_moves(pos: &Position, moves: &mut Vec<Move>) {
    let color = pos.side_to_move;
    let rooks = pos.pieces_colored(PieceType::Rook, color);
    let occupancy = pos.all_pieces();
    let our_pieces = pos.pieces(color);
    
    let mut rooks_bb = rooks;
    while rooks_bb != 0 {
        let from = rooks_bb.trailing_zeros() as u8;
        rooks_bb &= rooks_bb - 1;
        
        let attacks = get_rook_attacks(from, occupancy) & !our_pieces;
        add_moves_from_bitboard(from, attacks, moves);
    }
}

fn generate_queen_moves(pos: &Position, moves: &mut Vec<Move>) {
    let color = pos.side_to_move;
    let queens = pos.pieces_colored(PieceType::Queen, color);
    let occupancy = pos.all_pieces();
    let our_pieces = pos.pieces(color);
    
    let mut queens_bb = queens;
    while queens_bb != 0 {
        let from = queens_bb.trailing_zeros() as u8;
        queens_bb &= queens_bb - 1;
        
        let attacks = get_queen_attacks(from, occupancy) & !our_pieces;
        add_moves_from_bitboard(from, attacks, moves);
    }
}

fn generate_king_moves(pos: &Position, moves: &mut Vec<Move>) {
    let color = pos.side_to_move;
    let opponent = color.opposite();
    let from = pos.king_square(color);
    let our_pieces = pos.pieces(color);
    
    let opponent_attacks = all_attacks_for_king(pos, opponent);
    
    let mut attacks = get_king_attacks(from);
    let targets = attacks & !our_pieces & !opponent_attacks;
    add_moves_from_bitboard(from, targets, moves);
    
    if opponent_attacks & (1 << from) != 0 {
        return;
    }
    
    let (kingside_right, queenside_right) = match color {
        Color::White => (CASTLE_WK, CASTLE_WQ),
        Color::Black => (CASTLE_BK, CASTLE_BQ),
    };
    
    let occupancy = pos.all_pieces();
    
    if pos.castling_rights & kingside_right != 0 {
        let (empty_mask, safe_mask, king_to) = match color {
            Color::White => (0x60u64, 0x70u64, 6),
            Color::Black => (0x6000000000000000u64, 0x7000000000000000u64, 62),
        };
        
        if (occupancy & empty_mask) == 0 {
            let mut safe = true;
            let mut path = safe_mask;
            while path != 0 {
                let sq = path.trailing_zeros() as u8;
                path &= path - 1;
                if sq != from && (opponent_attacks & (1 << sq)) != 0 {
                    safe = false;
                    break;
                }
            }
            if safe {
                moves.push(Move::new(from, king_to, MoveType::Castle, PieceType::None));
            }
        }
    }
    
    if pos.castling_rights & queenside_right != 0 {
        let (empty_mask, safe_mask, king_to) = match color {
            Color::White => (0x0Eu64, 0x1Cu64, 2),
            Color::Black => (0x0E00000000000000u64, 0x1C00000000000000u64, 58),
        };
        
        if (occupancy & empty_mask) == 0 {
            let mut safe = true;
            let mut path = safe_mask;
            while path != 0 {
                let sq = path.trailing_zeros() as u8;
                path &= path - 1;
                if sq != from && (opponent_attacks & (1 << sq)) != 0 {
                    safe = false;
                    break;
                }
            }
            if safe {
                moves.push(Move::new(from, king_to, MoveType::Castle, PieceType::None));
            }
        }
    }
}


#[inline(always)]
fn is_path_attacked(pos: &Position, path: Bitboard, king_sq: u8, by_color: Color) -> bool {
    let mut squares_bb = path;
    while squares_bb != 0 {
        let sq = squares_bb.trailing_zeros() as u8;
        squares_bb &= squares_bb - 1;
        
        if sq != king_sq && pos.is_square_attacked(sq, by_color) {
            return true;
        }
    }
    false
}


#[inline(always)]
pub fn generate_moves_ovi(pos: &Position) -> (Vec<Move>,u64) {
    let mut moves = Vec::with_capacity(256);
    
    generate_pawn_moves(pos, &mut moves);
    generate_knight_moves(pos, &mut moves);
    generate_bishop_moves(pos, &mut moves);
    generate_rook_moves(pos, &mut moves);
    generate_queen_moves(pos, &mut moves);
    let attacks = generate_king_moves_ovi(pos, &mut moves);
    
    (moves,attacks)
}

fn generate_king_moves_ovi(pos: &Position, moves: &mut Vec<Move>)->u64 {
    let color = pos.side_to_move;
    let opponent = color.opposite();
    let from = pos.king_square(color);
    let our_pieces = pos.pieces(color);
    
    let opponent_attacks = all_attacks_for_king(pos, opponent);
    
    let mut attacks = get_king_attacks(from);
    let targets = attacks & !our_pieces & !opponent_attacks;
    add_moves_from_bitboard(from, targets, moves);
    
    if opponent_attacks & (1 << from) != 0 {
        return opponent_attacks;
    }
    
    let (kingside_right, queenside_right) = match color {
        Color::White => (CASTLE_WK, CASTLE_WQ),
        Color::Black => (CASTLE_BK, CASTLE_BQ),
    };
    
    let occupancy = pos.all_pieces();
    
    if pos.castling_rights & kingside_right != 0 {
        let (empty_mask, safe_mask, king_to) = match color {
            Color::White => (0x60u64, 0x70u64, 6),
            Color::Black => (0x6000000000000000u64, 0x7000000000000000u64, 62),
        };
        
        if (occupancy & empty_mask) == 0 {
            let mut safe = true;
            let mut path = safe_mask;
            while path != 0 {
                let sq = path.trailing_zeros() as u8;
                path &= path - 1;
                if sq != from && (opponent_attacks & (1 << sq)) != 0 {
                    safe = false;
                    break;
                }
            }
            if safe {
                moves.push(Move::new(from, king_to, MoveType::Castle, PieceType::None));
            }
        }
    }
    
    if pos.castling_rights & queenside_right != 0 {
        let (empty_mask, safe_mask, king_to) = match color {
            Color::White => (0x0Eu64, 0x1Cu64, 2),
            Color::Black => (0x0E00000000000000u64, 0x1C00000000000000u64, 58),
        };
        
        if (occupancy & empty_mask) == 0 {
            let mut safe = true;
            let mut path = safe_mask;
            while path != 0 {
                let sq = path.trailing_zeros() as u8;
                path &= path - 1;
                if sq != from && (opponent_attacks & (1 << sq)) != 0 {
                    safe = false;
                    break;
                }
            }
            if safe {
                moves.push(Move::new(from, king_to, MoveType::Castle, PieceType::None));
            }
        }
    }
    opponent_attacks
}