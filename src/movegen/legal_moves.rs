use crate::board::position::{Position, Move, Color, PieceType};
use crate::movegen::moves::generate_moves_ovi;
use crate::movegen::magic::all_attacks_for_king;


pub fn generate_legal_moves_ovi(pos: &Position) -> Vec<Move> {
    let is_in_check = pos.is_in_check(pos.side_to_move);
    let (pseudo_moves,opponent_attacks) = generate_moves_ovi(pos);
    let mut legal_moves = Vec::with_capacity(pseudo_moves.len().min(128));
    
    let our_color = pos.side_to_move;
    let opponent_color = our_color.opposite();

    let pinned_pieces = find_pinned_pieces(pos, our_color);
    
    for mv in pseudo_moves {
        if is_legal_move(&mv, pos, &pinned_pieces, is_in_check, opponent_attacks) {
            legal_moves.push(mv);
        }
    }
    
    legal_moves
}

fn find_pinned_pieces(pos: &Position, color: Color) -> Vec<(u8, (i8, i8))> {
    let mut pinned_pieces = Vec::new();
    let mut pieces_bb = pos.pieces(color);
    
    while pieces_bb != 0 {
        let square = pieces_bb.trailing_zeros() as u8;
        pieces_bb &= pieces_bb - 1;
        
        if let Some(pin_direction) = pos.get_pin_direction(square, color) {
            pinned_pieces.push((square, pin_direction));
        }
    }
    
    pinned_pieces
}

fn is_legal_move(
    mv: &Move, 
    pos: &Position, 
    pinned_pieces: &[(u8, (i8, i8))], 
    is_in_check: bool,
    opponent_attacks:u64
) -> bool {
    let from_square = mv.from();
    
    if let Some((_, pin_direction)) = pinned_pieces.iter().find(|(square, _)| *square == from_square) {
        if !is_pinned_move_legal(mv, pos, *pin_direction) {
            return false;
        }
    }
    
    if is_in_check {
        pos.is_block_check(mv, opponent_attacks)
    } else {
        true
    }
}

fn is_pinned_move_legal(mv: &Move, pos: &Position, pin_direction: (i8, i8)) -> bool {
    let from_square = mv.from();
    let piece_type = (pos.squares[from_square as usize] & 7) as u8;
    
    if piece_type == 2 {
        return false;
    }
    
    pos.is_move_along_pin(*mv, pin_direction)
}

pub fn generate_legal_moves(pos: &Position) -> Vec<Move> {
    generate_legal_moves_ovi(pos)
}



pub fn move_to_uci( mv: Move) -> String {
        let from = mv.from();
        let to = mv.to();
        let from_str = format!("{}{}", (b'a' + from % 8) as char, from / 8 + 1);
        let to_str = format!("{}{}", (b'a' + to % 8) as char, to / 8 + 1);
        
        let mut result = format!("{}{}", from_str, to_str);
        
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








#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_legal_moves_starting_position() {
        let pos = Position::new();
        let moves = generate_legal_moves(&pos);
        assert_eq!(moves.len(), 20);
    }
    
   
    
   
}