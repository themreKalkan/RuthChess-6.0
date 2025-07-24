use crate::board::bitboard::{Bitboard, popcount, square_mask, EMPTY};
use crate::board::position::Position;
use crate::board::position::{Move, PieceType, MoveType, Color};

const ROOK_MAGICS: [u64; 64] = [
    0x8a80104000800020, 0x140002000100040, 0x2801880a0017001, 0x100081001000420,
    0x200020010080420, 0x3001c0002010008, 0x8480008002000100, 0x2080088004402900,
    0x800098204000, 0x2024401000200040, 0x100802000801000, 0x120800800801000,
    0x208808088000400, 0x2802200800400, 0x2200800100020080, 0x801000060821100,
    0x80044006422000, 0x100808020004000, 0x12108a0010204200, 0x140848010000802,
    0x481828014002800, 0x8094004002004100, 0x4010040010010802, 0x20008806104,
    0x100400080208000, 0x2040002120081000, 0x21200680100081, 0x20100080080080,
    0x2000a00200410, 0x20080800400, 0x80088400100102, 0x80004600042881,
    0x4040008040800020, 0x440003000200801, 0x4200011004500, 0x188020010100100,
    0x14800401802800, 0x2080040080800200, 0x124080204001001, 0x200046502000484,
    0x480400080088020, 0x1000422010034000, 0x30200100110040, 0x100021010009,
    0x2002080100110004, 0x202008004008002, 0x20020004010100, 0x2048440040820001,
    0x101002200408200, 0x40802000401080, 0x4008142004410100, 0x2060820c0120200,
    0x1001004080100, 0x20c020080040080, 0x2935610830022400, 0x44440041009200,
    0x280001040802101, 0x2100190040002085, 0x80c0084100102001, 0x4024081001000421,
    0x20030a0244872, 0x12001008414402, 0x2006104900a0804, 0x1004081002402,
];

const BISHOP_MAGICS: [u64; 64] = [
    0x40040844404084, 0x2004208a004208, 0x10190041080202, 0x108060845042010,
    0x581104180800210, 0x2112080446200010, 0x1080820820060210, 0x3c0808410220200,
    0x4050404440404, 0x21001420088, 0x24d0080801082102, 0x1020a0a020400,
    0x40308200402, 0x4011002100800, 0x401484104104005, 0x801010402020200,
    0x400210c3880100, 0x404022024108200, 0x810018200204102, 0x4002801a02003,
    0x85040820080400, 0x810102c808880400, 0xe900410884800, 0x8002020480840102,
    0x220200865090201, 0x2010100a02021202, 0x152048408022401, 0x20080002081110,
    0x4001001021004000, 0x800040400a011002, 0xe4004081011002, 0x1c004001012080,
    0x8004200962a00220, 0x8422100208500202, 0x2000402200300c08, 0x8646020080080080,
    0x80020a0200100808, 0x2010004880111000, 0x623000a080011400, 0x42008c0340209202,
    0x209188240001000, 0x400408a884001800, 0x110400a6080400, 0x1840060a44020800,
    0x90080104000041, 0x201011000808101, 0x1a2208080504f080, 0x8012020600211212,
    0x500861011240000, 0x180806108200800, 0x4000020e01040044, 0x300000261044000a,
    0x802241102020002, 0x20906061210001, 0x5a84841004010310, 0x4010801011c04,
    0xa010109502200, 0x4a02012000, 0x500201010098b028, 0x8040002811040900,
    0x28000010020204, 0x6000020202d0240, 0x8918844842082200, 0x4010011029020020,
];

const ROOK_BITS: [u8; 64] = [
    12, 11, 11, 11, 11, 11, 11, 12,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    12, 11, 11, 11, 11, 11, 11, 12,
];

const BISHOP_BITS: [u8; 64] = [
    6, 5, 5, 5, 5, 5, 5, 6,
    5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5,
    6, 5, 5, 5, 5, 5, 5, 6,
];

static mut ROOK_ATTACKS: [Vec<Bitboard>; 64] = [const { Vec::new() }; 64];
static mut BISHOP_ATTACKS: [Vec<Bitboard>; 64] = [const { Vec::new() }; 64];
static mut ROOK_MASKS: [Bitboard; 64] = [0; 64];
static mut BISHOP_MASKS: [Bitboard; 64] = [0; 64];

static mut KNIGHT_ATTACKS: [Bitboard; 64] = [0; 64];
static mut KING_ATTACKS: [Bitboard; 64] = [0; 64];
static mut PAWN_ATTACKS: [[Bitboard; 64]; 2] = [[0; 64]; 2];

pub fn init_magics() {
    unsafe {
        for sq in 0..64 {
            ROOK_MASKS[sq] = rook_mask(sq as u8);
            BISHOP_MASKS[sq] = bishop_mask(sq as u8);
            
            let rook_size = 1 << ROOK_BITS[sq];
            let bishop_size = 1 << BISHOP_BITS[sq];
            
            ROOK_ATTACKS[sq] = vec![0; rook_size];
            BISHOP_ATTACKS[sq] = vec![0; bishop_size];
            
            init_slider_attacks(sq as u8, true);
            init_slider_attacks(sq as u8, false);
        }
        
        for sq in 0..64 {
            KNIGHT_ATTACKS[sq] = compute_knight_attacks(sq as u8);
            KING_ATTACKS[sq] = compute_king_attacks(sq as u8);
            PAWN_ATTACKS[0][sq] = compute_pawn_attacks(sq as u8, Color::White);
            PAWN_ATTACKS[1][sq] = compute_pawn_attacks(sq as u8, Color::Black);
        }
    }
}

#[inline(always)]
pub fn get_rook_attacks(square: u8, occupancy: Bitboard) -> Bitboard {
    unsafe {
        let sq_idx = square as usize;
        let masked_occ = occupancy & ROOK_MASKS[sq_idx];
        let index = (masked_occ.wrapping_mul(ROOK_MAGICS[sq_idx]) >> (64 - ROOK_BITS[sq_idx])) as usize;
        ROOK_ATTACKS[sq_idx][index]
    }
}

#[inline(always)]
pub fn get_bishop_attacks(square: u8, occupancy: Bitboard) -> Bitboard {
    unsafe {
        let sq_idx = square as usize;
        let masked_occ = occupancy & BISHOP_MASKS[sq_idx];
        let index = (masked_occ.wrapping_mul(BISHOP_MAGICS[sq_idx]) >> (64 - BISHOP_BITS[sq_idx])) as usize;
        BISHOP_ATTACKS[sq_idx][index]
    }
}

#[inline(always)]
pub fn get_queen_attacks(square: u8, occupancy: Bitboard) -> Bitboard {
    get_rook_attacks(square, occupancy) | get_bishop_attacks(square, occupancy)
}

#[inline(always)]
pub fn get_knight_attacks(square: u8) -> Bitboard {
    unsafe { KNIGHT_ATTACKS[square as usize] }
}

#[inline(always)]
pub fn get_king_attacks(square: u8) -> Bitboard {
    if(square == 64){
        return EMPTY;
    }
    unsafe { KING_ATTACKS[square as usize] }
}

#[inline(always)]
pub fn get_pawn_attacks(square: u8, color: Color) -> Bitboard {
    unsafe { PAWN_ATTACKS[color as usize][square as usize] }
}

pub fn compute_knight_attacks(sq: u8) -> Bitboard {
    let mut attacks = 0;
    let rank = sq / 8;
    let file = sq % 8;
    
    const KNIGHT_MOVES: [(i8, i8); 8] = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1)
    ];
    
    for &(dr, df) in &KNIGHT_MOVES {
        let nr = rank as i8 + dr;
        let nf = file as i8 + df;
        
        if nr >= 0 && nr < 8 && nf >= 0 && nf < 8 {
            attacks |= 1 << (nr * 8 + nf);
        }
    }
    
    attacks
}

pub fn compute_king_attacks(sq: u8) -> Bitboard {
    let mut attacks = 0;
    let rank = sq / 8;
    let file = sq % 8;
    
    for dr in -1..=1 {
        for df in -1..=1 {
            if dr == 0 && df == 0 { continue; }
            
            let nr = rank as i8 + dr;
            let nf = file as i8 + df;
            
            if nr >= 0 && nr < 8 && nf >= 0 && nf < 8 {
                attacks |= 1 << (nr * 8 + nf);
            }
        }
    }
    
    attacks
}

fn compute_pawn_attacks(sq: u8, color: Color) -> Bitboard {
    let mut attacks = 0;
    let rank = sq / 8;
    let file = sq % 8;
    
    let push_dir = match color {
        Color::White => 1i8,
        Color::Black => -1i8,
    };
    
    let nr = rank as i8 + push_dir;
    if nr >= 0 && nr < 8 {
        if file > 0 {
            attacks |= 1 << (nr * 8 + file as i8 - 1);
        }
        if file < 7 {
            attacks |= 1 << (nr * 8 + file as i8 + 1);
        }
    }
    
    attacks
}

fn init_slider_attacks(square: u8, is_rook: bool) {
    let mask = if is_rook {
        rook_mask(square)
    } else {
        bishop_mask(square)
    };
    
    let bits = if is_rook {
        ROOK_BITS[square as usize]
    } else {
        BISHOP_BITS[square as usize]
    };
    
    let permutations = 1 << bits;
    let magic = if is_rook {
        ROOK_MAGICS[square as usize]
    } else {
        BISHOP_MAGICS[square as usize]
    };
    
    for i in 0..permutations {
        let occupancy = generate_occupancy(i, mask, bits);
        let attacks = if is_rook {
            rook_attacks_slow(square, occupancy)
        } else {
            bishop_attacks_slow(square, occupancy)
        };
        
        let index = (occupancy.wrapping_mul(magic) >> (64 - bits)) as usize;
        
        unsafe {
            if is_rook {
                ROOK_ATTACKS[square as usize][index] = attacks;
            } else {
                BISHOP_ATTACKS[square as usize][index] = attacks;
            }
        }
    }
}

fn rook_mask(square: u8) -> Bitboard {
    let mut mask = 0;
    let rank = square / 8;
    let file = square % 8;
    
    for f in 1..7 {
        if f != file {
            mask |= 1 << (rank * 8 + f);
        }
    }
    
    for r in 1..7 {
        if r != rank {
            mask |= 1 << (r * 8 + file);
        }
    }
    
    mask
}

fn bishop_mask(square: u8) -> Bitboard {
    let mut mask = 0;
    let rank = square as i8 / 8;
    let file = square as i8 % 8;
    
    const DIRECTIONS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
    
    for &(dr, df) in &DIRECTIONS {
        let mut r = rank + dr;
        let mut f = file + df;
        
        while r > 0 && r < 7 && f > 0 && f < 7 {
            mask |= 1 << (r * 8 + f);
            r += dr;
            f += df;
        }
    }
    
    mask
}

#[inline(always)]
fn generate_occupancy(index: u32, mask: Bitboard, bits: u8) -> Bitboard {
    let mut occupancy = 0;
    let mut temp_mask = mask;
    
    for i in 0..bits {
        let square = temp_mask.trailing_zeros() as u8;
        temp_mask &= temp_mask - 1;
        
        if (index & (1 << i)) != 0 {
            occupancy |= 1 << square;
        }
    }
    
    occupancy
}

fn rook_attacks_slow(square: u8, occupancy: Bitboard) -> Bitboard {
    let mut attacks = 0;
    let rank = square / 8;
    let file = square % 8;
    
    for r in (rank + 1)..8 {
        let sq = r * 8 + file;
        attacks |= 1 << sq;
        if (occupancy & (1 << sq)) != 0 { break; }
    }
    
    for r in (0..rank).rev() {
        let sq = r * 8 + file;
        attacks |= 1 << sq;
        if (occupancy & (1 << sq)) != 0 { break; }
    }
    
    for f in (file + 1)..8 {
        let sq = rank * 8 + f;
        attacks |= 1 << sq;
        if (occupancy & (1 << sq)) != 0 { break; }
    }
    
    for f in (0..file).rev() {
        let sq = rank * 8 + f;
        attacks |= 1 << sq;
        if (occupancy & (1 << sq)) != 0 { break; }
    }
    
    attacks
}

fn bishop_attacks_slow(square: u8, occupancy: Bitboard) -> Bitboard {
    let mut attacks = 0;
    let rank = square as i8 / 8;
    let file = square as i8 % 8;
    
    const DIRECTIONS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
    
    for &(dr, df) in &DIRECTIONS {
        let mut r = rank + dr;
        let mut f = file + df;
        
        while r >= 0 && r < 8 && f >= 0 && f < 8 {
            let sq = (r * 8 + f) as u8;
            attacks |= 1 << sq;
            if (occupancy & (1 << sq)) != 0 { break; }
            r += dr;
            f += df;
        }
    }
    
    attacks
}

#[inline(always)]
pub fn all_attacks_for_king(pos: &Position, by_color: Color) -> Bitboard {
    let mut attacks = EMPTY;
    let occupancy = pos.occupied_without_king(by_color.opposite());
    let king_sq = pos.king_square(by_color);

    let pawns = pos.pieces_colored(PieceType::Pawn, by_color);
    attacks |= pawn_attacks(pawns, by_color);

    let knights = pos.pieces_colored(PieceType::Knight, by_color);
    attacks |= knight_attacks(knights);

    let bishops = pos.pieces_colored(PieceType::Bishop, by_color);
    attacks |= bishop_attacks(bishops, occupancy);

    let rooks = pos.pieces_colored(PieceType::Rook, by_color);
    attacks |= rook_attacks(rooks, occupancy);

    let queens = pos.pieces_colored(PieceType::Queen, by_color);
    attacks |= queen_attacks(queens, occupancy);

                attacks |= get_king_attacks(king_sq);


    attacks
}

#[inline(always)]
pub fn all_attacks(pos: &Position, by_color: Color) -> Bitboard {
    let mut attacks = EMPTY;
    let occupancy = pos.all_pieces();
    let king_sq = pos.king_square(by_color);

    let pawns = pos.pieces_colored(PieceType::Pawn, by_color);
    attacks |= pawn_attacks(pawns, by_color);

    let knights = pos.pieces_colored(PieceType::Knight, by_color);
    attacks |= knight_attacks(knights);

    let bishops = pos.pieces_colored(PieceType::Bishop, by_color);
    attacks |= bishop_attacks(bishops, occupancy);

    let rooks = pos.pieces_colored(PieceType::Rook, by_color);
    attacks |= rook_attacks(rooks, occupancy);

    let queens = pos.pieces_colored(PieceType::Queen, by_color);
    attacks |= queen_attacks(queens, occupancy);

                attacks |= get_king_attacks(king_sq);


    attacks
}

pub fn king_attacks(occ:Bitboard) -> Bitboard {
    let mut attacks = EMPTY;
    let mut bb = occ;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        attacks |= get_king_attacks(sq);
        bb &= bb - 1;
    }
    attacks
}

#[inline(always)]
pub fn pawn_attacks(pawns: Bitboard, color: Color) -> Bitboard {
    let mut attacks = EMPTY;
    let mut bb = pawns;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        attacks |= get_pawn_attacks(sq, color);
        bb &= bb - 1;
    }
    attacks
}

#[inline(always)]
pub fn knight_attacks(knights: Bitboard) -> Bitboard {
    let mut attacks = EMPTY;
    let mut bb = knights;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        attacks |= get_knight_attacks(sq);
        bb &= bb - 1;
    }
    attacks
}

#[inline(always)]
pub fn bishop_attacks(bishops: Bitboard, occupancy: Bitboard) -> Bitboard {
    let mut attacks = EMPTY;
    let mut bb = bishops;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        attacks |= get_bishop_attacks(sq, occupancy);
        bb &= bb - 1;
    }
    attacks
}

#[inline(always)]
pub fn rook_attacks(rooks: Bitboard, occupancy: Bitboard) -> Bitboard {
    let mut attacks = EMPTY;
    let mut bb = rooks;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        attacks |= get_rook_attacks(sq, occupancy);
        bb &= bb - 1;
    }
    attacks
}

#[inline(always)]
pub fn queen_attacks(queens: Bitboard, occupancy: Bitboard) -> Bitboard {
    let mut attacks = EMPTY;
    let mut bb = queens;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        attacks |= get_queen_attacks(sq, occupancy);
        bb &= bb - 1;
    }
    attacks
}

#[inline(always)]
pub fn add_moves_from_bitboard(from: u8, targets: Bitboard, moves: &mut Vec<Move>) {
    let mut targets_bb = targets;
    while targets_bb != 0 {
        let to = targets_bb.trailing_zeros() as u8;
        targets_bb &= targets_bb - 1;
        moves.push(Move::new(from, to, MoveType::Normal, PieceType::None));
    }
}

