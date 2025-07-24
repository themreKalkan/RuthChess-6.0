use criterion::{black_box, criterion_group, criterion_main, Criterion};
use RuthChessOVI::board::bitboard;

// GERÇEK TEK OPERASYON BENCHMARK'LARI

// Popcount - tek operasyon
fn popcount_benchmark(c: &mut Criterion) {
    let bb = 0xAAAA_AAAA_AAAA_AAAA; // Test value
    
    c.bench_function("popcount", |b| {
        b.iter(|| {
            // TEK operasyon
            black_box(bitboard::popcount(black_box(bb)))
        })
    });
}

// Pop LSB - tek operasyon  
fn pop_lsb_benchmark(c: &mut Criterion) {
    c.bench_function("pop_lsb", |b| {
        b.iter(|| {
            // TEK operasyon - her iterasyonda fresh value
            let mut bb = black_box(0x0000_0000_0000_0008);
            black_box(bitboard::pop_lsb(&mut bb))
        })
    });
}

// Has bit - tek operasyon
fn has_bit_benchmark(c: &mut Criterion) {
    let bb = 0x8000_0000_0000_0001;
    let square = 32u8;
    
    c.bench_function("has_bit", |b| {
        b.iter(|| {
            // TEK operasyon
            black_box(bitboard::has_bit(black_box(bb), black_box(square)))
        })
    });
}

// Set bit - tek operasyon
fn set_bit_benchmark(c: &mut Criterion) {
    c.bench_function("set_bit", |b| {
        b.iter(|| {
            // TEK operasyon - her iterasyonda fresh value
            let mut bb = black_box(0u64);
            bitboard::set_bit(&mut bb, black_box(32));
            black_box(bb)
        })
    });
}

// Clear bit - tek operasyon
fn clear_bit_benchmark(c: &mut Criterion) {
    c.bench_function("clear_bit", |b| {
        b.iter(|| {
            // TEK operasyon - her iterasyonda fresh value
            let mut bb = black_box(0xFFFF_FFFF_FFFF_FFFF);
            bitboard::clear_bit(&mut bb, black_box(32));
            black_box(bb)
        })
    });
}

// Toggle bit - tek operasyon
fn toggle_bit_benchmark(c: &mut Criterion) {
    c.bench_function("toggle_bit", |b| {
        b.iter(|| {
            // TEK operasyon - her iterasyonda fresh value
            let mut bb = black_box(0x5555_5555_5555_5555);
            bitboard::toggle_bit(&mut bb, black_box(32));
            black_box(bb)
        })
    });
}

// Square mask - tek operasyon
fn square_mask_benchmark(c: &mut Criterion) {
    let square = 32u8;
    
    c.bench_function("square_mask", |b| {
        b.iter(|| {
            // TEK operasyon
            black_box(bitboard::square_mask(black_box(square)))
        })
    });
}

// LSB - tek operasyon
fn lsb_benchmark(c: &mut Criterion) {
    let bb = 0x0000_0000_0000_1008;
    
    c.bench_function("lsb", |b| {
        b.iter(|| {
            // TEK operasyon
            black_box(bitboard::lsb(black_box(bb)))
        })
    });
}

// Reset LSB - tek operasyon
fn reset_lsb_benchmark(c: &mut Criterion) {
    let bb = 0x0000_0000_0000_1008;
    
    c.bench_function("reset_lsb", |b| {
        b.iter(|| {
            // TEK operasyon
            black_box(bitboard::reset_lsb(black_box(bb)))
        })
    });
}

// MSB - tek operasyon
fn msb_benchmark(c: &mut Criterion) {
    let bb = 0x0000_0000_0000_1008;
    
    c.bench_function("msb", |b| {
        b.iter(|| {
            // TEK operasyon
            black_box(bitboard::msb(black_box(bb)))
        })
    });
}

// Shift North - tek operasyon
fn shift_north_benchmark(c: &mut Criterion) {
    let bb = 0x0000_0008_1000_0000;
    
    c.bench_function("shift_north", |b| {
        b.iter(|| {
            // TEK operasyon
            black_box(bitboard::shift_north(black_box(bb)))
        })
    });
}

// Shift South - tek operasyon
fn shift_south_benchmark(c: &mut Criterion) {
    let bb = 0x0000_0008_1000_0000;
    
    c.bench_function("shift_south", |b| {
        b.iter(|| {
            // TEK operasyon
            black_box(bitboard::shift_south(black_box(bb)))
        })
    });
}

// Shift East - tek operasyon
fn shift_east_benchmark(c: &mut Criterion) {
    let bb = 0x0000_0008_1000_0000;
    
    c.bench_function("shift_east", |b| {
        b.iter(|| {
            // TEK operasyon
            black_box(bitboard::shift_east(black_box(bb)))
        })
    });
}

// Shift West - tek operasyon
fn shift_west_benchmark(c: &mut Criterion) {
    let bb = 0x0000_0008_1000_0000;
    
    c.bench_function("shift_west", |b| {
        b.iter(|| {
            // TEK operasyon
            black_box(bitboard::shift_west(black_box(bb)))
        })
    });
}

// Get rank - tek operasyon
fn get_rank_benchmark(c: &mut Criterion) {
    let square = 35u8; // e4
    
    c.bench_function("get_rank", |b| {
        b.iter(|| {
            // TEK operasyon
            black_box(bitboard::get_rank(black_box(square)))
        })
    });
}

// Get file - tek operasyon
fn get_file_benchmark(c: &mut Criterion) {
    let square = 35u8; // e4
    
    c.bench_function("get_file", |b| {
        b.iter(|| {
            // TEK operasyon
            black_box(bitboard::get_file(black_box(square)))
        })
    });
}

// Square distance - tek operasyon
fn square_distance_benchmark(c: &mut Criterion) {
    let sq1 = 28u8; // e4
    let sq2 = 36u8; // e5
    
    c.bench_function("square_distance", |b| {
        b.iter(|| {
            // TEK operasyon
            black_box(bitboard::square_distance(black_box(sq1), black_box(sq2)))
        })
    });
}

// Square to algebraic - tek operasyon
fn square_to_algebraic_benchmark(c: &mut Criterion) {
    let square = 28u8; // e4
    
    c.bench_function("square_to_algebraic", |b| {
        b.iter(|| {
            // TEK operasyon
            black_box(bitboard::square_to_algebraic(black_box(square)))
        })
    });
}

// Algebraic to square - tek operasyon
fn algebraic_to_square_benchmark(c: &mut Criterion) {
    let algebraic = "e4";
    
    c.bench_function("algebraic_to_square", |b| {
        b.iter(|| {
            // TEK operasyon
            black_box(bitboard::algebraic_to_square(black_box(algebraic)))
        })
    });
}

// Rank mask - tek operasyon
fn rank_mask_benchmark(c: &mut Criterion) {
    let rank = 4u8;
    
    c.bench_function("rank_mask", |b| {
        b.iter(|| {
            // TEK operasyon
            black_box(bitboard::rank_mask(black_box(rank)))
        })
    });
}

// File mask - tek operasyon
fn file_mask_benchmark(c: &mut Criterion) {
    let file = 4u8;
    
    c.bench_function("file_mask", |b| {
        b.iter(|| {
            // TEK operasyon
            black_box(bitboard::file_mask(black_box(file)))
        })
    });
}

// ÇOKLU OPERASYON BENCHMARKları (karşılaştırma için)

// Pop LSB loop - 64 operasyon (orijinal benchmark)
fn pop_lsb_64_benchmark(c: &mut Criterion) {
    c.bench_function("pop_lsb_64_ops", |b| {
        b.iter(|| {
            let mut bb = black_box(0xFFFF_FFFF_FFFF_FFFF);
            let mut sum = 0u64;
            while bb != 0 {
                sum += bitboard::pop_lsb(&mut bb) as u64;
            }
            black_box(sum)
        })
    });
}

// Has bit 64 - 64 operasyon (orijinal benchmark)
fn has_bit_64_benchmark(c: &mut Criterion) {
    let bb = 0x8000_0000_0000_0001;
    
    c.bench_function("has_bit_64_ops", |b| {
        b.iter(|| {
            let mut count = 0;
            for square in 0..64 {
                if bitboard::has_bit(bb, square) {
                    count += 1;
                }
            }
            black_box(count)
        })
    });
}

// Set bit 64 - 64 operasyon (orijinal benchmark)
fn set_bit_64_benchmark(c: &mut Criterion) {
    c.bench_function("set_bit_64_ops", |b| {
        b.iter(|| {
            let mut bb = black_box(0);
            for square in 0..64 {
                bitboard::set_bit(&mut bb, square);
            }
            black_box(bb)
        })
    });
}

criterion_group!(
    name = single_op_benches;
    config = Criterion::default()
        .sample_size(10000)  // Çok fazla sample tek operasyon için
        .measurement_time(std::time::Duration::from_secs(5));
    targets = 
        popcount_benchmark,
        pop_lsb_benchmark,
        has_bit_benchmark,
        set_bit_benchmark,
        clear_bit_benchmark,
        toggle_bit_benchmark,
        square_mask_benchmark,
        lsb_benchmark,
        reset_lsb_benchmark,
        msb_benchmark,
        shift_north_benchmark,
        shift_south_benchmark,
        shift_east_benchmark,
        shift_west_benchmark,
        get_rank_benchmark,
        get_file_benchmark,
        square_distance_benchmark,
        square_to_algebraic_benchmark,
        algebraic_to_square_benchmark,
        rank_mask_benchmark,
        file_mask_benchmark
);

criterion_group!(
    name = multi_op_benches;
    config = Criterion::default()
        .sample_size(10000)
        .measurement_time(std::time::Duration::from_secs(3));
    targets = 
        pop_lsb_64_benchmark,
        has_bit_64_benchmark,
        set_bit_64_benchmark
);

criterion_main!(single_op_benches, multi_op_benches);