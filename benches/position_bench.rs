use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, 
    Throughput, SamplingMode
};
use RuthChessOVI::board::position::{Position, PieceType, Color};
use std::time::Duration;
use RuthChessOVI::movegen::legal_moves::generate_legal_moves;

use RuthChessOVI::board::zobrist;
use RuthChessOVI::movegen::magic;
use RuthChessOVI::board::position::init_attack_tables;
use RuthChessOVI::eval::evaluate::init_eval;

fn position_startpos_benchmark(c: &mut Criterion) {
    zobrist::init_zobrist();
    magic::init_magics();
    init_attack_tables();
    init_eval();
    let mut group = c.benchmark_group("Position Initialization");
    group.sampling_mode(SamplingMode::Flat)
         .sample_size(1000);
    
    // Log scale için plot config sadece group'ta kullanılabilir
    group.plot_config(criterion::PlotConfiguration::default()
        .summary_scale(AxisScale::Logarithmic));
    
    group.bench_function("startpos", |b| {
        b.iter(|| {
            let pos = black_box(Position::startpos());
            black_box(pos);
        });
    });
    
    group.bench_function("from_fen", |b| {
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        b.iter(|| {
            let pos = black_box(Position::from_fen(fen).unwrap());
            black_box(pos);
        });
    });
    
    group.finish();
}

fn piece_operations_benchmark(c: &mut Criterion) {
    zobrist::init_zobrist();
    magic::init_magics();
    init_attack_tables();
    init_eval();
    let mut group = c.benchmark_group("Piece Operations");
    group.throughput(Throughput::Elements(64))
         .measurement_time(Duration::from_secs(15))
         .plot_config(criterion::PlotConfiguration::default()
             .summary_scale(AxisScale::Linear));
    
    group.bench_function("place_piece", |b| {
        let mut pos = Position::startpos();
        b.iter(|| {
            for sq in 0..64 {
                pos.place_piece(sq, PieceType::Knight, Color::White);
                black_box(&mut pos);
            }
        });
    });
    
    group.bench_function("remove_piece", |b| {
        let mut pos = Position::startpos();
        for sq in 0..64 {
            pos.place_piece(sq, PieceType::Knight, Color::White);
        }
        b.iter(|| {
            for sq in 0..64 {
                pos.remove_piece(sq);
                black_box(&mut pos);
            }
        });
    });
    
    group.finish();
}

fn move_operations_benchmark(c: &mut Criterion) {
    zobrist::init_zobrist();
    magic::init_magics();
    init_attack_tables();
    init_eval();
    let positions = vec![
        ("startpos", Position::startpos()),
        ("complex", Position::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap()),
    ];
    
    let mut group = c.benchmark_group("Move Operations");
    group.sample_size(2000)
         .confidence_level(0.99)
         .plot_config(criterion::PlotConfiguration::default()
             .summary_scale(AxisScale::Linear));
    
    for (name, pos) in positions {
        let moves = generate_legal_moves(&pos);
        group.throughput(Throughput::Elements(moves.len() as u64));
        
        group.bench_with_input(
            BenchmarkId::new("make_unmake", name),
            &moves,
            |b, moves| {
                b.iter(|| {
                    let mut pos_clone = pos.clone();
                    for mv in moves {
                        pos_clone.make_move(*mv);
                        pos_clone.unmake_move(*mv);
                        black_box(&mut pos_clone);
                    }
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    name = position_benches;
    config = Criterion::default()
        .sample_size(1000)
        .measurement_time(Duration::from_secs(15))
        .noise_threshold(0.02);
    targets =
        position_startpos_benchmark,
        piece_operations_benchmark,
        move_operations_benchmark,
);
criterion_main!(position_benches);