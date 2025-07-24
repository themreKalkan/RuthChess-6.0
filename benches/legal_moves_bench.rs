use criterion::{black_box, criterion_group, criterion_main, Criterion};
use RuthChessOVI::board::position::Position;
use RuthChessOVI::movegen::legal_moves::generate_legal_moves;
use RuthChessOVI::movegen::legal_moves::generate_legal_moves_ovi;
use RuthChessOVI::board::zobrist::init_zobrist;
use RuthChessOVI::movegen::magic::all_attacks_for_king;
use RuthChessOVI::board::position::Color;



fn movegen_bench(c: &mut Criterion) {
    init_zobrist();
    RuthChessOVI::movegen::magic::init_magics();
    c.bench_function("attack founder", |b| {
        b.iter(|| {
            let pos = black_box(Position::startpos());
            let moves = all_attacks_for_king(&pos, Color::White);
            black_box(moves);
        });
    });
}

fn movegen_bench_ovi(c: &mut Criterion) {
    RuthChessOVI::movegen::magic::init_magics();
    let pos = black_box(Position::startpos());

    c.bench_function("movegen_bench_ovi", |b| {
        b.iter(|| {
            let moves = generate_legal_moves_ovi(&pos);
            black_box(moves);
        });
    });
}




criterion_group!(
    name = movegen_benches;
    config = Criterion::default()
        .sample_size(10000);
    targets =
        movegen_bench_ovi,
);
criterion_main!(movegen_benches);
