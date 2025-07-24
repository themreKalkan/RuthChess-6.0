// benches/evaluate_bench.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use RuthChessOVI::board::position::{Position,Color};
use RuthChessOVI::eval::evaluate::{
    evaluate, evaluate_fast, evaluate_lazy, evaluate_for_color,
    evaluate_complexity, evaluate_detailed, is_drawn, is_winning,
    score_to_win_probability, endgame_scale_factor, needs_careful_evaluation,
    tactical_complexity, static_exchange_estimate
};
use RuthChessOVI::board::zobrist;
use RuthChessOVI::movegen::magic;
use RuthChessOVI::board::position::init_attack_tables;
use RuthChessOVI::eval::evaluate::init_eval;

fn bench_evaluate_startpos(c: &mut Criterion) {
    zobrist::init_zobrist();
    magic::init_magics();
    init_attack_tables();
    init_eval();
    let pos = Position::startpos();
    c.bench_function("evaluate_startpos", |b| {
        b.iter(|| {
            let score = evaluate(black_box(&pos));
            black_box(score);
        });
    });
}

fn bench_evaluate_fast_startpos(c: &mut Criterion) {
    zobrist::init_zobrist();
    magic::init_magics();
    init_attack_tables();
    init_eval();
    let pos = Position::startpos();
    c.bench_function("evaluate_fast_startpos", |b| {
        b.iter(|| {
            let score = evaluate_fast(black_box(&pos));
            black_box(score);
        });
    });
}

fn bench_evaluate_lazy_startpos(c: &mut Criterion) {
    zobrist::init_zobrist();
    magic::init_magics();
    init_attack_tables();
    init_eval();
    let pos = Position::startpos();
    c.bench_function("evaluate_lazy_startpos", |b| {
        b.iter(|| {
            let score = evaluate_lazy(black_box(&pos), -1000, 1000);
            black_box(score);
        });
    });
}

fn bench_evaluate_for_color_startpos(c: &mut Criterion) {
    zobrist::init_zobrist();
    magic::init_magics();
    init_attack_tables();
    init_eval();
    let pos = Position::startpos();
    c.bench_function("evaluate_for_color_startpos", |b| {
        b.iter(|| {
            let score = evaluate_for_color(black_box(&pos), black_box(Color::White));
            black_box(score);
        });
    });
}

fn bench_evaluate_complexity_startpos(c: &mut Criterion) {
    zobrist::init_zobrist();
    magic::init_magics();
    init_attack_tables();
    init_eval();
    let pos = Position::startpos();
    c.bench_function("evaluate_complexity_startpos", |b| {
        b.iter(|| {
            let complexity = evaluate_complexity(black_box(&pos));
            black_box(complexity);
        });
    });
}

fn bench_evaluate_detailed_startpos(c: &mut Criterion) {
    zobrist::init_zobrist();
    magic::init_magics();
    init_attack_tables();
    init_eval();
    let pos = Position::startpos();
    c.bench_function("evaluate_detailed_startpos", |b| {
        b.iter(|| {
            let debug_info = evaluate_detailed(black_box(&pos));
            black_box(debug_info);
        });
    });
}

fn bench_is_drawn_startpos(c: &mut Criterion) {
    zobrist::init_zobrist();
    magic::init_magics();
    init_attack_tables();
    init_eval();
    c.bench_function("is_drawn_startpos", |b| {
        b.iter(|| {
            let result = is_drawn(black_box(0));
            black_box(result);
        });
    });
}

fn bench_is_winning_startpos(c: &mut Criterion) {
    zobrist::init_zobrist();
    magic::init_magics();
    init_attack_tables();
    init_eval();
    let pos = Position::startpos();
    c.bench_function("is_winning_startpos", |b| {
        b.iter(|| {
            let result = is_winning(black_box(0));
            black_box(result);
        });
    });
}

fn bench_score_to_win_probability(c: &mut Criterion) {
    zobrist::init_zobrist();
    magic::init_magics();
    init_attack_tables();
    init_eval();
    c.bench_function("score_to_win_probability", |b| {
        b.iter(|| {
            let prob = score_to_win_probability(black_box(100));
            black_box(prob);
        });
    });
}

fn bench_endgame_scale_factor_startpos(c: &mut Criterion) {
    let pos = Position::startpos();
    zobrist::init_zobrist();
    magic::init_magics();
    init_attack_tables();
    init_eval();
    c.bench_function("endgame_scale_factor_startpos", |b| {
        b.iter(|| {
            let factor = endgame_scale_factor(black_box(&pos));
            black_box(factor);
        });
    });
}

fn bench_needs_careful_evaluation_startpos(c: &mut Criterion) {
    zobrist::init_zobrist();
    magic::init_magics();
    init_attack_tables();
    init_eval();
    let pos = Position::startpos();
    c.bench_function("needs_careful_evaluation_startpos", |b| {
        b.iter(|| {
            let result = needs_careful_evaluation(black_box(&pos));
            black_box(result);
        });
    });
}

fn bench_tactical_complexity_startpos(c: &mut Criterion) {
    zobrist::init_zobrist();
    magic::init_magics();
    init_attack_tables();
    init_eval();
    let pos = Position::startpos();
    c.bench_function("tactical_complexity_startpos", |b| {
        b.iter(|| {
            let complexity = tactical_complexity(black_box(&pos));
            black_box(complexity);
        });
    });
}

fn bench_static_exchange_estimate_startpos(c: &mut Criterion) {
    zobrist::init_zobrist();
    magic::init_magics();
    init_attack_tables();
    init_eval();
    let pos = Position::startpos();
    c.bench_function("static_exchange_estimate_startpos", |b| {
        b.iter(|| {
            // Evaluate for a central square (e4)
            let see = static_exchange_estimate(black_box(&pos), black_box(28));
            black_box(see);
        });
    });
}

criterion_group!(
    name = evaluate_benches;
    config = Criterion::default()
        .sample_size(1000)
        .measurement_time(std::time::Duration::from_secs(10));
    targets =
        bench_evaluate_startpos,
        bench_evaluate_fast_startpos,
        bench_evaluate_lazy_startpos,
        bench_evaluate_for_color_startpos,
        bench_evaluate_complexity_startpos,
        bench_evaluate_detailed_startpos,
        bench_is_drawn_startpos,
        bench_is_winning_startpos,
        bench_score_to_win_probability,
        bench_endgame_scale_factor_startpos,
        bench_needs_careful_evaluation_startpos,
        bench_tactical_complexity_startpos,
        bench_static_exchange_estimate_startpos,
);

criterion_main!(evaluate_benches);