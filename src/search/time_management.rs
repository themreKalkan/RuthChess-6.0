use std::time::{Instant, Duration};

#[derive(Debug, Clone)]
pub struct TimeManager {
    start_time: Instant,
    allocated_time: Duration,
    move_time: Option<Duration>,
    max_nodes: Option<u64>,
    infinite: bool,
    white_time: Option<u32>,
    black_time: Option<u32>,
    white_increment: Option<u32>,
    black_increment: Option<u32>,
    moves_to_go: Option<u32>,
    is_white_to_move: bool,
}

impl TimeManager {
    pub fn infinite() -> Self {
        Self {
            start_time: Instant::now(),
            allocated_time: Duration::from_secs(3600 * 24),
            move_time: None,
            max_nodes: None,
            infinite: true,
            white_time: None,
            black_time: None,
            white_increment: None,
            black_increment: None,
            moves_to_go: None,
            is_white_to_move: true,
        }
    }

    pub fn is_infinite(&self) -> bool {
        self.infinite
    }

    pub fn new(
        wtime: Option<u32>,
        btime: Option<u32>,
        movetime: Option<u32>,
        moves_to_go: Option<u32>,
        max_nodes: Option<u64>,
        infinite: bool,
    ) -> Self {
        if infinite {
            return Self::infinite();
        }

        let start_time = Instant::now();
        let allocated_time = if let Some(mt) = movetime {
            Duration::from_millis(mt as u64)
        } else {
            Self::calculate_time_allocation(wtime, btime, moves_to_go, true)
        };

        Self {
            start_time,
            allocated_time,
            move_time: movetime.map(|t| Duration::from_millis(t as u64)),
            max_nodes,
            infinite,
            white_time: wtime,
            black_time: btime,
            white_increment: None,
            black_increment: None,
            moves_to_go,
            is_white_to_move: true,
        }
    }

    pub fn new_with_increment(
        wtime: Option<u32>,
        btime: Option<u32>,
        movetime: Option<u32>,
        winc: Option<u32>,
        binc: Option<u32>,
        moves_to_go: Option<u32>,
        max_nodes: Option<u64>,
        infinite: bool,
        is_white_to_move: bool,
    ) -> Self {
        if infinite {
            return Self::infinite();
        }

        let start_time = Instant::now();
        let allocated_time = if let Some(mt) = movetime {
            Duration::from_millis(mt as u64)
        } else {
            Self::calculate_time_allocation_with_increment(
                wtime, btime, winc, binc, moves_to_go, is_white_to_move
            )
        };

        Self {
            start_time,
            allocated_time,
            move_time: movetime.map(|t| Duration::from_millis(t as u64)),
            max_nodes,
            infinite,
            white_time: wtime,
            black_time: btime,
            white_increment: winc,
            black_increment: binc,
            moves_to_go,
            is_white_to_move,
        }
    }

    fn calculate_time_allocation(
        wtime: Option<u32>,
        btime: Option<u32>,
        moves_to_go: Option<u32>,
        is_white_to_move: bool,
    ) -> Duration {
        let my_time = if is_white_to_move { wtime } else { btime };
        
        match (my_time, moves_to_go) {
            (Some(time), Some(mtg)) => {
                let moves_remaining = mtg.max(1);
                let base_time = time / (moves_remaining + 2);
                Duration::from_millis(base_time.min(time / 3) as u64)
            }
            (Some(time), None) => {
                let base_time = if time > 300_000 {
                    time / 40
                } else if time > 60_000 {
                    time / 30
                } else if time > 10_000 {
                    time / 20
                } else {
                    time / 10
                };
                Duration::from_millis(base_time.min(time / 2) as u64)
            }
            _ => {
                Duration::from_millis(10_000)
            }
        }
    }

    fn calculate_time_allocation_with_increment(
        wtime: Option<u32>,
        btime: Option<u32>,
        winc: Option<u32>,
        binc: Option<u32>,
        moves_to_go: Option<u32>,
        is_white_to_move: bool,
    ) -> Duration {
        let my_time = if is_white_to_move { wtime } else { btime };
        let my_increment = if is_white_to_move { winc } else { binc };
        
        match (my_time, my_increment, moves_to_go) {
            (Some(time), Some(inc), Some(mtg)) => {
                let moves_remaining = mtg.max(1);
                let total_time = time + (inc * moves_remaining / 2);
                let base_time = total_time / (moves_remaining + 2);
                Duration::from_millis(base_time.min(time / 3) as u64)
            }
            (Some(time), Some(inc), None) => {
                let effective_time = time + inc;
                let base_time = if effective_time > 300_000 {
                    (time / 40) + (inc * 8 / 10)
                } else if effective_time > 60_000 {
                    (time / 30) + (inc * 7 / 10)
                } else if effective_time > 10_000 {
                    (time / 20) + (inc * 6 / 10)
                } else {
                    (time / 10) + (inc / 2)
                };
                Duration::from_millis(base_time.min(effective_time / 2) as u64)
            }
            (Some(time), None, mtg) => {
                Self::calculate_time_allocation(Some(time), None, mtg, is_white_to_move)
            }
            _ => {
                Duration::from_millis(10_000)
            }
        }
    }

    pub fn should_stop(&self, start: Instant) -> bool {
        if self.infinite {
            return false;
        }

        if let Some(_max) = self.max_nodes {
        }

        let elapsed = start.elapsed();
        elapsed >= self.allocated_time
    }

    pub fn can_extend(&self, start: Instant, extension_factor: f64) -> bool {
        if self.infinite {
            return true;
        }

        let elapsed = start.elapsed();
        let extended_time = Duration::from_millis(
            (self.allocated_time.as_millis() as f64 * extension_factor) as u64
        );
        
        elapsed < extended_time
    }

    pub fn allocated_ms(&self) -> u64 {
        self.allocated_time.as_millis() as u64
    }

    pub fn remaining_time_ms(&self) -> Option<u32> {
        if self.is_white_to_move {
            self.white_time
        } else {
            self.black_time
        }
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    pub fn set_side_to_move(&mut self, is_white: bool) {
        if self.is_white_to_move != is_white {
            self.is_white_to_move = is_white;
            if !self.infinite && self.move_time.is_none() {
                self.allocated_time = Self::calculate_time_allocation_with_increment(
                    self.white_time,
                    self.black_time,
                    self.white_increment,
                    self.black_increment,
                    self.moves_to_go,
                    is_white,
                );
                self.start_time = Instant::now();
            }
        }
    }
}