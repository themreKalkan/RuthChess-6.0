# RuthChess

**A UCI chess engine written in Rust with OVI move generation**

By Emre Kalkan

## 🚀 What's Special

### **OVI (Optimized Verification Integration)**
RuthChess uses a unique move generation system that combines pseudo-legal move generation with attack detection in a single pass, reducing computational overhead.

### **Key Features**
- **Smart Pin Detection**: Real-time tracking of pinned pieces and legal moves
- **Parallel Search**: Multi-threaded alpha-beta with configurable threads
- **Advanced Evaluation**: 9-component evaluation system including king safety, threats, and space control
- **Magic Bitboards**: Fast move generation for sliding pieces

## 🔧 Installation

```bash
git clone https://github.com/yourusername/RuthChessOVI.git
cd RuthChessOVI
cargo build --release
./target/release/RuthChessOVI
```

## ⚙️ UCI Options

```
setoption name Hash value 256        # Hash table size (MB)
setoption name Threads value 4       # Search threads
setoption name Ponder value true     # Enable pondering
```

## 🧪 Testing

```bash
cargo bench                          # Run all benchmarks
cargo test                          # Run unit tests
```

## 📈 Performance

- **Nodes per second**: Optimized for modern multi-core systems
- **Memory efficient**: Aligned data structures for cache performance
- **Benchmark suite**: Built-in performance testing for all components

## 🎮 Usage

Compatible with any UCI chess GUI (Arena, Fritz, ChessBase, etc.)

```
uci
isready
position startpos moves e2e4 e7e5
go depth 10
```
