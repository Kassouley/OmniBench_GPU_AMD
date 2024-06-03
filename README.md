# OmniBench_GPU_AMD

OmniBench_GPU_AMD is a benchmarking tool designed for AMD GPUs. It allows you to perform correctness checks and measure the performance of various GPU kernels with different optimizations and problem dimensions.

## Prerequisites
- Compatible AMD GPU
- AMD ROCm drivers installed
- Make

## Installation

To install and compile the project, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/Kassouley/OmniBench_GPU_AMD.git
cd OmniBench_GPU_AMD
```

2. Use the omnibench.sh script or compile with:

```bash
make measure|check KERNEL=... OPT=... DIM=...
./benchmark/<kernel>/build/bin/measure|check ...
```

## Usage

OmniBench provides three main commands: check, measure, and export. Each command has specific options and arguments.

## General Usage

```bash
./omni_benchmark.sh <command> [options] <arguments>
```

## Commands

### check

Run a correctness check on a kernel with specific optimization and problem dimensions.

Usage:

```bash
./omni_benchmark.sh check [options] <kernel> <optimization> <problem_dim>
```
Options:

- -h, --help: Show help message and exit
- -v, --verbose: Enable verbose mode

Arguments:

- kernel: Kernel to use (e.g., matrixMultiply, saxpy, matrixTranspose, matrixVectorMultiply, matrixCopy)
- optimization: Optimization type (e.g., NOPT, TILE, UNROLL, STRIDE)
- problem_dim: Problem dimensions (size for vectors, size for square matrices)

### measure

Perform benchmarking on a kernel with specific optimization, benchmark type, and problem dimensions.

Usage:

```bash
./omni_benchmark.sh measure [options] <kernel> <optimization> <benchmark> <problem_dim>
```

Options:

- -h, --help: Show help message and exit
- -o, --output <f>: Save the benchmark results in the specified file
- -v, --verbose: Enable verbose mode

Arguments:

- kernel: Kernel to use (e.g., matrixMultiply, saxpy, matrixTranspose, matrixVectorMultiply, matrixCopy)
- optimization: Optimization type (e.g., NOPT, TILE, UNROLL, STRIDE)
- benchmark: Benchmark type (e.g., blockSizeVar, gridSizeVar, LdsSizeVar)
- problem_dim: Problem dimensions (size for vectors, size for square matrices)

### export

Export GPU information and CSV data to a markdown file.

Usage:

```bash
./omni_benchmark.sh export [options]
```

Options:

- -h, --help: Show help message and exit
- -o, --output <f>: Save the exported markdown file as the specified file
- -i, --input <f>: Specify the input CSV file to be converted to markdown table

## Examples

Run a correctness check

```bash
./omni_benchmark.sh check -v matrixMultiply TILE 1024
```
Perform a benchmark measurement

bash
```bash
./omni_benchmark.sh measure -o results.csv matrixMultiply TILE BlockSizeVar 1024
```
Export GPU info and CSV data to markdown

```bash
./omni_benchmark.sh export -i results.csv -o results.md
```

## Project Structure
- common/ : Contains the source code and header for the common benchmarking tool.
- benchmark/< kernel >/ : Contains the source code and header for a kernel benchmarking tool.
- README.md : Project documentation.
- omni_benchmark.sh : Main script for running benchmarks and checks.