<!--
 * @Description: 
 * @Date: 2026-08-18 13:58:12
 * @Author: Yaoquan Ma
-->
# VM Benchmarks

This directory stores virtual machine benchmark outputs used to compare the
performance of the normal VM and the TDX VM environments.

## Files

- perf_bench_all_tdx.txt: general performance results from the TDX VM.
- perf_bench_all_no_tdx.txt: general performance results from the normal VM.
- mem_benchmark_tdx.txt: memory performance test results from the TDX VM.
- mem_benchmark_no_tdx.txt: memory performance test results from the normal VM.

## Purpose

These benchmarks measure performance differences between the two VM environments independently of the distributed VLM inference experiments. For a fair comparison, the normal VM and TDX VM should use equivalent CPU, memory, kernel, software and cloud configurations wherever possible.

The corresponding environment check results are stored in the `../env/` directory.
