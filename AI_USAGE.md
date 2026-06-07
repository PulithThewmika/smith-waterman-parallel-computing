# AI Usage Declaration — SE3082 Assignment 03

Author: Pulith Thewmika (IT23656338)

The assignment permits the use of AI provided it is cited together with the
prompts used. This file documents that use. **You should edit this file to
reflect your own actual prompts and how you reviewed/modified the output.**

## Tool used
- Claude (Anthropic), used as a coding and explanation assistant during
  development of the four Smith–Waterman implementations and the
  benchmarking/plotting scripts.

## Representative prompts
1. "Explain the Smith–Waterman local alignment recurrence and why cells on the
   same anti-diagonal are independent and can be parallelised."
2. "Help me design and write a serial C baseline for Smith–Waterman that fills
   the scoring matrix and reports the max score and timing."
3. "Write an OpenMP version using an anti-diagonal wavefront; help me improve it
   to a tiled wavefront when the naive version did not scale."
4. "Write an MPI version that partitions columns across ranks and pipelines the
   boundary column between neighbours; collect the global max with MPI_Reduce."
5. "Write a CUDA version that launches one kernel per anti-diagonal with the
   block size as a command-line argument, using atomicMax for the running max."
6. "Write a bash benchmark runner and a Python (matplotlib) script that produces
   threads/processes/block-size vs time and speedup graphs."

## How the output was used and verified
- I read and understood each implementation before using it, adding/adjusting
  comments to match my own understanding.
- I verified correctness by confirming all four implementations produce the
  SAME maximum alignment score for the same sequences/seed.
- I ran all benchmarks myself on my own hardware and produced the graphs and
  written analysis in the report independently.

## What is my own work
- All performance measurements, screenshots, graph interpretation, the written
  report (Part C), design justifications, and the final integration/testing on
  my machine.
