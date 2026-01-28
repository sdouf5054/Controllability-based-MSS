# Controllability-based-MSS

**Question**: Can we estimate time-varying controllability of vocal components to separate better?  
**Goal**: ISMIR 2026 submission (in progress)  
**Status**: Track A pipeline implemented (difficulty/proxies/cues + analysis) — Jan 2026

## Track A (current)
Pipeline (tables):
manifest → segments → separation → beat tracking → difficulty → proxies → cues → analysis

### What Track A produces
- **difficulty(t)**: separation difficulty estimate per segment
- **proxies(t)**: proxy features related to controllability
- **cues(t)**:
  - **r(t)** repetition cue
  - **s(t)** structure cue
  - **b(t)** beat-sync cue
  - **p(t)** processing cue
- **analysis**: summary statistics + plots for Track A outputs
