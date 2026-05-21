# SmoreFit

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://drbergman-lab.github.io/SmoreFit.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://drbergman-lab.github.io/SmoreFit.jl/dev/)
[![Build Status](https://github.com/drbergman-lab/SmoreFit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/drbergman-lab/SmoreFit.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/drbergman-lab/SmoreFit.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/drbergman-lab/SmoreFit.jl)

Posterior inference on complex model (CM) parameter space for the [Smore](https://github.com/drbergman-lab/Smore.jl) surrogate modeling ecosystem. SmoreFit takes a fitted surrogate model (SM) from [SmoreBase.jl](https://github.com/drbergman-lab/SmoreBase.jl) and real-world observational data, and builds a posterior distribution over CM parameters.

**Status: not yet implemented.** See [PRD.md](PRD.md) for the planned API.

---

## Implementation Status

> For Claude Code sessions: this section is the authoritative record of what has been built. Update it as features are completed. See [PRD.md](PRD.md) for behavioral specifications and [progress.md](progress.md) for decision rationale.

### Remaining

- [ ] `buildPosterior` — posterior on CM parameter space given real-world data + SM UQ from SmoreBase
