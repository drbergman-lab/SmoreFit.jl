# progress.md — SmoreFit.jl Session Journal

> **Purpose:** Session-level decisions, rejected approaches, and open questions.
> Unlike [PRD.md](PRD.md) (specification) and [README.md](README.md) (completion status), this file captures the *reasoning* behind decisions — things that would otherwise exist only in ended chat history.

---

## Session: Initialization — Architecture decisions relevant to SmoreFit (2026-05-19)

### Key Decisions

**SmoreFit is where real-world data enters the pipeline**
SmoreBase fits the SM to CM-generated training data and quantifies SM parameter uncertainty. SmoreFit is the next step: given real-world observational data, use the SM (within its uncertainty region) as a fast likelihood evaluator to infer a posterior over CM parameter space.

**Planned API: `buildPosterior`**
`buildPosterior(sm, realWorldData, uqResults, cmBounds; ...) -> CMPosteriorResult`. Exact signatures and types are TBD — see [PRD.md](PRD.md) for open questions.

**`ParameterPrior` (from SmoreBase) is the natural type for CM bounds**
The same `ParameterPrior` type used for SM parameter priors can hold CM parameter priors, since it supports full `Distributions.jl` distributions. This makes SmoreFit a natural consumer of SmoreBase types without duplicating the bounds concept.

### Status
Nothing implemented. Package is a stub. Implementation deferred until the SmoreBase pipeline is validated end-to-end in SmoreExamples.
