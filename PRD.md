# Product Requirements Document — SmoreFit.jl

> **Purpose:** This document defines the complete feature set of SmoreFit in behavioral terms. It is the authoritative answer to "what should this system do?" Read this at the start of any feature session to establish alignment between intent and implementation plan.

---

## Product Overview

**Vision:** SmoreFit is where real-world observational data enters the Smore pipeline. Given a fitted surrogate model (SM) and its uncertainty quantification (from SmoreBase), SmoreFit infers a posterior distribution over complex model (CM) parameter space by comparing SM predictions to real-world observations.

**Target Users:** Computational modelers who have a fitted SM (from SmoreBase) and real-world data, and want to calibrate CM parameters against those observations.

**Status:** Not yet implemented. This document describes the planned API.

---

### Feature: CM Posterior Inference

**One-line description:** Infer a posterior distribution over CM parameter space given real-world data and SM UQ.

**Priority:** Must-have

**Motivation:**
The SM fitting step (SmoreBase) produces SM parameters and their uncertainty for each CM parameter set used to generate training data. The posterior inference step connects this to real-world observations: by sweeping CM parameter space and evaluating how well the SM (within its uncertainty region) fits the data, we can identify which CM parameters are consistent with observations.

**Behavioral specification (planned):**
- `buildPosterior(sm, realWorldData, uqResults, cmBounds; ...) -> CMPosteriorResult`
  - `sm::AbstractSurrogateModel` — the fitted surrogate model
  - `realWorldData` — real-world observational data (type TBD; likely a sibling type to `CMData`)
  - `uqResults::Vector{ProfileLikelihoodResult}` — SM UQ at known cohort points (from SmoreBase)
  - `cmBounds::ParameterPrior` — CM parameter bounds/priors
- Planned output: `CMPosteriorResult` — posterior samples or density over CM parameter space

**Open questions (to resolve at implementation time):**
- What type represents real-world observational data? A `CMData`-like struct without param_set axes, or a separate type?
- Is the posterior represented as samples (MCMC), a grid, or an approximate density?
- How is the SM uncertainty (`ProfileLikelihoodResult`) propagated into the likelihood evaluation?

**Acceptance criteria:** TBD at design time.
