# CLAUDE.md — SmoreFit.jl

## About the User
Assistant professor working on computational modeling of cancer-immune interactions. Research involves mechanistic modeling and agent-based modeling (ABM) frameworks. The "complex model" (CM) in this codebase is typically an ABM, but can be any slow, expensive simulator.

## Key Documents — Read These First

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Project overview + **Implementation Status** (what is built, what remains) |
| [PRD.md](PRD.md) | Behavioral specification for every feature — acceptance criteria and edge cases |
| [progress.md](progress.md) | Session journal: decisions made, approaches rejected, open questions |

Start any feature session by reading the relevant PRD entry and the Implementation Status section of `README.md`.

## Project Overview

SmoreFit.jl provides posterior inference on complex model (CM) parameter space, given a fitted surrogate model (SM) and real-world observational data. It is part of the [Smore](https://github.com/drbergman-lab/Smore.jl) ecosystem and depends on [SmoreBase.jl](https://github.com/drbergman-lab/SmoreBase.jl) for core types and the SM interface.

**Status: not yet implemented.** See [PRD.md](PRD.md) for the planned API.

A sibling package `SmoreExamples.jl` holds worked examples and model-specific code. Do **not** add model-specific code to this repo.

## Repository Structure

```
Project.toml
src/
└── SmoreFit.jl    # package entrypoint (stub)
test/
└── runtests.jl
```

## Scope

All SmoreFit feature work belongs here. For work on:
- Core types, SM fitting, UQ → `SmoreBase.jl`
- Global sensitivity analysis → `SmoreGSA.jl`
- Worked examples → `SmoreExamples.jl` (do **not** add model-specific code here)

## Worktree Sessions

When Claude Code launches a session inside a git worktree (primary working directory ends with `.claude/worktrees/<name>`), **all file reads and writes must use paths rooted at the worktree, not the main repo root.** The main repo may appear as an "Additional working directory" in the environment block — ignore it for file edits.

## Git Workflow

Claude Code (the CLI tool) runs directly on your machine and can freely run `git add`, `git commit`, `git checkout`, and all other git operations. No restrictions apply.

### Branching Rules
- Never modify `main` directly.
- Default base branch is `main` unless specified otherwise.
- Branch names: `feature/<short-desc>`.
- After merging, delete the feature branch.

## Naming Conventions

- **Functions:** `camelCase` (e.g., `buildPosterior`)
  - `camelCase` distinguishes function calls from variable/field names, consistent with ModelManager.jl
- **Internal helpers:** `_camelCase` prefix
- **Types / Structs:** `PascalCase` (e.g., `CMPosteriorResult`)
- **Constants / module-level refs:** `snake_case` for internal refs; `SCREAMING_SNAKE_CASE` for env vars
- **Files:** `snake_case.jl`
- **Exported vs internal:** public API exported from `src/SmoreFit.jl`; internal helpers prefixed `_`

## Git Rules

**Never stage or commit without explicit instruction.**
The human reviews diffs and stages files themselves. Do not run `git add`, `git stage`, or `git commit` unless the human explicitly asks you to. You may run read-only git commands (`git status`, `git diff`, `git log`, `git branch`) freely.

## Required Workflow for Any Change

1. Generate a **design brief** in the assistant response **before any code changes**.
2. Wait for human approval.
   1. Update PRD.md to include new feature or changes.
   2. Open a new entry in progress.md and log design process, decisions, open questions.
3. Create the feature branch: `git checkout -b feature/<desc>`.
4. Implement in the feature branch only.
5. Update [README.md](README.md) Implementation Status when a feature is complete.
6. Trim PRD.md and progress.md to reflect final implementation before merging.
7. Tell the human the branch is ready; they will review, stage, and commit.

**Design brief template:**
```
# Design Brief: [Feature/Refactor Name]

## Motivation
[1-2 sentences: why is this change needed?]

## Scope
- **Files affected:** `src/...`
- **New files:** (if applicable)
- **Breaking changes:** Yes/No

## Proposed Architecture
[2-3 paragraphs or diagram]

## Testing Strategy
- Unit tests for: [list]
- Integration tests: [if applicable]

## Estimated Effort
- Lines of code: ~[estimate]
- Risk level: Low / Medium / High
```

## Definition of Done

A feature is complete when **all** of the following are true:

1. **Tests pass:** `julia --project=. -e 'using Pkg; Pkg.test()'` runs green.
2. **Docstrings written:** Every exported function has a docstring with description, arguments, return value, and at least one example.
3. **README updated:** Implementation Status marks the feature complete.
4. **PRD reflects reality:** If implementation deviated, update the PRD entry.
5. **No regressions:** Full test suite has no new failures.

## Integration Essentials

- Package entrypoint: `src/SmoreFit.jl` — add `include(...)` and update `export` when adding new source files
- Run tests: `julia --project=. -e 'using Pkg; Pkg.test()'`

## Julia Environment Rules

- Always run Julia with `--project=.`
- Preferred test command: `julia --project=. -e 'using Pkg; Pkg.test()'`
- Do not edit `Manifest.toml` or add/bump dependencies without explicit approval.
