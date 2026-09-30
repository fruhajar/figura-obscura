# Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — how the nine crates fit together, the
  pipeline every front end shares, the threading and cancellation model, and the
  invariants worth not breaking.
- **[PERFORMANCE.md](PERFORMANCE.md)** — what dominates a run, what has been
  measured, how to reproduce it (`cargo xtask bench`), and the optimisations
  that were tried and rejected.
- **[HOST-BUILD.md](HOST-BUILD.md)** — building on a real machine, including the
  per-vendor GPU prerequisites and how to tell whether a GPU build is actually
  using the GPU.
- **[RELEASING.md](RELEASING.md)** — cutting a release: the version bump, the tag
  that builds all three platforms, what the notes have to say, and the smoke test
  before the draft is published.
- **[planning-handoff.md](planning-handoff.md)** — the original design handoff.
  Historical: it describes what was planned, not what shipped. Read
  `ARCHITECTURE.md` for the latter.

The project overview lives in [`../README.md`](../README.md), which is what
GitHub shows on the front page.
