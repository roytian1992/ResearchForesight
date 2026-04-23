# Results Cleanup 2026-04-16

## Purpose

Clear all historical artifacts under `ResearchForesight/results/` so the project no longer carries old experiment outputs.

## Scope

- cleaned directory:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results`
- policy:
  - remove all top-level contents under `results/`
  - keep the `results/` root directory itself

## Pre-check

- item count before cleanup: `84`
- active process check:
  - no process was found writing to `ResearchForesight/results/` at cleanup time

## Execution

- command:
  - `rm -rf /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/*`

## Verification

- item count after cleanup: `0`
- `find .../results -mindepth 1 -maxdepth 1` returns empty

## Notes

- this cleanup does not affect the ongoing benchmark rebuild, which is currently writing to `data/releases/` and `tmp/logs/`
