# 0001 – One repository with per-folder deploys

- **Status:** accepted
- **Date:** 2026-10-12

## Context
Five people work on edge code, ML, a backend and a dashboard. They share one event format.

## Decision
One GitHub repo. Each top-level folder has its own workflow with `paths:` filters, so a merge
only builds and deploys what changed. Shared contracts live in `shared/`.

## Consequences
- One PR can change the event schema, backend and dashboard together.
- Workflows must keep their `paths:` lists up to date when folders are added.
