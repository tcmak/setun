# Working with a Coding Agent on Setun

1. Read `README.md`, `PLAN.md` and `NEXT.md` first. Work on one task at a time, in `PLAN.md` order, unless told otherwise.  
2. A task is done only when every acceptance criterion passes via a command, test or report.  
3. Commit after each acceptance criterion: `<task-id>: <what>`.  
4. Never weaken a test, threshold or golden vector to get green. If one looks wrong, stop and write it up in `docs/open-questions.md`.  
5. Pin versions. No floating dependencies.  
6. If a task needs an open decision (see `PLAN.md` → Decisions), stop and ask.  
7. Keep `make ci` green and inside its time budget.  
8. RTL under `setun/accelerators/` is written by the maintainer through Season 2\. Write tests, reference code, scripts and docs; review RTL and explain errors when asked; do not commit RTL there unless asked.  
9. End every working session by updating `NEXT.md` and appending a line to `docs/log.md`.  
10. Strictly follows TDD practice and version control.
