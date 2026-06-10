# docs/

Design and planning notes, organized by **lifecycle stage**. These describe
*intent and history*, not current behavior — code and the package
[`README.md`](../README.md) are the source of truth for how the system works
today.

> **Resuming work / new session?** Read [`HANDOFF.md`](HANDOFF.md) first — the
> running note of where the last session left off and what to do next. Update it
> only when wrapping a session for handoff (or when asked) — not as a running
> changelog of mid-session progress; git history already records that.

Work moves through three folders in order:

| Folder | Holds |
|---|---|
| [`to-do/`](to-do/) | Work we plan to do. |
| [`execution/`](execution/) | The effort being worked on right now. Usually empty. |
| [`archive/`](archive/) | Completed efforts, kept as history. |

Pick up an effort by moving it `to-do/ → execution/`; finish it by moving it
`execution/ → archive/`.

> **Promotion to `execution/` is explicit and user-gated** — like
> [`HANDOFF.md`](HANDOFF.md), an effort moves into `execution/` *only when the
> user asks*, never automatically. New plans and specs land in `to-do/` and
> stay there until then. `execution/` holds only what is actively being worked
> right now, so it stays small and uncluttered.

## Writing an entry

Package each effort as a **self-contained folder with its own `README.md`** as
the front door, so a fresh session can read it and know the status and next
action (model: [`to-do/agent-orchestration/`](to-do/agent-orchestration/)). A
single `<name>.md` is fine for something small; promote it to a folder if it
grows. An empty file is a placeholder not yet expanded.

**Name the entry by the ask, not the provenance.** "leaderboard-dataset-
pinning" says what to do; "pilot-feedback" could mean anything. A grab-bag
findings list is a working note, not an entry — split it into named entries
before it lands here.
