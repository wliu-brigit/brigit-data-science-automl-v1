# projects/<project_name>/PROJECT_INSTRUCTIONS.md

Your domain guidance for the AutoML loop. The Manager loads this file fresh at
the start of every turn — it is how you steer the loop between runs.

**The canonical guide for writing it lives in the project itself:** every
scaffolded project's `README.md` has a "Writing PROJECT_INSTRUCTIONS.md"
section — the audience, the two rules (don't restate config; write only what
the loop can't infer), what belongs in each section, and the style bar. Read
`projects/<project_name>/README.md` first.

## How it's used

- The Manager prompt includes this file verbatim each turn.
- `profile` may suggest additions based on what it learns about the data.
- `review` references it when explaining trial decisions.

## How it's NOT used

- Not loaded by trial code.

## Filling it in

Setup ends with this file empty-ish. You can:

- Fill it before running `profile` (preferred — gives the agent context up front).
- Fill it iteratively as `profile` surfaces things you want to remember.
- Treat it as a living doc; update after every meaningful run.
