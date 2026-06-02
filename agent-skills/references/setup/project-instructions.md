# projects/<project_name>/PROJECT_INSTRUCTIONS.md

Your domain notes for the AutoML loop. The Manager loads this file fresh at the start of every turn — it's the place to encode anything the agent should keep in mind across iterations.

## Sections (template)

Each project has `projects/<project_name>/PROJECT_INSTRUCTIONS.md` with sections
you fill in:

- **Goal** — what "better" means for this project. State the primary metric and its business interpretation.
- **Constraints (hard)** — latency caps, fairness, regulatory constraints. The agent will treat these as inviolable.
- **Domain notes** — what you know about the data that the agent can't infer from features alone. Seasonality, leakage rules specific to your data, segments, fairness considerations.
- **Approaches to try** — model families, feature transforms, ensembling ideas, hyperparameter ranges that have worked in prior projects.
- **Approaches to avoid** — what you've ruled out and why. Saves the agent from rediscovering.
- **Open questions** — things you're curious about — explicit invitations for the agent to explore.

## Style

- **Scannable, not exhaustive.** The Manager reads this every turn — keep it tight (under ~500 words).
- **Concrete.** "Avoid models that take more than 30s to predict" beats "avoid slow models."
- **Versioned.** Update as you learn — this file is meant to evolve. Use git history to see why constraints changed.

## How it's used

- The Manager prompt includes this file verbatim each turn.
- `profile` may suggest additions based on what it learns about the data.
- `review` references it when explaining trial decisions.

## How it's NOT used

- Not loaded by trial code.
- Not loaded by trial code.

## Filling it in

Setup ends with this file empty-ish. You can:

- Fill it before running `profile` (preferred - gives the agent context up front).
- Fill it iteratively as `profile` surfaces things you want to remember.
- Treat it as a living doc; update after every meaningful run.
