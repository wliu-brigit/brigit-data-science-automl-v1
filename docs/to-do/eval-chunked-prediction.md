# Chunked prediction in the eval core

**Status: to-do (idea, not scheduled).** Raised 2026-06-12 during the neobank_ncm
native decision re-eval work.

## The ask

`evaluate()` scores the model with a single `model.predict(model_input=frame)` call
over the **entire** eval frame (`automl/eval/evaluate.py` `_predict_model`). For most
eval datasets that is fine. For large frames it is not: the neobank_ncm
`oot_new_links` decision frame is **~5.3M rows**, and a model whose preprocessor
materializes a dense matrix (the WoE/sklearn transformer → ~5.3M × 168 × 8 bytes ≈
**7 GB**) can exhaust RAM in one shot. The legacy financial notebook chunked scoring
at 250k rows (`projects/neobank_ncm/analysis/scoring.py:SCORE_CHUNK`) for exactly this
reason.

Make the eval path **bound prediction RAM by construction** — chunk the frame, predict
per chunk, concatenate — so any large eval dataset is safe without per-project
workarounds.

## Why it isn't done now

For the native decision re-eval we deliberately did **not** work around this with the
private `evaluate(_model=...)` hook (a reach into internals to inject a chunking model
wrapper). The decision was: let `evaluate()` predict as-is for now (the scoring has run
before), and if RAM reliability becomes a real problem, fix it **in the core** rather
than per project. This to-do is that core fix.

## Possible shape

- A chunk size on the eval path (config or `evaluate(..., predict_chunk_size=...)`),
  defaulting to "no chunking" so existing behavior is unchanged.
- `_predict_model` iterates `range(0, len(frame), chunk)` and concatenates, mirroring
  `scoring.score_daily`'s loop — but generic, in `automl/eval`.
- Keep it transparent to metrics: they still receive one `(df, y_pred)` at full length.

## Trigger

If the neobank_ncm native re-eval (`native-reeval-decision-metrics.md`) OOMs on the
5.3M-row frame in practice, promote this from idea to scheduled.
