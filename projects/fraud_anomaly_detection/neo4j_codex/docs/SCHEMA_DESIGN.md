# Neo4j mirror — schema working notes (NOT signed off)

Discussion record for the Neo4j graph track's schema. These are **notes from
an in-progress design conversation**, not a finished design: high-level
direction agreed in conversation, many details still open. The durable
guiding principles these notes answer to live in
[`../PRINCIPLES.md`](../PRINCIPLES.md).

## Agreed so far (2026-06-12 discussions)

- **Graph unit = identity** (P5). User + Entity nodes; entities are
  first-class because the entity is the enforcement unit — the top of the
  distillation ladder is a list of entity values.
- **Entity types:** device, bank account, persistent account, phone, address,
  email, **plus IP** (reversing the POC exclusion; per P3 it enters the graph
  but stays out of default projections). Practical note: v3 emits no IP key —
  next-SQL-touch item, batch with the link table.
- **Admission rule for future node types:** sharing rare-by-shape;
  junk/sentinel values screenable upstream; link timestamp as-of anchorable.
  (IP is admitted as information despite failing the first test — hence the
  projection guardrail.)
- **Advances are not nodes.** Roll-up rules instead: user nodes carry
  `n_advances`, `n_mature_advances`, `n_bad_advances`, `bad_advance_rate`
  (empty when no mature advance — unknown, not zero). Implemented in
  `export_neo4j_mirror.py` with tests.
- **Dual outcome semantics** (P6): ever-bad = sensitive evidence marker;
  strict = query-time threshold over `bad_advance_rate`, default 0.8.
  Cluster artifacts report both and self-document the threshold
  (`strict_dpd45_users`, `strict_dpd45_user_rate`, `strict_threshold`).
  Evidence behind the design: in the sample an 80% bar excludes 48 of 2,056
  ever-bad users, and 31 of those 48 are bust-out-shaped (bad advance is
  their last) — a hard purity definition would systematically drop the
  fraud-shaped trajectory.
- **Sample caveat (standing):** the 20k sample is advance-sampled (97.9%
  single-advance users); the user↔advance roll-up profile must be re-run on
  the v3 store as a validation anchor (tracked in TODO).
- **Scenarios stay as overlay nodes** + `MATCHED_SCENARIO` edges (analyst
  entry point, mirror-validation anchor); flags duplicated as user properties
  for GDS filtering — deliberate dual representation. The mirror carries
  scenario results, never scenario computation (P4).

## Edge model — agreed in discussion 2026-06-12 (DESIGN ONLY, not built)

> Not yet implemented: the current export still uses the old single
> `USED_<TYPE>` edge with a merged `sources` string and one `n_events` count.
> The model below is the agreed target for the v3 pour rework.

- **One edge per (user, entity, type); provenance carried as counts on the
  edge** — not split into `LINKED_*`/`ADVANCED_ON_*` types, and not merged
  with loss. The edge holds `n_advance_events`, `n_link_events`, `n_bad`, and
  per-provenance timestamps. "Linked but never advanced" (the strongest ring
  evidence) is then just `WHERE r.n_advance_events = 0`; a fully link-only
  *user* is `WHERE u.n_advances = 0` (no dedicated `LinkOnlyUser` label
  needed). Supersedes the earlier split-types lean — that rested on a
  losslessness argument that does not hold (type vs property both preserve
  provenance and both allow analysis-time selection). The only real defect to
  fix is the current export's merge, which sums across provenance and loses
  the per-provenance counts.
- **Outcome counts are construction-time facts; thresholds stay query-time**
  (P6). Edge counts *must* be construction-time — raw advances live in DuckDB,
  Neo4j never sees them. Entity-level outcome aggregates (sum over an entity's
  incoming advance edges = what a burned-entity list reads) are poured onto
  entity nodes as a materialized view, recomputed each daily rebuild (no
  staleness under full rebuild); they could be query-time but are poured
  because the burned-entity list ranks all entities daily.
- **`SHARES_RESOURCES` pair edges — agreed in principle, gated on v3 size.**
  The precompute is not for speed; it puts the multi-type corroboration rule
  into graph *structure* so community detection (which reads structure, not
  `WHERE` clauses) forms communities that respect it. Stay lossless (P3/P6):
  pour all sharing pairs over screened entities with `n_types`/`n_entities` as
  properties, apply the `n_types >= 2` bar at projection time. This is the one
  genuinely new *heavy* artifact (a 50-user entity = 1,225 pairs) — measure
  its real size at v3 scale before committing to build it.

Entity aggregates, pair edges, and the operational tier's GDS community
write-back are the same move: **materialize a derived view over the lossless
base, recompute each daily rebuild, never treat it as truth.** A candidate
principle once it has earned its place — not yet adopted.

## Still open — not yet discussed

1. **Cluster thresholds & review-score ranking — PARKED, data-driven
   (2026-06-12).** Agreed: the current additive `review_score` is too
   arbitrary and conflicts with the project's anti-additive
   (conjunctive-scenario) stance — do not treat it as the product ranking.
   Detailed qualify/rank design deferred to the data-backed analysis pass on
   v3, where it surfaces naturally. Directional lean recorded so it isn't
   relitigated from scratch: pour cluster FACTS (n_users, n_mature, strict +
   ever-bad counts/rates, n_types, n_scenario, n_net_new, n_entities, type
   composition); qualify conjunctively (gates, not weights; net-new
   first-class); rank by `ORDER BY` a transparent metric (strict-bad
   concentration among net-new, tie-broken by corroboration then size), not a
   baked score; thresholds stay query-time (P6).
2. **Durable registry & monitoring** — control registry (scenarios + entity
   blocklist), candidate lifecycle, enforcement state, monitoring history;
   all durable → DuckDB/warehouse, poured into the daily mirror as display
   state. The product backbone; deserves its own session.
3. **Burst descriptors** — collapsed edges lose burst shape; if discovery
   needs bursts visible in-graph, compute descriptors in DuckDB at export.
   Deferred until v3 discovery shows what it reaches for.
4. **Operational tier** (inherits all of the above): two-phase pour + GDS
   write-back of community IDs as indexed properties (kills the igraph/GDS
   split-brain); day-over-day community lineage via member-overlap matching
   in DuckDB (GDS `seedProperty` rejected — undefined behavior on component
   splits under full rebuild); degree-cap/supernode policy placement (lean:
   pour everything, cap at projection/query time, one documented policy).

## Research grounding (2026-06-12 deep-research pass)

External practice (vendor reference schemas, Grab production, GDS docs)
supports: identifier-as-node, typed edges, aggregated grain for ring
detection (event nodes only pay off for sequence-fraud/GNN workloads — both
out of scope), risk propagation over curated/corroborated edges only (single
shared identifier ≈ family, not fraud), community IDs persisted via GDS
write-back as indexed properties, WCC as the deterministic production
assignment with LabelProp as exploratory lens, and as-of construction as the
dominant correctness concern. Honest gaps: supernode/degree-cap thresholds
and sentinel handling have no verified external authority (our own
measurements govern), and real bank/fintech deployment schemas mostly failed
verification — treat external patterns as vendor-validated, not
battle-tested. Load-bearing sources: Neo4j GDS fraud series pt. 2, GDS WCC
docs, Grab engineering (graph-networks), Neo4j temporal-graph post.
