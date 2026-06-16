# Explore the fraud graph in the Neo4j Browser

Everything else in this unit is **code** — the control loop, the discovery
methods, the mirror exporter. This page is different: it's a **hands-on tour of
the Neo4j Browser**, the visual front end. You stand the graph up, paste a
query, and *see* the result — tables for the numbers, a rendered node-and-edge
canvas for the fraud rings. No code to write; just open the UI and explore.

## The one-line pitch

Pick a user at random and there's a **~9% chance** they go 45+ days delinquent
(DPD45). But follow the **shared infrastructure** — the same device, bank
account, or phone wired to several accounts — and you land in pockets that are
**90–100% bad**. The graph makes that concentration visible and findable. That
gap (9% → ~100%) is the whole argument.

---

## 1. Open the Neo4j Browser

The local mirror runs in Docker.

- **URL:** http://localhost:7474
- **User:** `neo4j`
- **Password:** `fraudpocpass`
- **Bolt** (for drivers, not needed for the browser): `bolt://localhost:7687`

If the page doesn't load, the container isn't running — start it from the repo
root:

```bash
bash projects/fraud_anomaly_detection/neo4j_codex/neo4j_mirror/scripts/setup_neo4j.sh
```

That stands up Neo4j + the GDS plugin and pours in the graph store. First boot
takes a couple of minutes; the script prints the URL and credentials when ready.

## 2. How to use the browser as a visual query front end

- **Run a query:** paste it into the command bar at the top, press
  **Cmd/Ctrl + Enter**.
- **Each run is a "frame":** results stack as cards in the stream below. Scroll
  back to revisit any earlier result while presenting — they stay cached for the
  session. Pin a frame (pin icon) to keep it on screen.
- **Save a query:** after running, click the **star icon** on the editor to save
  it as a favorite (left sidebar). The first `//` comment line becomes its name.
- **Table vs. graph view:** queries that `RETURN` numbers show a **Table**.
  Queries that return nodes/relationships (like the ring views) show a **Graph**
  canvas — drag nodes around, zoom, click a node to inspect its properties.
- **Color the bad users:** on a graph result, click the **`Dpd45User`** label
  chip in the legend (bottom of the canvas) and pick a bold color. The
  delinquent users light up so you can see the ring is almost all red.

> Tip: run the queries top to bottom — they're built as a story, each one
> setting up the next.

## 3. The graph, briefly

- **Nodes:** `User` (also tagged `FraudUser` / `ScenarioUser` / `Dpd45User` when
  applicable) and `Entity` (a shared resource — `Device`, `BankAccount`,
  `Phone`, `Address`, `PersistentAccount`).
- **Edges:** `USED_DEVICE`, `USED_BANK_ACCOUNT`, `USED_PERSISTENT_ACCOUNT`,
  `USED_PHONE`, `USED_ADDRESS` (User → Entity), and `MATCHED_SCENARIO`
  (User → named fraud scenario).
- **Key user flags:** `label_gross_dpd45` (went 45+ days delinquent — the broad
  bad-outcome signal, ~9% of users) and `is_fraud` (confirmed fraud, ~0.32%).

---

## 4. The query deck (paste-in starting point)

Results shown are from the current full-data store, so you know what to expect.

### 00 · Baseline — the number every later query must beat

```cypher
// ~868K users. DPD45 = 8.94%; confirmed fraud = 0.32%.
// Window read from edge timestamps: ~262 days, Aug 2025 -> Apr 2026.
MATCH (u:User)
WITH count(u) AS users,
     sum(CASE WHEN u.label_gross_dpd45 THEN 1 ELSE 0 END) AS dpd45,
     sum(CASE WHEN u.is_fraud THEN 1 ELSE 0 END) AS fraud
MATCH ()-[r:USED_DEVICE|USED_BANK_ACCOUNT|USED_PERSISTENT_ACCOUNT|USED_PHONE|USED_ADDRESS]->()
RETURN users,
       dpd45,
       round(100.0 * dpd45 / users, 2) AS dpd45_pct,
       fraud,
       round(100.0 * fraud / users, 2) AS fraud_pct,
       min(r.first_ts) AS window_start,
       max(r.last_ts)  AS window_end,
       duration.inDays(min(r.first_ts), max(r.last_ts)).days AS window_days;
```

**Result:** 868,082 users · DPD45 77,580 (**8.94%**) · fraud 2,753 (**0.32%**) ·
262-day window. *Say out loud: "random is 9% bad — now watch what shared
infrastructure does."*

### 01 · The hot clusters — shared resources whose users are almost all bad

```cypher
// Resources used by a small ring (5-40 users) where >=60% went DPD45.
// Top hits: banks/phones shared by 30-40 users where 100% went bad — ~11x the
// 9% baseline, on a single shared key. THIS TABLE IS THE ARGUMENT.
MATCH (e:Entity)<-[:USED_DEVICE|USED_BANK_ACCOUNT|USED_PERSISTENT_ACCOUNT|USED_PHONE|USED_ADDRESS]-(u:User)
WHERE e.n_users >= 5 AND e.n_users <= 40
WITH e,
     count(u) AS users,
     sum(CASE WHEN u.label_gross_dpd45 THEN 1 ELSE 0 END) AS bad
WITH e, users, bad, 1.0 * bad / users AS bad_rate
WHERE bad_rate >= 0.6
RETURN e.entity_type AS type,
       e.entity_value AS value,
       users,
       bad,
       round(100.0 * bad_rate, 1) AS dpd45_pct
ORDER BY users DESC, dpd45_pct DESC
LIMIT 15;
```

**Result:** a table of banks/phones/addresses shared by 28–38 users at
90–100% DPD45. Top row: bank `323274775-814499153`, 38 users, all 38 bad.

### 02 · See the ring — one shared bank account, drawn as a star

```cypher
// Returns NODES + EDGES (not a table) so the browser renders the picture: one
// bank account in the center, every user that touched it around it. 38 users,
// all 38 DPD45. Click the Dpd45User chip in the legend and give it a bold color.
MATCH (e:Entity {entity_type: 'bank', entity_value: '323274775-814499153'})
MATCH (e)<-[r:USED_DEVICE|USED_BANK_ACCOUNT|USED_PERSISTENT_ACCOUNT|USED_PHONE|USED_ADDRESS]-(u:User)
RETURN e, r, u;
```

**Result:** a star — one bank node, 38 user nodes around it. Color `Dpd45User`
and the whole star lights up.

### 03 · See the wider ring — two hops out from that bank account

```cypher
// Same seed, follow users out through ANY shared resource for up to 2 hops.
// Shows the ring beyond the obvious key: the other devices/phones/addresses the
// same people quietly share. Capped at 300 paths so the canvas stays readable.
MATCH (e:Entity {entity_type: 'bank', entity_value: '323274775-814499153'})<-[:USED_BANK_ACCOUNT]-(u:User)
MATCH p = (u)-[:USED_DEVICE|USED_BANK_ACCOUNT|USED_PERSISTENT_ACCOUNT|USED_PHONE|USED_ADDRESS*1..2]-(n)
RETURN p
LIMIT 300;
```

**Result:** a dense web — the seed users plus everything they share, sprawling
well beyond the original 38.

### 03b · Wider ring DPD45 rate — does the signal survive the expansion?

```cypher
// Seed = users on the bank account; expand 2 hops through ANY shared resource.
// Collect the distinct users in that neighborhood and compute their DPD45 rate.
MATCH (e:Entity {entity_type: 'bank', entity_value: '323274775-814499153'})<-[:USED_BANK_ACCOUNT]-(seed:User)
MATCH (seed)-[:USED_DEVICE|USED_BANK_ACCOUNT|USED_PERSISTENT_ACCOUNT|USED_PHONE|USED_ADDRESS*1..2]-(m:User)
WITH collect(DISTINCT seed) + collect(DISTINCT m) AS members
UNWIND members AS u
WITH DISTINCT u
RETURN count(u) AS ring_users,
       sum(CASE WHEN u.label_gross_dpd45 THEN 1 ELSE 0 END) AS dpd45_users,
       round(100.0 * sum(CASE WHEN u.label_gross_dpd45 THEN 1 ELSE 0 END) / count(u), 1) AS dpd45_pct;
```

**Result:** 891 users, 884 bad = **99.2%**. The ring grows 23× but stays ~99%
bad — the signal does not dilute as you expand it.

### 04 · Communities at scale — an algorithm finds the bad pockets, unsupervised

Run these **three statements one at a time**.

```cypher
// 04a · drop any stale projection first
CALL gds.graph.drop('demo', false) YIELD graphName;
```

```cypher
// 04b · build the in-memory projection of the whole user<->entity graph
CALL gds.graph.project('demo', ['User', 'Entity'], {
  USED_DEVICE:             {orientation: 'UNDIRECTED'},
  USED_BANK_ACCOUNT:       {orientation: 'UNDIRECTED'},
  USED_PERSISTENT_ACCOUNT: {orientation: 'UNDIRECTED'},
  USED_PHONE:              {orientation: 'UNDIRECTED'},
  USED_ADDRESS:            {orientation: 'UNDIRECTED'}
});
```

```cypher
// 04c · rank communities by DPD45 rate. Connected-components (WCC) carves the
// population into communities with NO labels at all; the worst run 80-100% bad
// vs the 9% baseline — proof the clusters aren't cherry-picked.
CALL gds.wcc.stream('demo')
YIELD nodeId, componentId
WITH gds.util.asNode(nodeId) AS n, componentId
WHERE n:User
WITH componentId,
     count(*) AS users,
     sum(CASE WHEN n.label_gross_dpd45 THEN 1 ELSE 0 END) AS bad
WHERE users >= 6 AND users <= 60
RETURN componentId,
       users,
       bad,
       round(100.0 * bad / users, 1) AS dpd45_pct
ORDER BY dpd45_pct DESC, users DESC
LIMIT 25;
```

**Result:** a table of communities, the worst at 80–100% DPD45. This is the
honest "the algorithm found these, we didn't hand-pick them" answer.

### 05 · Tie it to named scenarios — bad-rate per fraud pattern

```cypher
// The graph already carries named fraud scenarios (device burst, identity burst,
// shared persistent account, account reuse). Each scenario's matched users run
// well above baseline — graph patterns operationalized as rules.
MATCH (s:Scenario)<-[:MATCHED_SCENARIO]-(u:User)
RETURN s.name AS scenario,
       s.title AS title,
       count(u) AS matched_users,
       round(100.0 * sum(CASE WHEN u.label_gross_dpd45 THEN 1 ELSE 0 END) / count(u), 1) AS dpd45_pct
ORDER BY dpd45_pct DESC;
```

---

## 5. A good order to walk through it

1. **00** — establish the 9% baseline.
2. **01** — the table of 90–100% pockets. *"Follow the shared infrastructure."*
3. **02** — the star lights up red.
4. **03 + 03b** — the ring sprawls to 891 people and stays 99% bad.
5. **04** — the algorithm finds these communities without labels.
6. **05** — and they map onto named, enforceable fraud patterns.

## 6. Caveats (worth saying if asked)

- **DPD45 ≠ fraud.** `label_gross_dpd45` (~9%) is "went 45+ days delinquent" — a
  broad bad-outcome label. `is_fraud` (~0.32%) is confirmed fraud. The ring
  queries beat the 9% DPD45 baseline; both labels are in Query 00 for context.
- **03b is seeded from an already-bad key.** The 99.2% partly reflects that this
  community is tightly fraudulent — it's a drill-down, not a blind prediction.
  Query **04** (unsupervised communities) is the cleaner "we didn't tell it who's
  bad" claim.
- **Hardcoded entity.** Queries 02/03/03b pin bank `323274775-814499153`
  (verified on this store). If the store is rebuilt, rerun **01** and swap in a
  fresh top hit.
