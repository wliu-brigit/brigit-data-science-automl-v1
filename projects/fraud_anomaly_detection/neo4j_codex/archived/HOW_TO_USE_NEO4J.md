# How To Use The Neo4j Fraud Mirror

Neo4j Browser is useful only if you drive it with fraud-specific queries. The
left-side clicks show generic graph samples. They do not know the investigation
workflow, so they can look like a random cluster of dots.

## What The Dots Mean

Node labels:

- `Scenario`: a registered fraud scenario, such as `ring_device_burst`.
- `ReviewCluster`: a pre-ranked user/entity component for case review.
- `User`: a Brigit user.
- `ScenarioUser`: a user who matched at least one scenario.
- `FraudUser`: a user in the current `is_fraud` proxy band.
- `Dpd45User`: a user with the DPD45 bad-outcome label.
- `BankAccount`, `Device`, `Phone`, `Address`, `Email`, `PersistentAccount`:
  entity nodes the user touched.

Relationship types:

- `MATCHED_SCENARIO`: user -> scenario membership. This answers "which users
  matched this scenario?" It does **not** show the bank/device/phone reason by
  itself.
- `IN_REVIEW_CLUSTER`: user -> review cluster membership. This is a navigation
  edge for the ranked cluster table.
- `USED_DEVICE`, `USED_BANK_ACCOUNT`, `USED_PHONE`, `USED_ADDRESS`,
  `USED_PERSISTENT_ACCOUNT`: user -> entity links. These are the visible graph
  connections that answer "which bank/device/phone/address links these users?"

So if you click `MATCHED_SCENARIO` and see a dot cluster, read it as:

```text
Scenario node <-> many matching User nodes
```

That view says they all matched the same scenario. It does not mean they all
share the same bank account or device. To see that, inspect typed `USED_*`
relationships.

## Recommended Workflow: Suspicious Clusters -> Ring

1. Run `cypher/00_top_suspicious_clusters.cypher`.
   Pick a cluster with high `dpd45_user_rate`, enough users, and multiple
   entity types.
2. Run `cypher/01_cluster_ring_view.cypher`.
   This returns the visual ring: review-cluster users connected to the
   bank/device/phone/address nodes that explain the cluster.
3. Use `cypher/01_scenario_overview.cypher` and
   `cypher/02_scenario_to_users.cypher` as overlays when you want to understand
   how registered scenarios intersect with the cluster.
4. If an entity looks important, copy its `entity_type` and `entity_value`, then
   run `cypher/05_entity_drilldown.cypher`.

## How To Read A Bank Account Dot

If you click a `BankAccount` node and Neo4j says there are 25 users, that means
25 distinct users have a `USED_BANK_ACCOUNT` relationship to the same bank
account entity in this mirror.

The question to ask is not "why are there 25 bank accounts?" The question is:

```text
Which users touched this exact bank account, and how many are fraud/scenario/DPD45?
```

Use `05_entity_drilldown.cypher` for that. It lists the connected users and
their scenario/outcome status.

## Discovery

Use `cypher/06_discovery_candidates.cypher` for a first discovery pass. It finds:

```text
users who did not match a scenario
but share a bank/device/phone/address/etc. with scenario/fraud/DPD45 users
```

This is the useful Neo4j motion: start from known bad or known scenario users,
then look for nearby misses. It is not a validated rule by itself; it is a
review/discovery queue.

## Plug-The-Hole View

For this fraud workflow, Neo4j is most useful as a daily graph risk engine:

```text
known scenarios / fraud proxy / DPD45
  -> graph neighborhoods and GDS communities
  -> residual candidate users
  -> risky shared entities
  -> review queue or risklist
```

Use GDS to discover suspicious islands or dense communities. Use Cypher to turn
those findings into explainable evidence: which device, bank account, phone, or
address connected the user to the risky group. The operational output should be
a user/entity risklist or review packet, not a raw community ID.

## Browser Tips

- Do not start by clicking relationship types in the left sidebar. Those are
  generic samples.
- Start with the supplied Cypher files in `out/neo4j/cypher/`.
- Keep every visual query bounded with `LIMIT`.
- For full data, never visualize the whole graph. Use:
  `suspicious cluster -> selected user/entity -> local ego/ring`.
- Relationship arrows do not mean fraud causality. They mean data linkage:
  matched a scenario, belongs to a review cluster, or used an entity.
