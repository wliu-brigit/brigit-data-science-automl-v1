"""Persisted fraud entity graph — lossless DuckDB store + igraph analysis views.

Design: docs/superpowers/specs/2026-06-09-fraud-entity-graph-store-design.md
Store has no opinions (uncapped edges, full timestamps, all entity types,
self-contained advances snapshot); every judgment call — degree caps, layer
selection, as-of windows, scenario flags, weights — is a parameter of an
analysis-time view.

    from projects.fraud_anomaly_detection.graph import build, load, queries
"""
