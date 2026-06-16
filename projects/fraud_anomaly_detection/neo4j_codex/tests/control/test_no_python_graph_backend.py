from pathlib import Path


def test_active_control_does_not_import_igraph_or_python_graph_discovery():
    offenders: list[str] = []
    for path in Path("projects/fraud_anomaly_detection/neo4j_codex/control").rglob("*.py"):
        text = path.read_text()
        if "import igraph" in text:
            offenders.append(f"{path}: imports igraph")
        if "projects.fraud_anomaly_detection.graph.discover" in text:
            offenders.append(f"{path}: imports Python graph discovery")
        if "projects.fraud_anomaly_detection.graph.load" in text:
            offenders.append(f"{path}: imports Python graph loader")

    assert offenders == []
