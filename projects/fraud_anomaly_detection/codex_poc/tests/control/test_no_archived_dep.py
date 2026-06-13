import re
from pathlib import Path

CONTROL = Path("projects/fraud_anomaly_detection/codex_poc/control")


def test_control_has_no_archived_dependency():
    offenders = []
    for path in CONTROL.rglob("*.py"):
        if re.search(r"(from|import)\s+.*codex_poc\.archived", path.read_text()):
            offenders.append(str(path))

    assert not offenders, f"control/ must not depend on archived/: {offenders}"
