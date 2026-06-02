"""Create a human-owned trial seeded from a prior run."""

from __future__ import annotations

from automl.project import Session
from automl.trial.create import create


def fork(
    slug: str,
    *,
    seed: str = "best",
    strategy: str = "manual_fork",
    hypothesis: str = "",
    session: Session | None = None,
):
    """Create a human trial directory without running it."""

    return create(
        slug=slug,
        strategy=strategy,
        hypothesis=hypothesis,
        seed=seed,
        training_origin="human",
        session=session,
    )


__all__ = ["fork"]
