from dataclasses import dataclass
from pathlib import Path

# Stages shared by every classifier of a run, and the ones that live under each.
DATASET_STAGES = ("split", "graph", "regions", "complexity")
CLASSIFIER_STAGES = ("classify", "regress", "render")


@dataclass(frozen=True)
class RunPaths:
    """Where each stage writes: one folder per stage, under the dataset or the classifier."""

    dataset: Path
    classifier: Path

    def of(self, stage: str) -> Path:
        """The folder a stage owns."""
        if stage in DATASET_STAGES:
            return self.dataset / stage
        if stage in CLASSIFIER_STAGES:
            return self.classifier / stage
        raise ValueError(
            f"Unknown stage {stage!r}. Valid: {DATASET_STAGES + CLASSIFIER_STAGES}."
        )
