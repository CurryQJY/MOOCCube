from dataclasses import dataclass


@dataclass(frozen=True)
class LIRAConfig:
    n_users: int
    n_items: int
    content_dim: int
    embedding_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.35
    temperature: float = 0.07
    margin: float = 0.15
    steps: int = 3
    update_lr: float = 0.10
    min_fit: float = 0.05
    step_cap: float = 0.05
    total_cap: float = 0.10
    min_gain: float = 0.001
    refinement_loss_weight: float = 0.5
    stability_loss_weight: float = 0.01
    concept_weight: float = 0.25
    prerequisite_beta: float = 1.0
    difficulty_beta: float = 1.0
    pseudo_cold_ratio: float = 0.30
    pseudo_cold_min_popularity: int = 5

    def __post_init__(self) -> None:
        if min(self.n_users, self.n_items, self.content_dim, self.embedding_dim) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if not 0.0 <= self.pseudo_cold_ratio <= 1.0:
            raise ValueError("pseudo_cold_ratio must lie in [0, 1]")
        if min(
            self.update_lr,
            self.min_fit,
            self.step_cap,
            self.total_cap,
            self.min_gain,
            self.refinement_loss_weight,
            self.stability_loss_weight,
        ) < 0.0:
            raise ValueError("refinement controls must be non-negative")
