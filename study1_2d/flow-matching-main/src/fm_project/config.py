from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ExperimentConfig:
    dataset_name: str = "checkerboard"
    n_train: int = 20_000
    n_eval: int = 5_000
    batch_size: int = 512
    hidden_dim: int = 256
    time_dim: int = 64
    n_layers: int = 3
    lr: float = 1e-3
    n_epochs: int = 200
    steps_per_epoch: int = 50
    weight_decay: float = 0.0
    sigma: float = 0.1
    nfe_list: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    solver_methods: List[str] = field(default_factory=lambda: ["euler", "heun", "rk4"])
    seed: int = 42
    device: str = "auto"
    output_dir: str = "results"
    variants: Dict[str, str] = field(
        default_factory=lambda: {
            "ot": "ExactOptimalTransportConditionalFlowMatcher",
            "vp": "VariancePreservingConditionalFlowMatcher",
            "target": "TargetConditionalFlowMatcher",
            "schrodinger": "SchrodingerBridgeConditionalFlowMatcher",
        }
    )
