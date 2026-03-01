import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fm_project.config import ExperimentConfig
from fm_project.experiments import run_full_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run flow-matching experiments")
    parser.add_argument("--dataset", default="checkerboard", choices=["checkerboard", "two_moons", "gaussian_mixture"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--n-eval", type=int, default=5000)
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        dataset_name=args.dataset,
        n_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        n_train=args.n_train,
        n_eval=args.n_eval,
        output_dir=args.output_dir,
        sigma=args.sigma,
        seed=args.seed,
        device=args.device,
    )
    results = run_full_experiment(cfg)
    print(json.dumps(results["main_metrics"], indent=2))
    print(f"Saved full report to {Path(args.output_dir) / 'results.json'}")


if __name__ == "__main__":
    main()
