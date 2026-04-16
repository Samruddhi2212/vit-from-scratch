"""
Regression smoke tests: models, LEVIR/CD configs, CIFAR & ablation CLIs, script entry points.

No dataset downloads and no training loops. Run from repository root: ``pytest``
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Models (CIFAR + ablations) ────────────────────────────────────────────────


def test_vit_cifar_forward() -> None:
    from configs.config import ViTConfig
    from models.vit import ViT

    cfg = ViTConfig()
    m = ViT(cfg)
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    assert y.shape == (2, 10)


def test_ablation_variants_forward() -> None:
    from models.ablation_variants import ViTNoPosition
    from configs.config import ViTConfig

    m = ViTNoPosition(ViTConfig())
    x = torch.randn(1, 3, 32, 32)
    y = m(x)
    assert y.shape == (1, 10)


def test_cifar10_standalone_import() -> None:
    from utils.cifar10_standalone import ensure_cifar10_downloaded

    assert callable(ensure_cifar10_downloaded)


def test_change_detection_metrics_import() -> None:
    from utils.metrics import ChangeDetectionMetrics

    assert ChangeDetectionMetrics is not None


# ── LEVIR-CD: YAML merge + forward (mirrors train_change_detection main()) ───


def _cd_model_from_cfg(cfg: dict) -> torch.nn.Module:
    from models.siamese_swin import build_siamese_swin_cd
    from models.siamese_unet import SiameseUNet
    from models.siamese_vit import build_siamese_vit_cd

    model_name = cfg.get("model", "vit")
    if model_name == "unet":
        return SiameseUNet(in_channels=cfg["in_channels"])
    if model_name == "swin":
        return build_siamese_swin_cd(cfg)
    model_cfg = {k: cfg[k] for k in (
        "img_size", "patch_size", "in_channels",
        "embed_dim", "depth", "num_heads", "mlp_ratio",
        "dropout", "attn_dropout",
        "diff_type", "diff_out_dim", "decoder_dims",
    )}
    return build_siamese_vit_cd(model_cfg)


@pytest.mark.parametrize(
    ("rel_yaml", "expected_model"),
    [
        ("configs/train_config.yaml", "vit"),
        ("configs/train_unet_config.yaml", "unet"),
        ("configs/train_swin_config.yaml", "swin"),
    ],
)
def test_levir_yaml_merges_and_cd_model_forward(
    rel_yaml: str, expected_model: str
) -> None:
    from scripts.train_change_detection import _build_cfg, _parse_args

    cfg_path = REPO_ROOT / rel_yaml
    assert cfg_path.is_file(), f"missing {cfg_path}"
    args = _parse_args(["--config", str(cfg_path)])
    cfg = _build_cfg(args)
    assert cfg.get("model", "vit") == expected_model

    m = _cd_model_from_cfg(cfg)
    m.eval()
    img = 256
    x1 = torch.randn(1, 3, img, img)
    x2 = torch.randn(1, 3, img, img)
    with torch.no_grad():
        logits = m(x1, x2)
    assert logits.shape == (1, 1, img, img)


# ── CIFAR training & ablations: argparse (no DataLoader) ────────────────────


def test_train_cifar10_parse_args_defaults() -> None:
    from scripts.train_cifar10 import parse_args

    args = parse_args([])
    assert args.num_workers == 4
    assert args.seed == 42
    assert args.experiment_name == "vit_cifar10"


def test_run_ablations_parse_args() -> None:
    from scripts.run_ablations import parse_args

    args = parse_args([])
    assert args.ablation_epochs == 50
    assert args.num_workers == 4


def test_run_ablations_only_filter() -> None:
    from scripts.run_ablations import _want

    assert _want("baseline", None) is True
    assert _want("baseline", {"baseline"}) is True
    assert _want("baseline", {"other"}) is False


# ── Scripts: --help and lightweight runs ─────────────────────────────────────


def _run_help(script: str) -> None:
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / script), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert r.returncode == 0, f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}"


@pytest.mark.parametrize(
    "script",
    [
        "scripts/train_change_detection.py",
        "scripts/train_cifar10.py",
        "scripts/run_ablations.py",
        "scripts/plot_ablation_results.py",
        "scripts/export_ablation_json.py",
        "scripts/visualize_predictions.py",
        "scripts/visualize_training.py",
        "scripts/visualize_oscd.py",
    ],
)
def test_script_help_exits_zero(script: str) -> None:
    _run_help(script)


def test_verify_pytorch_stack_runs() -> None:
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "verify_pytorch_stack.py")],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert r.returncode == 0, r.stderr


def test_inspect_training_progress_with_temp_history(tmp_path: Path) -> None:
    hist = tmp_path / "vit_cifar10_history.pt"
    torch.save(
        {"train_loss": [0.5, 0.4], "val_acc": [55.0, 60.0]},
        hist,
    )
    r = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "inspect_training_progress.py"),
            str(hist),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert "Epochs completed" in r.stdout


def test_export_ablation_json_missing_input_exits_nonzero() -> None:
    r = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "export_ablation_json.py"),
            "--input",
            str(REPO_ROOT / "outputs" / "ablations" / "__definitely_missing__.pt"),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert r.returncode != 0


def test_scripts_paths_repo_root() -> None:
    from scripts._paths import repo_root as rr, setup_sys_path

    assert rr() == REPO_ROOT
    assert setup_sys_path() == REPO_ROOT


def test_import_cli_modules() -> None:
    import scripts.plot_all_training_curves  # noqa: F401
    import scripts.train_change_detection  # noqa: F401
