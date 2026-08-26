"""Analyse Al2O3 defects DFT singlepoints (forces)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ase.atoms import Atoms
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import load_metrics_config, mae, write_struct_info
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)

BENCHMARK_DIR = "al2o3_defects"
CALC_PATH = CALCS_ROOT / "nebs" / BENCHMARK_DIR / "outputs"
OUT_PATH = APP_ROOT / "data" / "nebs" / BENCHMARK_DIR

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


@dataclass(frozen=True)
class _Case:
    """Definition of a single Al2O3 defects dataset."""

    key: str
    label: str


CASES: tuple[_Case, ...] = (_Case(key="all_dopants", label="all dopants"),)


@dataclass(frozen=True)
class _Results:
    """Computed errors and assets for one (case, model) pair."""

    force_mae: float
    ref_forces_flat: list[float]
    pred_forces_flat: list[float]
    structs: list[Atoms]


def _load_structs(case_key: str, model: str) -> list[Atoms] | None:
    """
    Load MLIP-evaluated NEB frames (with DFT reference stored).

    Parameters
    ----------
    case_key
        Case key.
    model
        Model name.

    Returns
    -------
    list[ase.atoms.Atoms] | None
        Frames if the calculation output exists, otherwise ``None``.
    """
    path = CALC_PATH / model / case_key / f"{BENCHMARK_DIR}.extxyz"
    if not path.exists():
        return None
    structs = read(path, index=":")
    if not isinstance(structs, list) or not structs:
        raise ValueError(f"Unexpected output content: {path}")
    return structs


def _compute(structs: list[Atoms]) -> _Results:
    """
    Compute global force MAEs.

    Parameters
    ----------
    structs
        Ordered frames with ``ref_forces`` and ``pred_forces``.

    Returns
    -------
    _Results
        Computed errors and frames for downstream plotting/app.
    """
    ref_f = np.concatenate([s.arrays["ref_forces"].flatten() for s in structs])
    pred_f = np.concatenate([s.arrays["pred_forces"].flatten() for s in structs])

    return _Results(
        force_mae=mae(ref_f, pred_f),
        ref_forces_flat=ref_f.tolist(),
        pred_forces_flat=pred_f.tolist(),
        structs=structs,
    )


def _write_assets(case_key: str, model: str, results: _Results) -> None:
    """
    Write structure trajectory for the app.

    Parameters
    ----------
    case_key
        Case key.
    model
        Model name.
    results
        Computed errors and frames.
    """
    out_dir = OUT_PATH / case_key / model
    out_dir.mkdir(parents=True, exist_ok=True)
    write(out_dir / f"{model}-frames.extxyz", results.structs)


def _write_plots(case_key: str, case_label: str, model: str, results: _Results) -> None:
    """
    Write Dash/Plotly JSON plots for one (case, model) pair.

    Parameters
    ----------
    case_key
        Case key.
    case_label
        Human-friendly case label.
    model
        Model name.
    results
        Computed errors and frames.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    @plot_parity(
        filename=OUT_PATH / f"figure_{model}_{case_key}_forces.json",
        title=f"Force comparison ({case_label})",
        x_label="Predicted force component / (eV/Å)",
        y_label="Reference force component / (eV/Å)",
    )
    def plot_forces() -> dict[str, tuple[list[float], list[float]]]:
        """
        Plot force parity for a single model and case.

        Returns
        -------
        dict[str, tuple[list[float], list[float]]]
            Mapping of model name to pred/ref arrays.
        """
        return {model: [results.pred_forces_flat, results.ref_forces_flat]}

    plot_forces()


@pytest.fixture
def case_results() -> dict[tuple[str, str], _Results]:
    """
    Compute results for all available (case, model) combinations.

    Returns
    -------
    dict[tuple[str, str], _Results]
        Results indexed by ``(case_key, model_name)``.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    results: dict[tuple[str, str], _Results] = {}
    for case in CASES:
        for model in MODELS:
            structs = _load_structs(case.key, model)
            if structs is None:
                continue
            computed = _compute(structs)
            _write_assets(case.key, model, computed)
            _write_plots(case.key, case.label, model, computed)
            results[(case.key, model)] = computed
    return results


@pytest.fixture
def metrics_dict(
    case_results: dict[tuple[str, str], _Results], barrier_y: dict[str, float]
) -> dict[str, dict[str, float]]:
    """
    Build raw metric dict for the benchmark table.

    Parameters
    ----------
    case_results
        Results indexed by ``(case_key, model_name)``.
    barrier_y
        Barrier heights for Yttrium.

    Returns
    -------
    dict[str, dict[str, float]]
        Metric values for all models.
    """
    metrics: dict[str, dict[str, float]] = {}

    metrics["Barrier (Y)"] = barrier_y

    for case in CASES:
        force_key = f"Force MAE ({case.label})"
        metrics[force_key] = {}
        for model in MODELS:
            result = case_results.get((case.key, model))
            if result is None:
                continue
            metrics[force_key][model] = result.force_mae
    return metrics


@pytest.fixture
@build_table(
    filename=OUT_PATH / f"{BENCHMARK_DIR}_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(metrics_dict: dict[str, dict[str, float]]) -> dict[str, dict]:
    """
    Build the benchmark table JSON.

    Parameters
    ----------
    metrics_dict
        Metric values for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return metrics_dict


def test_al2o3_defects(metrics: dict[str, dict]) -> None:
    """
    Run analysis for Al2O3 defects DFT singlepoints.

    Parameters
    ----------
    metrics
        Benchmark metrics table.
    """
    # Write a mock structure info so the app can visualize something
    write_struct_info(
        data_path=list(CALC_PATH.glob("mock/*/al2o3_defects.extxyz")),
        out_path=OUT_PATH,
        index=0,
    )

    write_struct_info(
        data_path=list(CALC_PATH.glob("mock/*/al2o3_Y_neb-neb-band.extxyz")),
        out_path=OUT_PATH,
        index=0,
    )


def plot_al2o3_neb(model_name: str) -> None:
    """
    Plot the Yttrium NEB path and save structure files.

    Parameters
    ----------
    model_name
        Name of the model.
    """
    from ml_peg.analysis.utils.decorators import plot_scatter

    @plot_scatter(
        filename=OUT_PATH / model_name / "figure_neb_Y.json",
        title="NEB path (Yttrium)",
        x_label="Image",
        y_label="Energy / eV",
        show_line=True,
    )
    def plot_neb() -> dict[str, tuple[list[float], list[float]]]:
        """
        Plot a single NEB band for Yttrium.

        Returns
        -------
        dict[str, tuple[list[float], list[float]]]
            Mapping of model name to x/y arrays.
        """
        results = {}
        structs = read(
            CALC_PATH / model_name / "al2o3_Y_neb-neb-band.extxyz",
            index=":",
        )
        results[model_name] = [
            list(range(len(structs))),
            [struct.get_potential_energy() for struct in structs],
        ]
        structs_dir = OUT_PATH / model_name
        structs_dir.mkdir(parents=True, exist_ok=True)
        write(structs_dir / "al2o3_Y-neb-band.extxyz", structs)
        return results

    plot_neb()


@pytest.fixture
def barrier_y() -> dict[str, float]:
    """
    Get Yttrium diffusion barrier.

    Returns
    -------
    dict[str, float]
        Barrier heights for all models.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    results = {}
    for model_name in MODELS:
        try:
            plot_al2o3_neb(model_name)
            with open(
                CALC_PATH / model_name / "al2o3_Y_neb-neb-results.dat",
                encoding="utf8",
            ) as f:
                data = f.readlines()
                pred_barrier, _, _ = tuple(float(x) for x in data[1].split())
            results[model_name] = pred_barrier
        except (FileNotFoundError, KeyError, IndexError):
            results[model_name] = float("nan")
    return results
