"""Analyse Al2O3 defects DFT singlepoints (forces)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ase.atoms import Atoms
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity, plot_scatter
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


# A hop fails if its NEB crashed (NaN) or its hump above the higher endpoint exceeds
# this sanity cap; measuring from the higher endpoint keeps uphill hops (barrier ~ dE,
# e.g. Y moving into an O site) from counting as blow-ups.
BARRIER_LIMIT = 5.0  # eV

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
    def plot_forces() -> dict[str, list[float]]:
        """
        Plot force parity for a single model and case.

        Returns
        -------
        dict[str, list[float]]
            Reference and predicted force components.
        """
        return {"ref": results.ref_forces_flat, model: results.pred_forces_flat}

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
    case_results: dict[tuple[str, str], _Results],
    failure_rate: dict[str, float],
) -> dict[str, dict[str, float]]:
    """
    Build raw metric dict for the benchmark table.

    Parameters
    ----------
    case_results
        Results indexed by ``(case_key, model_name)``.
    failure_rate
        Fraction of failed dopant-hop NEBs per model.

    Returns
    -------
    dict[str, dict[str, float]]
        Metric values for all models.
    """
    metrics: dict[str, dict[str, float]] = {}

    metrics["NEB failure rate"] = failure_rate

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
    Run analysis for Al2O3 defects DFT singlepoints and dopant-hop NEBs.

    Parameters
    ----------
    metrics
        Benchmark metrics table.
    """
    write_struct_info(
        data_path=[
            *CALC_PATH.glob("mock/*/al2o3_defects.extxyz"),
            *CALC_PATH.glob("mock/stability/*-neb-band.extxyz"),
        ],
        out_path=OUT_PATH,
    )


def _hop_barriers(model_name: str) -> dict[str, float]:
    """
    Read the NEB hump above the higher endpoint for every dopant hop of a model.

    Parameters
    ----------
    model_name
        Name of the model.

    Returns
    -------
    dict[str, float]
        ``barrier - max(dE, 0)`` (eV, NaN if the NEB failed) indexed by hop label.
    """
    barriers = {}
    for path in sorted((CALC_PATH / model_name / "stability").glob("*.dat")):
        barrier, delta_e, _ = np.loadtxt(path)
        barriers[path.name.removesuffix("-neb-results.dat")] = float(
            barrier - max(delta_e, 0.0)
        )
    return barriers


def _plot_hop_barriers(model_name: str, barriers: dict[str, float]) -> None:
    """
    Plot the barrier of every dopant hop for a model.

    Parameters
    ----------
    model_name
        Name of the model.
    barriers
        Barrier indexed by hop label.
    """

    @plot_scatter(
        filename=OUT_PATH / model_name / "figure_hop_barriers.json",
        title=f"Dopant-hop NEB barriers ({model_name})",
        x_label="Hop",
        y_label="Barrier above higher endpoint / eV",
        hoverdata={"Hop": list(barriers)},
        horizontal_lines=[{"y": BARRIER_LIMIT, "name": "failure cap"}],
    )
    def plot_barriers() -> dict[str, list[list]]:
        """
        Plot hop barriers.

        Returns
        -------
        dict[str, list[list]]
            Mapping of model name to x/y arrays.
        """
        return {model_name: [list(range(len(barriers))), list(barriers.values())]}

    plot_barriers()


@pytest.fixture
def failure_rate() -> dict[str, float]:
    """
    Get the fraction of dopant-hop NEBs that crashed or blew up per model.

    Returns
    -------
    dict[str, float]
        Failure fraction in [0, 1], or NaN if no hops were run.
    """
    rates = {}
    for model_name in MODELS:
        barriers = _hop_barriers(model_name)
        if not barriers:
            rates[model_name] = float("nan")
            continue
        _plot_hop_barriers(model_name, barriers)
        failed = [not b <= BARRIER_LIMIT for b in barriers.values()]  # NaN fails
        rates[model_name] = float(np.mean(failed))
    return rates
