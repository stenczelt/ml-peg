"""Run Al2O3 defects app."""

from __future__ import annotations

from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.load import read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Al2O3 Dopant NEBs"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/nebs.html#al2o3-defects"
)
DATA_PATH = APP_ROOT / "data" / "nebs" / "al2o3_defects"
INFO_PATH = DATA_PATH / "info.json"


class Al2O3DefectsApp(BaseApp):
    """Al2O3 defects benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        force_plots = {}
        neb_plots = {}
        for model in MODELS:
            try:
                force_plots[model] = read_plot(
                    DATA_PATH / f"figure_{model}_all_dopants_forces.json",
                    id=f"{BENCHMARK_NAME}-figure-force-{model}",
                )
            except FileNotFoundError:
                pass

            try:
                neb_plots[model] = read_plot(
                    DATA_PATH / model / "figure_hop_barriers.json",
                    id=f"{BENCHMARK_NAME}-figure-neb-{model}",
                )
            except FileNotFoundError:
                pass

        from ml_peg.app.utils.build_callbacks import plot_from_table_cell

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            plots={
                "Force MAE (all dopants)": force_plots,
                "NEB failure rate": neb_plots,
            },
        )


def get_app() -> Al2O3DefectsApp:
    """
    Get Al2O3 defects benchmark app layout and callback registration.

    Returns
    -------
    Al2O3DefectsApp
        Benchmark layout and callback registration.
    """
    return Al2O3DefectsApp(
        name=BENCHMARK_NAME,
        description="PES accuracy of dopant NEB paths in alpha-Al2O3.",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "al2o3_defects_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
        info_path=INFO_PATH,
    )
