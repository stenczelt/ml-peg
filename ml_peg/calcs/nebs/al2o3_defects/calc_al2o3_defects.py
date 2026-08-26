"""Evaluate MLIPs on Al2O3 defects DFT singlepoints (energies + forces)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any
from warnings import warn

from ase.atoms import Atoms
from ase.build import bulk
from ase.io import read, write
from janus_core.calculations.geom_opt import GeomOpt
from janus_core.calculations.neb import NEB
import numpy as np
import pytest
from tqdm.auto import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"
S3_KEY = "inputs/nebs/al2o3_defects/al2o3_defects.zip"
S3_FILENAME = "al2o3_defects.zip"


@dataclass(frozen=True)
class _Case:
    """Definition of a single Al2O3 defects NEB dataset."""

    key: str
    ref_file: str


CASES: tuple[_Case, ...] = (_Case(key="all_dopants", ref_file="rmse-input.xyz"),)


def _read_frames(path: Path) -> list[Atoms]:
    """
    Read all frames from an xyz trajectory.

    Parameters
    ----------
    path
        Path to xyz trajectory.

    Returns
    -------
    list[ase.atoms.Atoms]
        Frames as ASE atoms objects.
    """
    frames = read(path, index=":")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames found in {path}")
    return frames


def _ref_energy_ev(atoms: Atoms) -> float:
    """
    Extract reference (DFT) energy in eV from a frame.

    Parameters
    ----------
    atoms
        Frame with ``CASTEP_energy`` in ``atoms.info``.

    Returns
    -------
    float
        Reference energy in eV.
    """
    if "CASTEP_energy" in atoms.info:
        return float(atoms.info["CASTEP_energy"])
    if "ref_energy_ev" in atoms.info:
        return float(atoms.info["ref_energy_ev"])
    raise KeyError("Missing CASTEP_energy or ref_energy_ev in reference trajectory.")


def _ref_forces(atoms: Atoms) -> np.ndarray:
    """
    Extract reference (DFT) forces in eV/Å from a frame.

    Parameters
    ----------
    atoms
        Frame with ``CASTEP_forces`` in ``atoms.arrays``.

    Returns
    -------
    numpy.ndarray
        Reference forces array with shape ``(n_atoms, 3)``.
    """
    if "CASTEP_forces" in atoms.arrays:
        return np.asarray(atoms.arrays["CASTEP_forces"], dtype=float)
    if "ref_forces" in atoms.arrays:
        return np.asarray(atoms.arrays["ref_forces"], dtype=float)
    raise KeyError("Missing CASTEP_forces or ref_forces in reference trajectory.")


@pytest.mark.slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_al2o3_defects(mlip: tuple[str, Any]) -> None:
    """
    Compare MLIP energies/forces to DFT along fixed NEB images for doped Al2O3.

    Outputs per-case trajectories containing:
    - ref_energy_ev, ref_forces
    - pred_energy_ev, pred_forces

    Parameters
    ----------
    mlip
        Tuple of ``(model_name, model)`` as provided by ``MODELS.items()``.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")

    local_files_present = all((DATA_PATH / case.ref_file).exists() for case in CASES)
    if local_files_present:
        data_dir = DATA_PATH
    else:
        data_dir = download_s3_data(key=S3_KEY, filename=S3_FILENAME) / "al2o3_defects"

    for case in CASES:
        frames = _read_frames(data_dir / case.ref_file)
        out_dir = OUT_PATH / model_name / case.key
        out_dir.mkdir(parents=True, exist_ok=True)

        results: list[Atoms] = []
        it = frames
        # Only show tqdm progress bars in an interactive terminal; otherwise the
        # carriage-return updates tend to spam CI/log outputs.
        if sys.stderr.isatty():
            it = tqdm(
                frames,
                desc=f"{model_name} {case.key}",
                unit="img",
                leave=False,
            )
        for atoms in it:
            ref_energy_ev = _ref_energy_ev(atoms)
            ref_forces = _ref_forces(atoms)

            atoms_pred = atoms.copy()
            atoms_pred.calc = calc
            # Set default charge and spin
            atoms_pred.info.setdefault("charge", 0)
            atoms_pred.info.setdefault("spin", 1)

            out_atoms = atoms.copy()
            out_atoms.info["ref_energy_ev"] = ref_energy_ev
            out_atoms.arrays["ref_forces"] = ref_forces

            # Note on ordering:
            # The CASTEP references in the memo reorder atoms on output.
            # In a full implementation, the geometries supplied in rmse-input.xyz
            # are expected to already be in CASTEP-order so the atoms map 1-to-1.

            try:
                pred_energy = float(atoms_pred.get_potential_energy())
                pred_forces = np.asarray(atoms_pred.get_forces(), dtype=float)
            except Exception as exc:
                warn(
                    f"Error calculating energy for {case}: {exc}",
                    stacklevel=2,
                )
                pred_energy = np.nan
                pred_forces = np.full((len(atoms_pred), 3), np.nan)
            out_atoms.info["pred_energy_ev"] = pred_energy
            out_atoms.arrays["pred_forces"] = pred_forces
            results.append(out_atoms)

        write(out_dir / "al2o3_defects.extxyz", results)


@pytest.fixture(scope="module")
def al2o3_unrelaxed_endpoints() -> tuple[Atoms, Atoms]:
    """
    Generate unrelaxed endpoints for Yttrium diffusion.

    Returns
    -------
    tuple[ase.atoms.Atoms, ase.atoms.Atoms]
        The initial and final states.
    """
    al2o3 = bulk("Al2O3", crystalstructure="corundum", a=4.76, c=12.99)
    supercell = al2o3 * (2, 2, 1)
    supercell[0].symbol = "Y"

    init_state = supercell.copy()
    del init_state[1]  # an adjacent Al atom

    final_state = supercell.copy()
    final_state.positions[0] = supercell.positions[1]
    del final_state[1]

    return init_state, final_state


@pytest.fixture(scope="module")
def al2o3_relaxed_endpoints(
    al2o3_unrelaxed_endpoints: tuple[Atoms, Atoms]
) -> dict[str, tuple[Atoms, Atoms]]:
    """
    Relax endpoints for all models.

    Parameters
    ----------
    al2o3_unrelaxed_endpoints
        Unrelaxed endpoints from fixture.

    Returns
    -------
    dict[str, tuple[ase.atoms.Atoms, ase.atoms.Atoms]]
        Relaxed endpoints per model.
    """
    relaxed_endpoints = {}
    init_unrelaxed, final_unrelaxed = al2o3_unrelaxed_endpoints

    for model_name, model in MODELS.items():
        calc = model.get_calculator(precision="high")
        init_state = init_unrelaxed.copy()
        final_state = final_unrelaxed.copy()
        init_state.calc = calc
        final_state.calc = calc

        # We limit steps to 50 for CI speed
        try:
            opt_init = GeomOpt(
                struct=init_state,
                write_results=True,
                file_prefix=OUT_PATH / model_name / "al2o3_Y_start",
                steps=50,
            )
            opt_init.run()
            init_relaxed = opt_init.struct

            opt_final = GeomOpt(
                struct=final_state,
                write_results=True,
                file_prefix=OUT_PATH / model_name / "al2o3_Y_end",
                steps=50,
            )
            opt_final.run()
            final_relaxed = opt_final.struct

            relaxed_endpoints[model_name] = (init_relaxed, final_relaxed)
        except Exception as exc:
            warn(f"Error relaxing endpoints for {model_name}: {exc}", stacklevel=2)
            relaxed_endpoints[model_name] = (init_state, final_state)  # fallback

    return relaxed_endpoints


@pytest.mark.slow
@pytest.mark.parametrize("model_name", MODELS)
def test_al2o3_defects_barrier(
    al2o3_relaxed_endpoints: dict[str, tuple[Atoms, Atoms]],
    model_name: str
) -> None:
    """Run model-driven NEB to find the barrier for Y diffusion.

    Parameters
    ----------
    al2o3_relaxed_endpoints
        Relaxed endpoints per model.
    model_name
        Name of model to use.
    """
    init_struct, final_struct = al2o3_relaxed_endpoints[model_name]

    neb = NEB(
        init_struct=init_struct,
        final_struct=final_struct,
        n_images=7,  # Fewer images for speed
        interpolator="pymatgen",
        minimize=True,
        plot_band=True,
        write_band=True,
        file_prefix=OUT_PATH / model_name / "al2o3_Y_neb",
        steps=50,  # Fewer steps for speed
    )

    try:
        neb.interpolate()
        neb.interpolator = None
        for image in neb.images:
            image.info.setdefault("charge", 0)
            image.info.setdefault("spin", 1)
        neb.run()
    except Exception as e:
        warn(f"Error running NEB for {model_name}: {e}", stacklevel=2)
        # Write out nan results
        out_dir = OUT_PATH / model_name
        out_dir.mkdir(exist_ok=True, parents=True)
        write(out_dir / "al2o3_Y_neb-neb-band.extxyz", [init_struct, final_struct])
        with open(out_dir / "al2o3_Y_neb-neb-results.dat", "w", encoding="utf8") as out:
            print("#Barrier [eV] | delta E [eV] | Max force [eV/Å] ", file=out)
            print(np.nan, np.nan, np.nan, file=out)
