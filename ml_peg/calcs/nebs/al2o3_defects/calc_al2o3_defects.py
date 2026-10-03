"""Evaluate MLIPs on Al2O3 defects DFT singlepoints (energies + forces)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any
from warnings import warn

from ase.atoms import Atoms
from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs
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


# Test 1 (stability): short model-driven NEBs over every nearest-neighbour hop of a
# dopant. Edit these to widen the sweep; each hop is an independent test.
STABILITY_DOPANTS = ("Y",)
STABILITY_SITES = (0, 12)  # 0: Al site, 12: O site, as in the MACE-MP-0 paper
DOPANT_CUTOFF = 2.0  # Å, neighbour-search radius for the dopant (paper value)
RELAX_STEPS = 10
NEB_STEPS = 10  # ponytail: enough to expose blow-ups, not to converge the band
N_IMAGES = 5


def _supercell() -> Atoms:
    """
    Build the 120-atom alpha-Al2O3 supercell.

    Returns
    -------
    ase.atoms.Atoms
        2x2x1 supercell of mp-1143, as in the MACE-MP-0 paper.
    """
    atoms = read(DATA_PATH / "al2o3_mp-1143.xyz") * (2, 2, 1)
    atoms.info.update(charge=0, spin=1)
    return atoms


def _enumerate_hops() -> list[tuple[str, int, int]]:
    """
    Find all in-cell nearest-neighbour hops for each dopant and site.

    Returns
    -------
    list[tuple[str, int, int]]
        ``(dopant, site, neighbour)`` triples.
    """
    hops = []
    for dopant in STABILITY_DOPANTS:
        for site in STABILITY_SITES:
            atoms = _supercell()
            atoms[site].symbol = dopant
            cutoffs = natural_cutoffs(atoms)
            cutoffs[site] = DOPANT_CUTOFF
            # bothways: the paper's one-way list missed neighbours below the site index
            nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
            nl.update(atoms)
            for neighbour, offset in zip(*nl.get_neighbors(site), strict=True):
                if not offset.any():  # skip periodic images
                    hops.append((dopant, site, int(neighbour)))
    return hops


HOPS = _enumerate_hops()


def _hop_endpoints(dopant: str, site: int, neighbour: int) -> tuple[Atoms, Atoms]:
    """
    Build unrelaxed, index-aligned endpoints for a dopant hop.

    Start: dopant on ``site`` with ``neighbour`` removed. End: dopant moved into the
    vacated ``neighbour`` position.

    Parameters
    ----------
    dopant
        Chemical symbol of the dopant.
    site
        Index of the substituted site in the supercell.
    neighbour
        Index of the neighbouring atom to remove.

    Returns
    -------
    tuple[ase.atoms.Atoms, ase.atoms.Atoms]
        Start and end structures.
    """
    atoms = _supercell()
    atoms[site].symbol = dopant
    start = atoms.copy()
    del start[neighbour]
    end = start.copy()
    end.positions[site - (neighbour < site)] = atoms.positions[neighbour]
    return start, end


@pytest.mark.slow
@pytest.mark.parametrize("hop", HOPS, ids=lambda h: "{}_{}_to_{}".format(*h))
@pytest.mark.parametrize("mlip", MODELS.items())
def test_al2o3_dopant_stability(
    mlip: tuple[str, Any], hop: tuple[str, int, int]
) -> None:
    """
    Run a short model-driven NEB for one dopant hop.

    Writes ``<label>-neb-results.dat``; the barrier is NaN if the run fails.

    Parameters
    ----------
    mlip
        Tuple of ``(model_name, model)`` as provided by ``MODELS.items()``.
    hop
        ``(dopant, site, neighbour)`` triple.
    """
    model_name, model = mlip
    label = "{}_{}_to_{}".format(*hop)
    out_dir = OUT_PATH / model_name / "stability"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        start, end = _hop_endpoints(*hop)
        start.calc = model.get_calculator(precision="high")
        end.calc = start.calc
        neb = NEB(
            init_struct=start,
            final_struct=end,
            n_images=N_IMAGES,
            interpolator="ase",  # index-aligned; pymatgen autosort fails on the hop
            minimize=True,
            minimize_kwargs={"filter_class": None, "steps": RELAX_STEPS},
            write_band=True,
            file_prefix=out_dir / label,
            steps=NEB_STEPS,
        )
        neb.run()
    except Exception as exc:
        warn(f"Error running NEB {label} for {model_name}: {exc}", stacklevel=2)
        with open(out_dir / f"{label}-neb-results.dat", "w", encoding="utf8") as out:
            print("#Barrier [eV] | delta E [eV] | Max force [eV/Å] ", file=out)
            print(np.nan, np.nan, np.nan, file=out)
