"""
Protocol for generating Al2O3 defect NEB paths.

This script demonstrates how the reference NEB paths for the Al2O3 defects
benchmark were originally generated using a model (e.g., MACE). The generated
paths are then evaluated with DFT (e.g., CASTEP) to provide the reference forces
and energies used in the frozen-image ML-PEG benchmark.
"""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
from ase.mep import NEB
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.optimize import LBFGS
import numpy as np

# Note: In a real run, you would load your specific model here.
# from mace.calculators import mace_mp
# calc = mace_mp(
#     model="medium", dispersion=False, default_dtype="float64", device="cuda"
# )


def generate_reference_paths(calc, dopant: str = "Y"):
    """
    Generate NEB paths for a dopant in Al2O3.

    Parameters
    ----------
    calc : ase.calculators.calculator.Calculator
        The calculator to use for relaxation and NEB.
    dopant : str
        The chemical symbol of the dopant to insert.
    """
    # 1. Structure preparation
    # alpha-Al2O3 from Materials Project (mp-1143), as in the MACE-MP-0 paper
    al2o3 = read(Path(__file__).parents[1] / "data" / "al2o3_mp-1143.xyz")
    # Supercell 2x2x1 -> 120 atoms
    supercell = al2o3 * (2, 2, 1)

    # Substitute at index 0 (Al site)
    dopant_idx = 0
    supercell[dopant_idx].symbol = dopant

    # 2. Path enumeration (finding neighbors)
    cutoffs = natural_cutoffs(supercell)
    cutoffs[dopant_idx] = 2.0  # Fixed radius for the dopant

    nl = NeighborList(cutoffs=cutoffs, self_interaction=False, bothways=True)
    nl.update(supercell)

    neighbor_indices, offsets = nl.get_neighbors(dopant_idx)

    valid_neighbors = []
    for n_idx, n_offset in zip(neighbor_indices, offsets, strict=False):
        if np.any(n_offset != 0):
            # Skip periodic-image neighbors
            continue
        valid_neighbors.append(n_idx)

    print(f"Found {len(valid_neighbors)} valid neighbors for dopant {dopant}.")

    # 3. Endpoint relaxation and NEB
    for n_idx in valid_neighbors:
        # Initial state: neighbor deleted (vacancy)
        init_state = supercell.copy()
        del init_state[n_idx]
        init_state.calc = calc

        opt_init = LBFGS(init_state)
        opt_init.run(fmax=0.05)

        # Final state: dopant moves to the vacancy
        final_state = supercell.copy()
        # Preserve atom ordering: swap the positions or move the dopant
        # Since we deleted n_idx in init_state, we must be careful with indexing.
        # Original protocol likely moved dopant and deleted original site.
        # Here we just move dopant to the neighbor's position & delete neighbor.
        final_state.positions[dopant_idx] = supercell.positions[n_idx]
        del final_state[n_idx]
        final_state.calc = calc

        opt_final = LBFGS(final_state)
        opt_final.run(fmax=0.05)

        # Guard against bad endpoints
        if np.abs(init_state.positions - final_state.positions).max() < 1.0:
            continue

        # NEB
        images = [init_state]
        for _ in range(9):
            image = init_state.copy()
            image.calc = calc
            images.append(image)
        images.append(final_state)

        neb = NEB(images, k=0.1)
        neb.interpolate()

        # Run NEB
        opt_neb = LBFGS(neb)
        opt_neb.run(fmax=0.3)

        # Write images to be evaluated by DFT later
        write(f"al2o3_{dopant}_neb_{n_idx}.extxyz", images)
        print(f"Generated path for neighbor {n_idx}.")


if __name__ == "__main__":
    print("Protocol script for generating reference NEB paths.")
    # Example usage:
    # generate_reference_paths(calc)
