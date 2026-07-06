#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ULTRAMETRICITY ANALYSIS OF PROTEIN ENERGY LANDSCAPES
BASED ON KINETIC METRICS AND MARKOV STATE MODELS (MSM)
Author: A. P. Zubarev
Version: 1.0
================================================================================
METHOD DESCRIPTION AND PROGRAM PURPOSE
--------------------------------------------------------------------------------
This program implements a method for testing the hypothesis of ultrametric
organization in protein energy landscapes. The method is based on analyzing
transition kinetics between macrostates (basins of attraction) using a kinetic
metric constructed via spectral decomposition of the symmetrized Kramers
transition rate matrix and the Mahalanobis distance.

The program performs the following computational stages:

1. MOLECULAR DYNAMICS (MD) STAGE
   Generation of an equilibrium conformational ensemble via Langevin dynamics
   using the OpenMM package. Supports GB99dms and Amber14/OBC2 implicit solvent
   force fields.
   
   ROBUST STRUCTURE PREPARATION:
   Before MD, the structure is automatically cleaned: non-protein residues
   (ligands, ions, water) are removed, the longest polypeptide chain is selected,
   and missing atoms/hydrogens are added. This ensures compatibility with OpenMM
   force fields for any PDB entry.
   
   IN-MEMORY BUFFERING:
   Instead of writing directly to disk via DCDReporter, the program uses a custom
   MemoryReporter that accumulates coordinates and energies in Python lists.
   This solves the fundamental issue of "garbage" frames during NaN events:
   RAM lists can be truncated atomically, whereas binary DCD files cannot be
   rolled back. A single DCD file is written to disk via mdtraj ONLY ONCE at
   the very end, after successful simulation completion.
   
   If rollbacks exceed the limit (MD_MAX_ROLLBACKS), the integration step is
   automatically reduced and the simulation restarts from scratch.

2. MARKOV STATE MODEL (MSM) CONSTRUCTION STAGE
   Continuous conformational space is discretized by clustering trajectory
   frames into microstates using the k-centers algorithm in RMSD metric.
   The choice of atoms for RMSD calculation is determined by GEOMETRY_RMSD_MODE:
   - 'ca':  C-alpha atoms only (faster, reflects backbone folding).
   - 'all': All-atom RMSD (stricter, accounts for side chains).
   
   IMPORTANT: The selected metric is used UNIFORMLY for both clustering and
   neighbor graph construction, ensuring internal consistency of space
   discretization.
   
   For each microstate (cluster), free energy is computed via the partition
   function over frame energies within that cluster.
   
   A neighbor graph is constructed using one of two methods:
   - Energy-based: Two clusters are neighbors if their free energy difference
     does not exceed threshold dF_max = -RT ln(eta). This criterion is invariant
     to energy shifts and ensures correct connectivity at all energy levels.
   - Adaptive Topological Graph: For each cluster, an effective radius is
     computed (specified percentile of distances from center to cluster points).
     Two clusters are neighbors if the distance between centers does not exceed
     alpha * (R_i + R_j), where alpha is NEIGHBOR_ALPHA.
   
   Based on free energies and the neighbor graph, a symmetrized transition rate
   matrix K_sym is built. Elements K_sym_ij are computed via Kramers formula:
   exp(-|F_i - F_j| / (2RT)) for neighboring microstates, 0 otherwise.
   
   Macrostates (basins of attraction) are identified via gradient descent on
   the free energy landscape. Attractors (local minima and plateaus) are
   identified automatically; each microstate is assigned to the basin reached
   by gradient descent. BASIN_EQ_THRESHOLD defines the energy equivalence
   threshold below which states belong to the same plateau.
   
   CRITICAL ALGORITHM NOTE:
   1. Attractors are identified first (states with no neighbors strictly lower
      than F_i - epsilon_eq).
   2. Equivalence subgraph analysis is performed ONLY among attractors to merge
      degenerate minima/plateaus correctly.
   3. Gradient descent applies exclusively to unlabeled slope states.
   
   Analysis is performed strictly by connected components. For each component
   with >= MIN_BASINS_PER_COMPONENT basins, ultrametricity is computed
   independently. Final ultrametricity is a weighted average. Fragmentation
   index f_inter is also computed.

3. SPECTRAL ANALYSIS AND ULTRAMETRICITY TESTING STAGE
   For each significant connected component, spectral decomposition of K_sym_c
   is performed. Noise modes are filtered via spectral gap criterion.
   
   Basin characteristic vectors are projected onto physical eigenmodes.
   Distance between basins is the Mahalanobis distance: weighted Euclidean norm
   of projection differences with weights 1/|lambda_k|. Slower relaxation modes
   contribute more to the distance.
   
   Ultrametricity is tested by enumerating all basin triplets. A triplet is
   non-trivially ultrametric if the two largest distances are approximately
   equal (relative diff <= epsilon) and the third is significantly smaller
   (relative diff > delta). The fraction of such triplets is u_nt.

4. EXECUTION ARCHITECTURE AND MODES
   The program operates in one or more of the following modes:
   
   - FROM_FILES_ANALYSIS: Skip MD, load existing DCD/CIF/PDB files. MSM is
     built once, then enabled analytical modes run sequentially. MAIN_ANALYSIS
     is forced to 1 sample in this mode.
   
   - SENSITIVITY_ANALYSIS: Run analysis for multiple BASIN_EQ_THRESHOLD values
     specified in SENSITIVITY_RANGE. Cannot be combined with ALPHA_ANALYSIS.
   
   - ALPHA_ANALYSIS: Run analysis for multiple NEIGHBOR_ALPHA values specified
     in ALPHA_RANGE. Cannot be combined with SENSITIVITY_ANALYSIS.
   
   - NULL_H1_ENERGY_SHUFFLE: Perform baseline calculation, fix topology/basins,
     then run NULL_H1_NUM_SAMPLES permutations of free energies to estimate
     statistical significance (p-value).
   
   - MAIN_ANALYSIS: Full cycle (MD -> MSM -> Analysis) repeated MAIN_NUM_SAMPLES
     times with different seeds. Computes mean, std, CV for stability assessment.

TECHNICAL REQUIREMENTS
--------------------------------------------------------------------------------
Runtime: Windows 10/11, Linux, macOS. Python 3.10-3.12.
Required packages (pip install):
- openmm, pdbfixer, mdtraj, numpy, scipy, threadpoolctl, torch

HARDWARE REQUIREMENTS
--------------------------------------------------------------------------------
GPU (NVIDIA CUDA) recommended for MD. CPU fallback is automatic.
MSM and spectral analysis stages run on CPU.

USER PARAMETERS
--------------------------------------------------------------------------------
All configurable parameters are grouped below with detailed comments.
================================================================================
"""

import os
import sys
import time
import warnings
import gc
import glob
import numpy as np
import multiprocessing
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# ============================================================================
# IMPORTANT FOR WINDOWS: CUDA JIT COMPILATION SUPPORT (NVRTC)
# ============================================================================
try:
    import torch
except ImportError:
    pass

import openmm
from openmm import unit
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy.spatial.distance import cdist
from itertools import combinations

warnings.filterwarnings('ignore')

# ============================================================================
# USER PARAMETERS
# ============================================================================

# ----------------------------------------------------------------------------
# GROUP 1: EXECUTION MODES AND GENERAL SETTINGS
# ----------------------------------------------------------------------------

MAX_CPU_THREADS = 6
"""
Maximum number of CPU threads for computations (OpenMP, BLAS, LAPACK).
Set this below your total logical core count to leave resources for the OS.
If 0 or None, uses all available logical cores.
Recommended: 22 (for 24-thread CPU) or 0 (auto).
"""

FROM_FILES_ANALYSIS = True
"""
Enable analysis from pre-existing files (skip MD stage).
True  - Load existing DCD, CIF/PDB, and optionally energies.txt files.
        MSM is built ONCE, then enabled analytical modes run sequentially.
        MAIN_ANALYSIS is forced to 1 sample in this mode.
False - Execute full cycle starting from MD for each enabled mode.
File naming convention: {base}_trajectory.dcd, {base}.cif/.pdb, {base}_energies.txt
Recommended: False (initial run), True (re-analysis of existing trajectory).
"""

SENSITIVITY_ANALYSIS = False
"""
Enable sensitivity analysis to BASIN_EQ_THRESHOLD.
If True, runs analysis for each value in SENSITIVITY_RANGE and outputs a table.
Cannot be True simultaneously with ALPHA_ANALYSIS.
Recommended: True (for landscape stability study).
"""

SENSITIVITY_RANGE = [1.0, 3.0, 5.0, 7.0]
"""
List of BASIN_EQ_THRESHOLD values (kJ/mol) for sensitivity analysis.
Used ONLY when SENSITIVITY_ANALYSIS = True.
Empty list [] disables the mode. Single element equals single run.
Recommended: [2.0, 2.5, 3.0] or wider [1.0, 3.0, 5.0, 7.0, 10.0].
"""

ALPHA_ANALYSIS = False
"""
Enable sensitivity analysis to NEIGHBOR_ALPHA parameter.
If True, runs analysis for each value in ALPHA_RANGE and outputs a table.
Cannot be True simultaneously with SENSITIVITY_ANALYSIS.
Recommended: False (or True for topology influence study).
"""

ALPHA_RANGE = [0.5, 0.6, 0.7, 0.8, 1.0]
"""
List of NEIGHBOR_ALPHA values for sensitivity analysis.
Used ONLY when ALPHA_ANALYSIS = True and NEIGHBOR_METHOD = 'adaptive_topology'.
Empty list [] disables the mode.
Recommended: [0.5, 0.6, 0.7, 0.8, 1.0].
"""

NULL_H1_ENERGY_SHUFFLE = False
"""
Enable null hypothesis H1 test (Energy Shuffle).
True  - Baseline calculation fixes topology/basins, then NULL_H1_NUM_SAMPLES
        random permutations of free energies are performed to estimate p-value.
False - H1 test skipped.
Computational cost: ~3-5 seconds per permutation.
Recommended: True.
"""

NULL_H1_NUM_SAMPLES = 100
"""
Number of realizations for H1 null model (Energy Shuffle).
Determines p-value precision.
Recommended: 100 (p-value accuracy ~0.01). Range: 10-1000.
"""

MAIN_ANALYSIS = True
"""
Enable Main Analysis mode.
True  - Full cycle (MD -> Clustering -> MSM -> Analysis) executed MAIN_NUM_SAMPLES
        times with different seeds. Computes mean, std, CV for stability report.
False - Main analysis skipped.
NOTE: In FROM_FILES_ANALYSIS mode, MAIN_NUM_SAMPLES is forced to 1.
Recommended: True (final results), False (quick tests).
"""

MAIN_NUM_SAMPLES = 1
"""
Number of independent runs for MAIN_ANALYSIS.
Each run includes full pipeline: MD -> Clustering -> MSM -> Analysis.
Forced to 1 in FROM_FILES_ANALYSIS mode.
Recommended: 5-10 (balance between reliability and time). Range: 1-100.
"""

NULL_SEED_BASE = 12345
"""
Base seed for RNG in null models and main analysis.
Each realization uses seed = NULL_SEED_BASE + run_index for reproducibility.
Recommended: Any integer (e.g., 12345).
"""

USE_GPU = True
"""
Use GPU (CUDA) for molecular dynamics.
True  - Attempts NVIDIA GPU. Falls back to CPU automatically on failure.
False - Forces CPU-only execution.
Performance: GPU ~2-3 min / 500k steps; CPU ~20-30 min / 500k steps.
Recommended: True (if GPU available).
"""

VERBOSE = True
"""
Verbose console output.
True  - Detailed progress info for all stages.
False - Key results only. Useful for batch runs or log redirection.
Recommended: True.
"""

# ----------------------------------------------------------------------------
# GROUP 2: MOLECULAR DYNAMICS (MD) PARAMETERS
# ----------------------------------------------------------------------------

PDB_ID = "1L2Y"
"""
RCSB PDB identifier for the protein.
Defines starting structure if STARTING_STRUCTURE_FILE is empty.
In FROM_FILES_ANALYSIS mode, used as base name for file search.
Recommended test values: "1L2Y", "1UBQ", "1JSF", "2RVD".
"""

STARTING_STRUCTURE_FILE = ""
"""
Path to custom starting structure (PDB or CIF format).
If empty, downloads native structure from RCSB PDB using PDB_ID.
Using alternative structures helps avoid native trap artifacts.
In FROM_FILES_ANALYSIS mode, its basename is used for file search.
"""

ADD_HYDROGENS_TO_CUSTOM_START = False
"""
Whether to add hydrogens to custom starting structure.
False - File already contains hydrogens.
True  - Heavy atoms only; hydrogens added automatically.
Recommended: False.
"""

USE_GB99DMS = True
"""
Force field selection for MD.
True  - GB99dms implicit solvent (optimized for disordered proteins).
False - Amber14-all with OBC2 implicit solvent (conservative choice).
In FROM_FILES_ANALYSIS mode, also used for energy recalculation if needed.
Recommended: True.
"""

TEMPERATURE_K = 1200.0
"""
Langevin dynamics temperature (Kelvin).
Defines thermal fluctuation scale. High T (800-1200K) enables ergodic sampling
of unfolded states. Analysis temperature (KRAMERS_TEMPERATURE) can differ.
WARNING: Classical force fields may lose validity above 1200K.
Recommended: 800-1000K for global landscape exploration. Range: 200-2000K.
"""

MD_STEPS = 10000000
"""
Total Langevin integration steps.
Effective timestep adjusts automatically for stability.
Frames saved = MD_STEPS / MD_REPORT_INTERVAL.
Recommended: 2,000,000+ for statistical reliability. Range: 10k-100M.
"""

MD_REPORT_INTERVAL = 100
"""
Frame saving interval (in integration steps).
Recommended: 50 (yields ~10k frames for 500k steps). Range: 10-10000.
"""

MINIMIZE_STARTING_STRUCTURE = True
"""
Perform local energy minimization before MD.
Eliminates steric clashes preventing initial explosion.
Strongly recommended: True.
"""

MD_SEED = 42
"""
RNG seed for Langevin dynamics (velocities initialization).
Ensures trajectory reproducibility. Ignored in MAIN_ANALYSIS mode.
Recommended: Any integer (e.g., 42).
"""

MD_TIME_STEP_FS = 2.0
"""
Initial integration timestep (femtoseconds).
Standard value with bond constraints. Auto-reduced on instability.
Recommended: 2.0. Range: 0.5-5.0.
"""

MD_ROLLBACK_STEPS = 5000
"""
Chunk size for rollback on NaN instability.
Coordinates/velocities saved every chunk. On NaN, state restored from RAM.
Recommended: 1000 (or multiple of report interval). Range: 100-10000.
"""

MD_MAX_ROLLBACKS = 5
"""
Max consecutive rollbacks allowed without timestep reduction.
Exceeding this triggers timestep reduction and full restart.
0 = No rollbacks; immediate timestep reduction on first NaN.
Recommended: 5. Range: 0-100.
"""

MD_TIME_STEP_REDUCTION_FACTOR = 2.0
"""
Timestep reduction factor when rollback limit exceeded.
Current dt is divided by this value; simulation restarts from scratch.
Recommended: 2.0. Range: 1.1-10.0.
"""

MD_MIN_TIME_STEP_FS = 0.25
"""
Minimum allowed integration timestep (fs).
Below this, simulation aborts as physically unstable.
Recommended: 0.25.
"""

# ----------------------------------------------------------------------------
# GROUP 3: MSM AND NEIGHBOR GRAPH PARAMETERS
# ----------------------------------------------------------------------------

KRAMERS_TEMPERATURE = 300.0
"""
Temperature in Kramers rate formula (Kelvin).
Used ONLY for transition rate matrix construction.
Can differ from MD temperature (TEMPERATURE_K).
Recommended: 300.0 K (physiological). Range: 100-500K.
"""

NEIGHBOR_METHOD = 'adaptive_topology'
"""
Method for defining microstate neighborhood.
'energy'            - Energy-based: |F_i - F_j| <= -RT ln(eta).
                      Requires ETA_NEIGHBOR_MODE and ETA_NEIGHBOR.
'adaptive_topology' - Adaptive topological graph: d(c_i, c_j) <= alpha*(R_i+R_j).
                      Objective, no manual distance threshold needed.
Recommended: 'adaptive_topology'.
"""

ETA_NEIGHBOR_MODE = 'manual'
"""
Mode for selecting eta parameter (energy-based neighborhood only).
'manual' - Use user-defined ETA_NEIGHBOR.
'auto'   - Automatic selection based on graph topology analysis.
           Selects strictest eta yielding valid connectivity (f_inter <= 0.9).
Recommended: 'auto' (objective), 'manual' (comparison).
"""

ETA_NEIGHBOR = 1e-6
"""
Dimensionless transition rate threshold for energy-based neighborhood.
Used ONLY when NEIGHBOR_METHOD='energy' AND ETA_NEIGHBOR_MODE='manual'.
dF_max = -RT ln(ETA_NEIGHBOR). Example: 1e-6 -> ~34.5 kJ/mol at 300K.
Recommended: 1e-6. Range: 1e-12 to 1e-3.
"""

NEIGHBOR_RADIUS_PERCENTILE = 95
"""
Percentile for effective cluster radius calculation (%).
Used ONLY when NEIGHBOR_METHOD='adaptive_topology'.
95% means 95% of cluster points lie within this radius from center.
Recommended: 95. Range: 50-99.
"""

NEIGHBOR_ALPHA = 0.6
"""
Scaling coefficient for adaptive topological neighborhood.
Condition: d(c_i, c_j) <= alpha * (R_i + R_j)
alpha=1.0: Sphere touching (Delaunay-like). Standard default.
alpha>1.0: Denser graph, allows gaps. Useful for sparse sampling.
alpha<1.0: Sparser graph, requires overlap. Removes spurious long-range edges.
Recommended: 1.0 (default), 0.6 (for stricter steric barriers). Range: 0.5-2.0.
"""

GEOMETRY_RMSD_MODE = 'all'
"""
Unified RMSD mode for ALL geometric operations (clustering + graph).
'ca'  - C-alpha atoms only. Faster, good for backbone.
'all' - All atoms. Stricter, accounts for side chains, slower.
Recommended: 'all'. Use 'ca' for long trajectories (>20k frames).
"""

N_CLUSTERS = 5000
"""
Target number of microstates (clusters) for discretization.
Actual count may be lower after filtering small clusters.
Recommended: 1000 (~10k frames), 4000 (~40k frames). Range: 100-10000.
"""

MIN_CLUSTER_SIZE = 10
"""
Minimum frames per cluster to be considered significant.
Smaller clusters treated as noise and excluded.
Recommended: 5-10. Range: 2-100.
"""

BASIN_EQ_THRESHOLD = 3.0
"""
Energy equivalence threshold (epsilon_eq) for plateaus/minima (kJ/mol).
Neighbors with |F_i - F_j| <= epsilon_eq considered equivalent (plateau).
~3.0 kJ/mol corresponds to ~1 kBT at 300K.
Recommended: 3.0. Range: 0.1-20.0.
"""

MIN_BASIN_SIZE = 10
"""
Minimum microstates per basin of attraction.
Smaller basins discarded as statistically insignificant.
Recommended: 5. Range: 1-20.
"""

MIN_BASINS_PER_COMPONENT = 5
"""
Minimum basins required in a connected component for analysis.
Components with fewer basins are discarded. Need >=3 for triplets,
but >=10 recommended for statistical reliability.
Recommended: 10. Range: 3-100.
"""

# ----------------------------------------------------------------------------
# GROUP 4: SPECTRAL ANALYSIS AND ULTRAMETRICITY PARAMETERS
# ----------------------------------------------------------------------------

NUM_EIGENMODES = 50
"""
Max eigenmodes for K_sym spectral decomposition.
After noise filtering, remaining n_phys modes define kinetic metric.
Recommended: 50. Range: 10-500.
"""

EIGS_SIGMA = 1e-4
"""
Shift-invert sigma for Lanczos algorithm (finding eigenvalues near zero).
Smallest nonzero eigenvalues correspond to slowest relaxation processes.
Recommended: 1e-4. Range: 1e-6 to 1e-3.
"""

EIGS_TOL = 1e-8
"""
Convergence tolerance for Lanczos (eigsh).
Recommended: 1e-8. Range: 1e-10 to 1e-6.
"""

EIGS_DENSE_THRESHOLD = 6000
"""
Matrix size threshold for switching to dense diagonalization.
Components smaller than this use np.linalg.eigh instead of sparse eigsh.
Recommended: 6000. Range: 100-10000.
"""

SPECTRAL_GAP_THRESHOLD = 1e6
"""
Threshold for |lambda_k|/|lambda_{k-1}| ratio to detect spectral gap.
Modes below gap treated as numerical noise.
Recommended: 1e6. Range: 1e4-1e8.
"""

ULTRAMETRIC_EPSILON = 0.05
"""
Relative tolerance epsilon for triplet classification.
Non-trivial ultrametricity requires (d_max - d_mid)/d_mid <= epsilon.
Must be strictly less than ULTRAMETRIC_DELTA.
Recommended: 0.05 (5%). Range: 0.01-0.20.
"""

ULTRAMETRIC_DELTA = 0.1
"""
Minimum relative separation for non-trivial ultrametricity.
Requires (d_mid - d_min)/d_mid > delta.
Must be strictly greater than ULTRAMETRIC_EPSILON.
Recommended: 0.1 (10%). Range: 0.05-0.50.
"""

# ============================================================================
# BLAS/LAPACK/OPENMP THREAD LIMITING
# ============================================================================
if MAX_CPU_THREADS and MAX_CPU_THREADS > 0:
    num_cores = min(MAX_CPU_THREADS, multiprocessing.cpu_count())
else:
    num_cores = multiprocessing.cpu_count()

os.environ['OMP_NUM_THREADS'] = str(num_cores)
os.environ['MKL_NUM_THREADS'] = str(num_cores)
os.environ['OPENBLAS_NUM_THREADS'] = str(num_cores)

try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=num_cores, user_api='blas')
except ImportError:
    pass

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================
R_KCAL = 0.001987204259
"""Universal gas constant in kcal/(mol*K)."""

KB_KJ = 0.008314462618
"""Universal gas constant in kJ/(mol*K)."""


# ============================================================================
# ROBUST STRUCTURE PREPARATION
# ============================================================================
def prepare_structure(input_file, use_gb99dms, verbose=True):
    from pdbfixer import PDBFixer
    from openmm.app import Modeller, ForceField
    from collections import defaultdict

    STANDARD_AMINO_ACIDS = [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
    ]
    EXTRA_AMINO_ACIDS = [
        'SEC', 'PYL', 'ASX', 'GLX', 'UNK', 'ACE', 'NME',
        'MSE', 'SEP', 'TPO', 'PTR', 'HYP',
        'HID', 'HIE', 'HIP', 'CYX', 'ASH', 'GLH', 'LYN', 'CYM'
    ]
    ALL_AMINO_ACIDS = STANDARD_AMINO_ACIDS + EXTRA_AMINO_ACIDS

    if verbose:
        print("  [PDBFixer] Loading and fixing structure...")

    fixer = PDBFixer(filename=input_file)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)

    modeller = Modeller(fixer.topology, fixer.positions)

    # Remove non-protein residues
    non_protein_residues = []
    for residue in modeller.topology.residues():
        if residue.name not in ALL_AMINO_ACIDS:
            non_protein_residues.append(residue)

    if non_protein_residues:
        if verbose:
            print(f"  [Modeller] Removing {len(non_protein_residues)} non-protein residues...")
        modeller.delete(non_protein_residues)

    # Select longest chain
    chains = defaultdict(list)
    for residue in modeller.topology.residues():
        chain_id = residue.chain.id if hasattr(residue, 'chain') else ' '
        if residue.name in STANDARD_AMINO_ACIDS:
            chains[chain_id].append(residue)

    if chains:
        longest_chain_id = max(chains.keys(), key=lambda c: len(chains[c]))
        longest_chain_residues = set(chains[longest_chain_id])
        residues_to_delete = [
            r for r in modeller.topology.residues()
            if r not in longest_chain_residues
        ]
        if residues_to_delete:
            if verbose:
                print(f"  [Modeller] Selected chain '{longest_chain_id}'. "
                      f"Removing {len(residues_to_delete)} residues from other chains...")
            modeller.delete(residues_to_delete)

    # Add hydrogens according to force field
    if verbose:
        print(f"  [Modeller] Adding hydrogens consistent with force field...")

    if use_gb99dms:
        gb_path = "GB99dms.xml"
        if not os.path.exists(gb_path):
            if verbose:
                print("  Downloading GB99dms.xml...")
            import urllib.request
            url = "https://raw.githubusercontent.com/greener-group/GB99dms/main/GB99dms.xml"
            urllib.request.urlretrieve(url, gb_path)
        ff = ForceField(gb_path)
    else:
        ff = ForceField("amber14-all.xml", "implicit/obc2.xml")

    modeller.addHydrogens(ff, pH=7.0)

    topology = modeller.topology
    positions = modeller.positions

    if verbose:
        n_atoms = topology.getNumAtoms()
        n_res = sum(1 for _ in topology.residues())
        print(f"  Final structure: {n_atoms} atoms, {n_res} residues")

    return topology, positions, ff


# ============================================================================
# LOAD TRAJECTORY AND ENERGIES FROM FILES (FOR FROM_FILES_ANALYSIS MODE)
# ============================================================================
def load_trajectory_and_energies_from_files(pdb_id, starting_structure_file,
                                            use_gb99dms, use_gpu,
                                            kramers_temperature, verbose=True):
    from openmm import app
    import mdtraj as md

    if verbose:
        print("\n" + "=" * 70)
        print("LOADING DATA FROM FILES (FROM_FILES_ANALYSIS mode)")
        print("=" * 70)

    # Determine structure file
    if starting_structure_file and os.path.exists(starting_structure_file):
        base_name = os.path.splitext(os.path.basename(starting_structure_file))[0]
        structure_file = starting_structure_file
        if verbose:
            print(f"  Using custom structure: {starting_structure_file}")
            print(f"  Base name: {base_name}")
    else:
        base_name = pdb_id
        local_cif = f"{pdb_id}.cif"
        local_pdb = f"{pdb_id}.pdb"
        if os.path.exists(local_cif):
            structure_file = local_cif
        elif os.path.exists(local_pdb):
            structure_file = local_pdb
        else:
            if verbose:
                print(f"\n[ERROR] Structure files not found:")
                print(f"  - {local_cif}")
                print(f"  - {local_pdb}")
                print(f"  - {starting_structure_file if starting_structure_file else '(not specified)'}")
                print(f"  FROM_FILES_ANALYSIS requires a structure file.")
            return None, None, None

    if verbose:
        print(f"  Structure file: {structure_file}")
        print(f"  Base name: {base_name}")

    # Check DCD file
    dcd_file = f"{base_name}_trajectory.dcd"
    if not os.path.exists(dcd_file):
        if verbose:
            print(f"\n[ERROR] Trajectory file not found: {dcd_file}")
            print(f"  FROM_FILES_ANALYSIS requires a DCD file.")
        return None, None, None

    if verbose:
        print(f"  Trajectory found: {dcd_file}")

    pdb_file = f"{base_name}_topology.pdb"
    topology_file_for_mdtraj = pdb_file if os.path.exists(pdb_file) else structure_file
    if verbose:
        print(f"  Topology file for mdtraj: {topology_file_for_mdtraj}")

    # Load or compute energies
    energy_file = f"{base_name}_energies.txt"
    energy_file_exists = os.path.exists(energy_file)

    try:
        traj = md.load(dcd_file, top=topology_file_for_mdtraj)
        n_frames = traj.n_frames
        if verbose:
            print(f"  Trajectory loaded: {n_frames} frames, {traj.n_atoms} atoms")
    except Exception as e:
        if verbose:
            print(f"\n[ERROR] Failed to load trajectory: {type(e).__name__}")
            print(f"  Message: {str(e)[:200]}")
        return None, None, None

    if energy_file_exists:
        try:
            energies = np.loadtxt(energy_file)
            if len(energies) != n_frames:
                if verbose:
                    print(f"\n[WARNING] Mismatch: trajectory has {n_frames} frames, "
                          f"energies file has {len(energies)} values.")
                    print(f"  Energies will be recalculated.")
                energy_file_exists = False
            else:
                if verbose:
                    print(f"  Energies loaded: {energy_file}")
                    print(f"  Energies: min={energies.min():.1f}, max={energies.max():.1f}, "
                          f"mean={energies.mean():.1f} kJ/mol")
        except Exception as e:
            if verbose:
                print(f"\n[WARNING] Failed to read energies file: {type(e).__name__}")
                print(f"  Energies will be recalculated.")
            energy_file_exists = False

    if not energy_file_exists:
        if verbose:
            print(f"\nRecalculating energies via OpenMM (may take ~30-60 sec)...")
            print(f"  Force field: {'GB99dms' if use_gb99dms else 'Amber14/OBC2'}")

        try:
            topology_omm, _, ff = prepare_structure(structure_file, use_gb99dms, verbose)
        except ImportError:
            raise ImportError("pdbfixer is required: pip install pdbfixer")

        if topology_omm.getNumAtoms() != traj.n_atoms:
            if verbose:
                print(f"\n[ERROR] Atom count mismatch: cleaned structure ({topology_omm.getNumAtoms()}) "
                      f"vs trajectory ({traj.n_atoms}).")
                print(f"  Ensure DCD was created by this program or provide matching *_topology.pdb.")
            return None, None, None

        if use_gb99dms:
            system = ff.createSystem(
                topology_omm,
                nonbondedMethod=app.CutoffNonPeriodic,
                nonbondedCutoff=1.0 * unit.nanometer,
                constraints=app.HBonds,
                implicitSolventKappa=0.7 / unit.nanometer
            )
        else:
            system = ff.createSystem(
                topology_omm,
                nonbondedMethod=app.CutoffNonPeriodic,
                nonbondedCutoff=1.0 * unit.nanometer,
                constraints=app.HBonds
            )

        platform = openmm.Platform.getPlatformByName("CPU")
        props = {"Threads": "1"}
        integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
        context = openmm.Context(system, integrator, platform, props)

        energies_list = []
        t0 = time.time()
        for frame_idx in range(n_frames):
            coords_nm = traj.xyz[frame_idx]
            positions = [openmm.Vec3(x, y, z) * unit.nanometer
                         for x, y, z in coords_nm]
            context.setPositions(positions)
            state = context.getState(getEnergy=True)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            energies_list.append(energy)

            if verbose and (frame_idx + 1) % max(1, n_frames // 10) == 0:
                elapsed = time.time() - t0
                print(f"    Processed {frame_idx+1}/{n_frames} frames ({elapsed:.1f} sec)")

        del context, integrator
        gc.collect()

        energies = np.array(energies_list)
        np.savetxt(energy_file, energies, fmt='%.6f')

        if verbose:
            elapsed = time.time() - t0
            print(f"  Energies recalculated in {elapsed:.1f} sec")
            print(f"  Saved: {energy_file}")
            print(f"  Energies: min={energies.min():.1f}, max={energies.max():.1f}, "
                  f"mean={energies.mean():.1f} kJ/mol")

    del traj
    gc.collect()

    return dcd_file, topology_file_for_mdtraj, energy_file


# ============================================================================
# AUTOMATIC ETA PARAMETER SELECTION
# ============================================================================
def auto_select_eta_neighbor(free_energies, RT, min_basin_size,
                             min_basins_per_comp, eq_threshold,
                             verbose=True):
    L = 30
    log_eta_min = -12.0
    log_eta_max = -2.0
    delta_log = (log_eta_max - log_eta_min) / L
    eta_grid = np.array([10.0 ** (log_eta_min + l * delta_log) for l in range(L + 1)])
    eta_grid = np.sort(eta_grid)[::-1]

    if verbose:
        print("\nAutomatic eta selection:")
        print(f"    Grid: {len(eta_grid)} points in [{eta_grid.min():.2e}, {eta_grid.max():.2e}]")

    candidates = []
    for eta_val in eta_grid:
        adjacency = build_neighbor_graph_energy(
            free_energies, RT, eta_val, verbose=False
        )
        components = find_connected_components(adjacency)

        sig_components = []
        total_basins_all = 0
        sum_triplets_all = 0

        for comp_micro_indices in components:
            adj_c = adjacency[np.ix_(comp_micro_indices, comp_micro_indices)]
            fe_c = free_energies[comp_micro_indices]
            fe_c = fe_c - fe_c.min()

            basin_labels_local, _, _ = identify_basins_gradient_descent(
                fe_c, adj_c, eq_threshold
            )
            unique_basins, basin_counts = np.unique(basin_labels_local, return_counts=True)
            valid_basins = unique_basins[basin_counts >= min_basin_size]
            n_basins_c = len(valid_basins)

            if n_basins_c >= min_basins_per_comp:
                sig_components.append(n_basins_c)

            total_basins_all += n_basins_c
            triplets_c = n_basins_c * (n_basins_c - 1) * (n_basins_c - 2) // 6
            sum_triplets_all += triplets_c

        C_sig = len(sig_components)
        K_max = max(sig_components) if sig_components else 0

        if total_basins_all >= 3:
            total_triplets_possible = (total_basins_all * (total_basins_all - 1) *
                                       (total_basins_all - 2) // 6)
            f_inter = (1.0 - sum_triplets_all / total_triplets_possible
                       if total_triplets_possible > 0 else 1.0)
        else:
            f_inter = 1.0

        if C_sig >= 1 and f_inter <= 0.9:
            candidates.append((eta_val, C_sig, K_max, f_inter))

    if len(candidates) == 0:
        fallback_eta = eta_grid[-1]
        df_max_fallback = -RT * np.log(fallback_eta)
        if verbose:
            print(f"    [WARNING] No eta satisfies criteria.")
            print(f"    Using smallest eta = {fallback_eta:.2e}")
            print(f"    (dF_max = {df_max_fallback:.1f} kJ/mol)")
        return fallback_eta

    candidates.sort(key=lambda x: (-x[0], -x[2]))
    best = candidates[0]
    best_eta = best[0]
    df_max_best = -RT * np.log(best_eta)

    if verbose:
        print(f"    Selected eta = {best_eta:.2e}")
        print(f"    dF_max = {df_max_best:.1f} kJ/mol ({df_max_best/RT:.1f} RT)")
        print(f"    C_sig = {best[1]}, K_max = {best[2]}, f_inter = {best[3]:.4f}")
        print(f"    Valid candidates: {len(candidates)} of {len(eta_grid)}")

    return best_eta


# ============================================================================
# STAGE 1: MOLECULAR DYNAMICS
# ============================================================================
def run_md_simulation(pdb_id, use_gb99dms, temp_k, n_steps, report_interval,
                      use_gpu, seed, starting_structure_file,
                      add_hydrogens_to_custom_start, minimize_starting_structure,
                      md_time_step_fs, md_rollback_steps, md_max_rollbacks,
                      md_time_step_reduction_factor, md_min_time_step_fs,
                      verbose=True):
    from openmm import app
    import mdtraj as md

    if verbose:
        print("\n" + "=" * 70)
        print("STAGE 1: MOLECULAR DYNAMICS (LANGEVIN DYNAMICS)")
        print("=" * 70)
        print(f"  Protein: {pdb_id}, T = {temp_k:.1f} K, steps: {n_steps}")
        print(f"  Save interval: {report_interval}, seed: {seed}")
        print(f"  Force field: {'GB99dms' if use_gb99dms else 'Amber14/OBC2'}")
        print(f"  GPU: {'Yes' if use_gpu else 'No'}")

    # Determine input structure
    if starting_structure_file and os.path.exists(starting_structure_file):
        input_file = starting_structure_file
        if verbose:
            print(f"\nLoading custom structure: {input_file}")
    else:
        local_cif = f"{pdb_id}.cif"
        local_pdb = f"{pdb_id}.pdb"
        input_file = None

        if os.path.exists(local_cif):
            input_file = local_cif
            if verbose:
                print(f"\nLocal CIF found: {local_cif}")
        elif os.path.exists(local_pdb):
            input_file = local_pdb
            if verbose:
                print(f"\nLocal PDB found: {local_pdb}")
        else:
            if verbose:
                print(f"\nNo local files found. Downloading {pdb_id} from RCSB PDB...")
            import urllib.request
            cif_url = f"https://files.rcsb.org/download/{pdb_id}.cif"
            download_path = os.path.abspath(local_cif)
            try:
                urllib.request.urlretrieve(cif_url, download_path)
                input_file = download_path
                if verbose:
                    print(f"  CIF downloaded: {download_path}")
            except Exception:
                if verbose:
                    print(f"  CIF download failed, trying PDB...")
                pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                download_path = os.path.abspath(local_pdb)
                urllib.request.urlretrieve(pdb_url, download_path)
                input_file = download_path
                if verbose:
                    print(f"  PDB downloaded: {download_path}")

    # Prepare structure
    try:
        topology, positions, ff = prepare_structure(input_file, use_gb99dms, verbose)
    except ImportError:
        raise ImportError("pdbfixer is required: pip install pdbfixer")

    # Create system
    if use_gb99dms:
        system = ff.createSystem(
            topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=app.HBonds,
            implicitSolventKappa=0.7 / unit.nanometer
        )
    else:
        system = ff.createSystem(
            topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=app.HBonds
        )

    # Platform selection
    platform = None
    props = {}
    if use_gpu:
        try:
            platform = openmm.Platform.getPlatformByName("CUDA")
            props = {
                "Precision": "mixed",
                "DeviceIndex": "0",
                "CudaPrecision": "mixed",
                "DeterministicForces": "false"
            }
            test_integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
            test_sim = app.Simulation(topology, system, test_integrator, platform, props)
            test_sim.context.setPositions(positions)
            test_sim.context.getState(getEnergy=True)
            del test_sim, test_integrator
            gc.collect()
            if verbose:
                print("  Platform: CUDA (mixed precision)")
        except Exception as e:
            if verbose:
                print(f"  [WARNING] CUDA init error (missing nvrtc DLL or driver): {type(e).__name__}")
                print(f"             Automatic fallback to CPU.")
            platform = openmm.Platform.getPlatformByName("CPU")
            props = {"Threads": "1"}
            if verbose:
                print("  Platform: CPU (fallback)")
    else:
        platform = openmm.Platform.getPlatformByName("CPU")
        props = {"Threads": "1"}
        if verbose:
            print("  Platform: CPU (GPU disabled by user)")

    current_dt_fs = md_time_step_fs
    final_dcd_file = f"{pdb_id}_trajectory.dcd"
    energy_file = f"{pdb_id}_energies.txt"
    pdb_file = f"{pdb_id}_topology.pdb"

    # Save cleaned topology
    with open(pdb_file, 'w') as f:
        app.PDBFile.writeFile(topology, positions, f)
    if verbose:
        print(f"  Cleaned topology saved: {pdb_file}")

    # Remove old output files
    for f in [final_dcd_file, energy_file]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    # Memory reporter for safe rollback
    class MemoryReporter:
        def __init__(self, interval):
            self.interval = interval
            self.positions = []
            self.energies = []

        def describeNextReport(self, simulation):
            steps_left = self.interval - simulation.currentStep % self.interval
            return (steps_left, True, False, False, True, None)

        def report(self, simulation, state):
            self.positions.append(state.getPositions(asNumpy=True))
            self.energies.append(
                state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            )

    # Main MD loop with rollback support
    while True:
        if current_dt_fs < md_min_time_step_fs:
            if verbose:
                print(f"\n[ERROR] Integration step fell below minimum ({md_min_time_step_fs} fs).")
                print("           System is physically unstable.")
            return None, None, None

        dt = current_dt_fs * 0.001

        if verbose and current_dt_fs < md_time_step_fs:
            print(f"\n[RESTART] Integration step reduced to {current_dt_fs} fs...")

        try:
            integrator = openmm.LangevinIntegrator(
                temp_k * unit.kelvin,
                1.0 / unit.picosecond,
                dt * unit.picosecond
            )
            integrator.setConstraintTolerance(1e-5)
            integrator.setRandomNumberSeed(seed)

            sim = app.Simulation(topology, system, integrator, platform, props)
            sim.context.setPositions(positions)

            if minimize_starting_structure:
                if verbose:
                    print("  Minimizing starting structure energy...")
                sim.minimizeEnergy(maxIterations=2000)

            sim.context.setVelocitiesToTemperature(temp_k * unit.kelvin, seed)

            reporter = MemoryReporter(report_interval)
            sim.reporters.append(reporter)

            total_time_ps = n_steps * dt
            n_frames = n_steps // report_interval
            print(f"  Starting MD: {n_steps} steps ({total_time_ps:.1f} ps, dt={current_dt_fs} fs), "
                  f"~{n_frames} frames to save")

            t0 = time.time()
            state = sim.context.getState(getPositions=True, getVelocities=True)
            resume_pos = state.getPositions(asNumpy=True)
            resume_vel = state.getVelocities(asNumpy=True)

            frames_saved = 0
            total_steps_done = 0
            consecutive_rollbacks = 0
            dof = 3 * system.getNumParticles() - system.getNumConstraints()

            if verbose:
                print('#"Step","Potential Energy (kJ/mole)","Temperature (K)"')

            sim_failed = False

            while total_steps_done < n_steps:
                chunk_size = min(md_rollback_steps, n_steps - total_steps_done)

                try:
                    sim.step(chunk_size)
                    total_steps_done += chunk_size

                    if verbose and (total_steps_done % max(1, n_steps // 10) == 0
                                    or total_steps_done == n_steps):
                        state = sim.context.getState(getEnergy=True)
                        pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                        ke = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
                        temp = (2.0 * ke) / (dof * KB_KJ) if dof > 0 else 0.0
                        print(f"{total_steps_done},{pe},{temp}")

                    state = sim.context.getState(getPositions=True, getVelocities=True)
                    pos_np = state.getPositions(asNumpy=True)
                    vel_np = state.getVelocities(asNumpy=True)

                    if np.isnan(pos_np).any() or np.isnan(vel_np).any():
                        if verbose:
                            print(f"\n[ROLLBACK] Hidden NaNs detected at step {total_steps_done}. Initiating rollback.")
                        raise ValueError("Hidden NaNs detected in getState")

                    resume_pos = pos_np
                    resume_vel = vel_np
                    frames_saved = len(reporter.positions)
                    consecutive_rollbacks = 0

                except Exception as e:
                    is_nan_error = ("NaN" in str(e) or isinstance(e, openmm.OpenMMException)
                                    or isinstance(e, ValueError))

                    if is_nan_error:
                        consecutive_rollbacks += 1

                        if consecutive_rollbacks > md_max_rollbacks:
                            if verbose:
                                print(f"\n[WARNING] Consecutive rollback limit exceeded ({md_max_rollbacks}).")
                                print(f"             Reducing timestep by factor {md_time_step_reduction_factor} and restarting.")
                            current_dt_fs /= md_time_step_reduction_factor
                            sim_failed = True
                            break
                        else:
                            if verbose:
                                print(f"\n[ROLLBACK] NaN error at step {total_steps_done}. "
                                      f"Rolling back {md_rollback_steps} steps (attempt {consecutive_rollbacks}/{md_max_rollbacks}).")

                            del reporter.positions[frames_saved:]
                            del reporter.energies[frames_saved:]
                            sim.reporters = []
                            del sim, integrator
                            gc.collect()

                            integrator = openmm.LangevinIntegrator(
                                temp_k * unit.kelvin,
                                1.0 / unit.picosecond,
                                dt * unit.picosecond
                            )
                            integrator.setConstraintTolerance(1e-5)
                            integrator.setRandomNumberSeed(
                                seed + consecutive_rollbacks + int(time.time()) % 1000
                            )

                            sim = app.Simulation(topology, system, integrator, platform, props)
                            sim.context.setPositions(resume_pos)
                            sim.context.setVelocities(resume_vel)
                            sim.reporters.append(reporter)
                    else:
                        raise

            if sim_failed:
                sim.reporters = []
                del sim, integrator
                gc.collect()
                continue

            sim.reporters = []
            del sim, integrator
            gc.collect()

            elapsed = time.time() - t0
            if verbose:
                print(f"  MD completed in {elapsed:.1f} sec ({elapsed/60:.1f} min)")

            if len(reporter.positions) > 0:
                if verbose:
                    print(f"  Writing DCD file to disk ({len(reporter.positions)} frames)...")

                topology_mdtraj = md.load(pdb_file).topology
                traj = md.Trajectory(xyz=np.array(reporter.positions), topology=topology_mdtraj)
                traj.save_dcd(final_dcd_file)
                np.savetxt(energy_file, np.array(reporter.energies), fmt='%.6f')

                if verbose:
                    print(f"  Trajectory saved: {final_dcd_file}")
                    print(f"  Energies saved: {energy_file} ({len(reporter.energies)} values)")
            else:
                energy_file = None
                final_dcd_file = None
                if verbose:
                    print("  WARNING: Position/energy lists are empty!")

            return final_dcd_file, pdb_file, energy_file

        except Exception as e:
            if verbose:
                print(f"\n[WARNING] Critical error: {type(e).__name__}")
                print(f"             Message: {str(e)[:150]}")
            try:
                sim.reporters = []
                del sim, integrator
            except Exception:
                pass
            gc.collect()
            return None, None, None


# ============================================================================
# K-CENTERS CLUSTERING
# ============================================================================
class ClusterKCenters:
    def __init__(self, traj, n_clusters, atom_indices):
        self.n_clusters = n_clusters
        self.atom_indices = atom_indices
        self.labels, self.center_indices = self._fit(traj)

    def _fit(self, traj):
        import mdtraj as md
        n_frames = traj.n_frames
        n_clusters = min(self.n_clusters, n_frames)

        center_indices = np.zeros(n_clusters, dtype=int)
        labels = np.zeros(n_frames, dtype=int)
        min_distances = np.full(n_frames, np.inf)

        for i in range(n_clusters):
            if i == 0:
                c_idx = 0
            else:
                c_idx = np.argmax(min_distances)

            center_indices[i] = c_idx

            rmsds_to_c = md.rmsd(
                traj, traj, c_idx,
                atom_indices=self.atom_indices,
                precentered=False
            )

            closer_mask = rmsds_to_c < min_distances
            labels[closer_mask] = i
            min_distances[closer_mask] = rmsds_to_c[closer_mask]

        return labels, center_indices

    def assign(self, traj):
        return self.labels


# ============================================================================
# STAGE 2: MSM AND NEIGHBOR GRAPH CONSTRUCTION
# ============================================================================
def build_neighbor_graph_energy(free_energies, RT, eta, verbose=True):
    n_states = len(free_energies)
    delta_f_max = -RT * np.log(eta)

    diff = np.abs(free_energies[:, None] - free_energies[None, :])
    adjacency = diff <= delta_f_max
    np.fill_diagonal(adjacency, False)

    n_edges = np.sum(adjacency) // 2

    if verbose:
        print(f"  Energy-based neighborhood: dF_max = {delta_f_max:.1f} kJ/mol "
              f"(eta={eta:.0e}, T={RT/KB_KJ:.0f} K)")
        print(f"  Edges in graph: {n_edges} of {n_states*(n_states-1)//2} possible")

    return adjacency


def build_adaptive_topological_graph(traj, n_states, radius_percentile,
                                     rmsd_mode, neighbor_alpha,
                                     original_center_indices,
                                     valid_clusters, filtered_labels,
                                     verbose=True):
    import mdtraj as md

    center_frames_indices = [
        original_center_indices[old_label] for old_label in valid_clusters
    ]
    centers_traj = traj.slice(center_frames_indices)

    if rmsd_mode == 'ca':
        atom_indices = [a.index for a in traj.topology.atoms if a.name == 'CA']
    else:
        atom_indices = None

    if verbose:
        n_atoms = len(atom_indices) if atom_indices else traj.n_atoms
        print(f"  Adaptive topological graph: computing effective radii")
        print(f"    percentile: {radius_percentile}%, atoms: {n_atoms}, alpha: {neighbor_alpha}")

    radii = np.zeros(n_states)
    for i in range(n_states):
        cluster_frame_indices = np.where(filtered_labels == i)[0]
        if len(cluster_frame_indices) > 1:
            cluster_traj = traj.slice(cluster_frame_indices)
            rmsds_to_center = md.rmsd(
                cluster_traj, centers_traj, i,
                atom_indices=atom_indices, precentered=False
            )
            radii[i] = np.percentile(rmsds_to_center, radius_percentile)
        else:
            radii[i] = 0.0

    if verbose:
        print(f"  Effective radii: min={radii.min():.4f} nm, "
              f"max={radii.max():.4f} nm, mean={radii.mean():.4f} nm")
        print(f"  Building graph: d(c_i, c_j) <= {neighbor_alpha} * (R_i + R_j)")

    adjacency = np.zeros((n_states, n_states), dtype=bool)
    for i in range(n_states):
        rmsds_to_i = md.rmsd(
            centers_traj, centers_traj, i,
            atom_indices=atom_indices, precentered=False
        )
        thresholds = neighbor_alpha * (radii[i] + radii)
        neighbors_mask = rmsds_to_i <= thresholds
        neighbors_mask[i] = False
        adjacency[i, neighbors_mask] = True
        adjacency[neighbors_mask, i] = True

    n_edges = np.sum(adjacency) // 2
    if verbose:
        print(f"  Edges in graph: {n_edges} of {n_states*(n_states-1)//2} possible")

    return adjacency


def build_msm_and_graph(dcd_file, pdb_file, energy_file, n_clusters,
                        min_cluster_size, temp_k, neighbor_method, eta,
                        radius_percentile, rmsd_mode, neighbor_alpha,
                        verbose=True):
    import mdtraj as md

    if verbose:
        print("\n" + "=" * 70)
        print("STAGE 2: MSM AND NEIGHBOR GRAPH CONSTRUCTION")
        print("=" * 70)
        print(f"  Loading trajectory: {dcd_file}")
        print(f"  Neighborhood method: {neighbor_method}")
        print(f"  RMSD metric: {'C-alpha' if rmsd_mode == 'ca' else 'all atoms'}")

        if neighbor_method == 'adaptive_topology':
            print(f"  Neighbor alpha: {neighbor_alpha}")
        elif neighbor_method == 'energy':
            print(f"  Eta (rate threshold): {eta:.0e}")

        print(f"  Kramers temperature: {temp_k} K")

    traj = md.load(dcd_file, top=pdb_file)
    n_frames = traj.n_frames

    if energy_file is None or not os.path.exists(energy_file):
        raise ValueError("Energy file not found.")

    energies = np.loadtxt(energy_file)
    if len(energies) != n_frames:
        raise ValueError(
            f"Frame count mismatch: trajectory {n_frames}, energies {len(energies)}"
        )

    if verbose:
        print(f"  Frames: {n_frames}, atoms in topology: {traj.n_atoms}")
        print(f"  Energies: min={energies.min():.1f}, max={energies.max():.1f}, "
              f"mean={energies.mean():.1f} kJ/mol")

    if rmsd_mode == 'ca':
        cluster_atom_indices = [a.index for a in traj.topology.atoms if a.name == 'CA']
        atom_desc = f"C-alpha ({len(cluster_atom_indices)})"
    else:
        cluster_atom_indices = None
        atom_desc = f"all atoms ({traj.n_atoms})"

    if rmsd_mode == 'ca' and len(cluster_atom_indices) < 3:
        raise ValueError(f"Too few C-alpha atoms: {len(cluster_atom_indices)} (need >= 3)")

    if verbose:
        print(f"  K-centers clustering: k={n_clusters}, atoms: {atom_desc}")

    t0 = time.time()
    cluster_obj = ClusterKCenters(traj, n_clusters, cluster_atom_indices)
    labels = cluster_obj.assign(traj)
    original_center_indices = cluster_obj.center_indices
    elapsed = time.time() - t0

    if verbose:
        print(f"  Clustering completed in {elapsed:.1f} sec")

    unique, counts = np.unique(labels, return_counts=True)
    valid_clusters = unique[counts >= min_cluster_size]

    if len(valid_clusters) < 3:
        if verbose:
            print(f"  ERROR: Only {len(valid_clusters)} significant clusters (need >= 3).")
        return None, None, None, None

    label_map = {old: new for new, old in enumerate(valid_clusters)}
    filtered_labels = np.array(
        [label_map[l] if l in label_map else -1 for l in labels]
    )
    n_states = len(valid_clusters)

    if verbose:
        print(f"  Significant microstates: {n_states} "
              f"(discarded {len(unique) - n_states} small clusters)")

    # Compute free energies
    RT = KB_KJ * temp_k
    free_energies = np.zeros(n_states)
    for i in range(n_states):
        mask = (filtered_labels == i)
        if np.sum(mask) > 0:
            cluster_energies = energies[mask]
            e_min = cluster_energies.min()
            boltzmann_sum = np.sum(np.exp(-(cluster_energies - e_min) / RT))
            free_energies[i] = e_min - RT * np.log(max(boltzmann_sum, 1e-300))

    free_energies -= free_energies.min()

    if verbose:
        print(f"  Free energies (relative): min={free_energies.min():.1f}, "
              f"max={free_energies.max():.1f} kJ/mol")

    # Build neighbor graph
    if neighbor_method == 'energy':
        adjacency = build_neighbor_graph_energy(free_energies, RT, eta, verbose=verbose)
    elif neighbor_method == 'adaptive_topology':
        adjacency = build_adaptive_topological_graph(
            traj, n_states, radius_percentile, rmsd_mode, neighbor_alpha,
            original_center_indices, valid_clusters, filtered_labels,
            verbose=verbose
        )
    else:
        raise ValueError(
            f"Unknown neighborhood method: '{neighbor_method}'. "
            f"Allowed: 'energy', 'adaptive_topology'."
        )

    # Build symmetrized rate matrix
    diff = np.abs(free_energies[:, None] - free_energies[None, :])
    rates = np.exp(-diff / (2.0 * RT))
    K_sym = np.where(adjacency, rates, 0.0)
    np.fill_diagonal(K_sym, 0.0)
    diag_values = -np.sum(K_sym, axis=1)
    np.fill_diagonal(K_sym, diag_values)

    n_nonzero = np.count_nonzero(K_sym) - n_states

    graph = csr_matrix(adjacency)
    n_components, _ = connected_components(csgraph=graph, directed=False, return_labels=True)

    if verbose:
        print(f"  K_sym matrix: {n_states}x{n_states}, "
              f"non-zero off-diagonal elements: {n_nonzero}")
        print(f"  Connected components in neighbor graph: {n_components}")

    del traj
    gc.collect()

    return K_sym, adjacency, free_energies, n_states


# ============================================================================
# CONNECTED COMPONENT AND BASIN ANALYSIS
# ============================================================================
def find_connected_components(adjacency):
    graph = csr_matrix(adjacency)
    n_components, labels = connected_components(
        csgraph=graph, directed=False, return_labels=True
    )
    components = [np.where(labels == i)[0] for i in range(n_components)]
    components.sort(key=len, reverse=True)
    return components


def identify_basins_gradient_descent(free_energies, adjacency, eq_threshold):
    """
    Strict physical definition of basins of attraction.
    1. Identify attractors (local minima considering noise epsilon_eq).
    2. Form plateaus only among attractors.
    3. Slope states "flow down" to nearest deeper nodes.
    
    Returns: basin_labels, basin_count, stats_dict
    """
    n_states = len(free_energies)

    # STEP 1: Identify attractors
    # Attractor = state with NO neighbors strictly lower than F_i - eq_threshold
    diff = free_energies[:, None] - free_energies[None, :]
    strictly_lower_neighbors = adjacency & (diff > eq_threshold)
    n_lower = np.sum(strictly_lower_neighbors, axis=1)
    is_attractor = (n_lower == 0)
    attractor_indices = np.where(is_attractor)[0]
    n_attractors = len(attractor_indices)

    basin_labels = np.full(n_states, -1, dtype=int)
    basin_count = 0
    n_plateaus = 0

    # STEP 2: Merge degenerate minima/plateaus (ONLY among attractors)
    if n_attractors > 0:
        att_adj = adjacency[np.ix_(attractor_indices, attractor_indices)]
        att_energies = free_energies[attractor_indices]
        att_diff = np.abs(att_energies[:, None] - att_energies[None, :])
        eq_adj_att = att_adj & (att_diff <= eq_threshold)
        np.fill_diagonal(eq_adj_att, False)

        eq_graph_att = csr_matrix(eq_adj_att)
        n_plateaus, plateau_labels = connected_components(
            csgraph=eq_graph_att, directed=False, return_labels=True
        )

        basin_labels[attractor_indices] = plateau_labels
        basin_count = n_plateaus

    # STEP 3: Gradient descent for slope states
    sorted_indices = np.argsort(free_energies)
    for i in sorted_indices:
        if basin_labels[i] != -1:
            continue  # Already labeled (attractor or plateau member)

        neighbors = np.where(adjacency[i])[0]
        if len(neighbors) == 0:
            basin_labels[i] = basin_count
            basin_count += 1
            continue

        lower_neighbors = neighbors[free_energies[neighbors] < free_energies[i] - eq_threshold]
        if len(lower_neighbors) > 0:
            best_neighbor = lower_neighbors[np.argmin(free_energies[lower_neighbors])]
            basin_labels[i] = basin_labels[best_neighbor]
        else:
            eq_neighbors = neighbors[np.abs(free_energies[neighbors] - free_energies[i]) <= eq_threshold]
            assigned_eq_neighbors = eq_neighbors[basin_labels[eq_neighbors] != -1]
            if len(assigned_eq_neighbors) > 0:
                best_neighbor = assigned_eq_neighbors[np.argmin(free_energies[assigned_eq_neighbors])]
                basin_labels[i] = basin_labels[best_neighbor]
            else:
                basin_labels[i] = basin_count
                basin_count += 1

    stats = {
        'n_attractors': n_attractors,
        'n_plateaus': n_plateaus,
        'n_basins_before_filter': basin_count
    }

    return basin_labels, basin_count, stats


# ============================================================================
# STAGE 3: KINETIC METRIC AND ULTRAMETRICITY TESTING
# ============================================================================
def compute_kinetic_metric_and_ultrametricity(K_sym_c, basin_labels_c,
                                              n_basins_c, num_modes,
                                              eigs_sigma, eigs_tol,
                                              dense_threshold,
                                              gap_threshold, eps, delta,
                                              quiet=False):
    n_micro = K_sym_c.shape[0]
    k_request = min(num_modes + 1, n_micro - 1)

    eigenvalues = None
    eigenvectors = None

    if n_micro >= dense_threshold:
        try:
            eigenvalues, eigenvectors = eigsh(
                K_sym_c, k=k_request, which='LM',
                sigma=eigs_sigma, tol=eigs_tol
            )
        except ArpackNoConvergence:
            if not quiet:
                print(f"    [WARNING] ARPACK did not converge. Switching to dense diagonalization.")
            eigenvalues = None
            eigenvectors = None
        except Exception:
            eigenvalues = None
            eigenvectors = None

    if eigenvalues is None or eigenvectors is None:
        evals_full, evecs_full = np.linalg.eigh(K_sym_c)
        idx = np.argsort(np.abs(evals_full))[:k_request]
        eigenvalues = evals_full[idx]
        eigenvectors = evecs_full[:, idx]

    sort_idx = np.argsort(np.abs(eigenvalues))
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Spectral gap filtering
    abs_vals = np.abs(eigenvalues)
    num_noise = 0
    for k in range(1, len(abs_vals)):
        if abs_vals[k - 1] < 1e-30:
            ratio = float('inf') if abs_vals[k] > 1e-30 else 1.0
        else:
            ratio = abs_vals[k] / abs_vals[k - 1]

        if ratio > gap_threshold:
            num_noise = k
            break

    phys_eigenvalues = eigenvalues[num_noise:]
    phys_eigenvectors = eigenvectors[:, num_noise:]
    n_phys = len(phys_eigenvalues)

    if n_phys == 0:
        return None

    if not quiet:
        print(f"    Physical modes: {n_phys} (noise modes discarded: {num_noise})")

    # Characteristic vectors and projections
    chi = np.zeros((n_basins_c, n_micro), dtype=np.float64)
    for a in range(n_basins_c):
        mask = (basin_labels_c == a)
        size = np.sum(mask)
        if size > 0:
            chi[a, mask] = 1.0 / np.sqrt(size)

    proj = chi @ phys_eigenvectors
    weights = 1.0 / np.abs(phys_eigenvalues)
    proj_scaled = proj * np.sqrt(weights)

    D = cdist(proj_scaled, proj_scaled, metric='euclidean')

    # Triplet classification
    triplets = list(combinations(range(n_basins_c), 3))
    triv_count = 0
    nontriv_count = 0
    non_count = 0

    for i, j, k in triplets:
        d = sorted([D[i, j], D[i, k], D[j, k]])
        d_min, d_mid, d_max = d[0], d[1], d[2]

        if d_max <= 1e-12:
            triv_count += 1
            continue

        if d_mid < 1e-12:
            non_count += 1
            continue

        if d_min < 1e-12:
            cond_triv = False
        else:
            cond_triv = (d_max - d_min) / d_min <= eps

        cond_nontriv = ((d_max - d_mid) / d_mid <= eps and
                        (d_mid - d_min) / d_mid > delta)

        if cond_nontriv:
            nontriv_count += 1
        elif cond_triv:
            triv_count += 1
        else:
            non_count += 1

    total = triv_count + nontriv_count + non_count
    u_triv = triv_count / total * 100 if total > 0 else 0.0
    u_nontriv = nontriv_count / total * 100 if total > 0 else 0.0
    u_non = non_count / total * 100 if total > 0 else 0.0

    return {
        'distance_matrix': D,
        'n_basins': n_basins_c,
        'n_physical_modes': n_phys,
        'u_trivial': u_triv,
        'u_nontrivial': u_nontriv,
        'u_nonultrametric': u_non,
        'n_triplets': total,
        'eigenvalues': phys_eigenvalues
    }


def analyze_landscape_components(K_sym, adjacency, free_energies, n_micro,
                                 temp_k, eq_threshold, min_basin_size,
                                 min_basins_per_comp, num_modes,
                                 eigs_sigma, eigs_tol, dense_threshold,
                                 gap_threshold, eps, delta, verbose=True):
    components = find_connected_components(adjacency)
    n_components_total = len(components)

    valid_components_data = []
    all_components_stats = []
    total_basins_all = 0
    sum_triplets_all = 0
    max_basins_in_any_component = 0
    total_valid_basins_all_components = 0

    if verbose:
        print(f"\nConnected components: {n_components_total}")

    for comp_idx, comp_micro_indices in enumerate(components):
        n_micro_c = len(comp_micro_indices)
        adj_c = adjacency[np.ix_(comp_micro_indices, comp_micro_indices)]
        n_edges_c = np.sum(adj_c) // 2
        fe_c = free_energies[comp_micro_indices]
        fe_c = fe_c - fe_c.min()

        basin_labels_local, n_basins_raw, stats = identify_basins_gradient_descent(
            fe_c, adj_c, eq_threshold
        )

        unique_basins, basin_counts = np.unique(basin_labels_local, return_counts=True)
        valid_basins = unique_basins[basin_counts >= min_basin_size]
        n_basins_c = len(valid_basins)

        status_str = "ACCEPTED" if n_basins_c >= min_basins_per_comp else "DISCARDED"

        all_components_stats.append({
            'comp_idx': comp_idx,
            'n_micro_c': n_micro_c,
            'n_edges_c': n_edges_c,
            'n_attractors': stats['n_attractors'],
            'n_plateaus': stats['n_plateaus'],
            'n_basins_before_filter': n_basins_raw,
            'n_basins_after_filter': n_basins_c,
            'status': status_str
        })

        if n_basins_c > max_basins_in_any_component:
            max_basins_in_any_component = n_basins_c

        total_valid_basins_all_components += n_basins_c

        if n_basins_c < min_basins_per_comp:
            continue

        basin_map = {old: new for new, old in enumerate(valid_basins)}
        final_basin_labels = np.array(
            [basin_map[b] if b in basin_map else -1 for b in basin_labels_local]
        )

        K_sym_c = K_sym[np.ix_(comp_micro_indices, comp_micro_indices)]
        triplets_c = n_basins_c * (n_basins_c - 1) * (n_basins_c - 2) // 6

        total_basins_all += n_basins_c
        sum_triplets_all += triplets_c

        valid_components_data.append({
            'K_sym_c': K_sym_c,
            'basin_labels_c': final_basin_labels,
            'n_basins_c': n_basins_c,
            'triplets_c': triplets_c,
            'n_micro_c': n_micro_c,
            'n_edges_c': n_edges_c,
            'comp_idx': comp_idx
        })

    # Diagnostic table
    if verbose:
        print(f"\n{'='*95}")
        print("BASIN FORMATION DIAGNOSTICS (first 10 connected components):")
        print(f"{'='*95}")
        print(f"{'#':<4} | {'Micro':<6} | {'Edges':<8} | {'Loc.Min.':<11} | {'Plateaus':<14} | {'Basins':<14} | {'Basins':<15} | Status")
        print(f"{'':<4} | {'':<6} | {'':<8} | {'(pre-merge)':<11} | {'(post-merge)':<14} | {'(post-descent)':<14} | {'(post-filter)':<15} |")
        print("-" * 95)

        limit = min(len(all_components_stats), 10)
        for i in range(limit):
            s = all_components_stats[i]
            print(f"{s['comp_idx']+1:<4} | {s['n_micro_c']:<6} | {s['n_edges_c']:<8} | "
                  f"{s['n_attractors']:<11} | {s['n_plateaus']:<14} | {s['n_basins_before_filter']:<14} | "
                  f"{s['n_basins_after_filter']:<15} | {s['status']}")

        if len(all_components_stats) > 10:
            print(f"... (showing 10 of {len(all_components_stats)} components)")
        print("-" * 95)

    # Fragmentation index
    n_significant_components = len(valid_components_data)

    if total_basins_all >= 3:
        total_triplets_possible = (total_basins_all * (total_basins_all - 1) *
                                   (total_basins_all - 2) // 6)
        f_inter = (1.0 - sum_triplets_all / total_triplets_possible
                   if total_triplets_possible > 0 else 1.0)
    else:
        f_inter = 1.0

    if verbose:
        print(f"\nCOMPONENT SUMMARY:")
        print(f"  Total connected components: {n_components_total}")
        print(f"  Significant components: {n_significant_components}")
        print(f"  Total basins in significant components: {total_basins_all}")

        if n_significant_components > 1:
            print(f"  Fragmentation index f_inter: {f_inter:.4f}")
            if f_inter > 0.8:
                print(f"  [WARNING] High fragmentation. Results reflect")
                print(f"  hierarchy within components, not global landscape.")
        elif n_significant_components == 1:
            print(f"  Fragmentation index f_inter: 0.0000 (all basins in one component)")
        else:
            print(f"  Fragmentation index f_inter: not computed (no significant basins)")

    # Compute ultrametricity per component
    weighted_u_nt = 0.0
    weighted_u_tr = 0.0
    weighted_u_non = 0.0
    total_weight = 0
    total_physical_modes = 0

    for comp_data in valid_components_data:
        res = compute_kinetic_metric_and_ultrametricity(
            comp_data['K_sym_c'],
            comp_data['basin_labels_c'],
            comp_data['n_basins_c'],
            num_modes, eigs_sigma, eigs_tol, dense_threshold,
            gap_threshold, eps, delta,
            quiet=True
        )

        if res is not None and res['n_triplets'] > 0:
            w = res['n_triplets']
            comp_data['u_nt'] = res['u_nontrivial']
            comp_data['u_tr'] = res['u_trivial']
            comp_data['u_non'] = res['u_nonultrametric']
            comp_data['triplets_actual'] = w

            weighted_u_nt += res['u_nontrivial'] * w
            weighted_u_tr += res['u_trivial'] * w
            weighted_u_non += res['u_nonultrametric'] * w
            total_weight += w
            total_physical_modes += res['n_physical_modes']
        else:
            comp_data['u_nt'] = None
            comp_data['u_tr'] = None
            comp_data['u_non'] = None
            comp_data['triplets_actual'] = 0

    # Significant components table
    if verbose and n_significant_components > 0:
        print(f"\nSIGNIFICANT CONNECTED COMPONENTS TABLE:")
        header = (f"{'#':<4} | {'Clusters':<9} | {'Edges':<8} | {'Triplets':<8} | "
                  f"{'u_nt (%)':<9} | {'u_tr (%)':<9} | {'u_non (%)':<9}")
        print(header)
        print("-" * len(header))

        limit = min(n_significant_components, 10)
        for i in range(limit):
            cd = valid_components_data[i]
            u_nt_str = f"{cd.get('u_nt', 0.0):.2f}" if cd.get('u_nt') is not None else "---"
            u_tr_str = f"{cd.get('u_tr', 0.0):.2f}" if cd.get('u_tr') is not None else "---"
            u_non_str = f"{cd.get('u_non', 0.0):.2f}" if cd.get('u_non') is not None else "---"

            print(f"{cd['comp_idx']+1:<4} | {cd['n_micro_c']:<9} | {cd['n_edges_c']:<8} | {cd['triplets_c']:<8} | "
                  f"{u_nt_str:<9} | {u_tr_str:<9} | {u_non_str:<9}")

        if n_significant_components > 10:
            print(f"... (showing 10 of {n_significant_components} significant components)")
        print("-" * len(header))

    # Weighted averages
    if total_weight > 0:
        final_u_nt = weighted_u_nt / total_weight
        final_u_tr = weighted_u_tr / total_weight
        final_u_non = weighted_u_non / total_weight
    else:
        final_u_nt = final_u_tr = final_u_non = 0.0

    # Failure diagnosis
    failure_reason = None
    if n_significant_components == 0:
        if n_components_total == 0:
            failure_reason = "neighbor graph contains no connected components"
        elif total_valid_basins_all_components == 0:
            failure_reason = f"no component has basins with size >= {min_basin_size}"
        elif max_basins_in_any_component < min_basins_per_comp:
            failure_reason = (f"max basins in any component ({max_basins_in_any_component}) "
                              f"is below MIN_BASINS_PER_COMPONENT ({min_basins_per_comp})")
        else:
            failure_reason = "no significant connected components found"
    elif total_weight == 0:
        if total_basins_all < 3:
            failure_reason = (f"total basins across significant components ({total_basins_all}) "
                              f"is less than 3 (cannot form triplets)")
        else:
            failure_reason = ("failed to compute kinetic metric for any triplet "
                              "(possibly all modes filtered as noise)")

    if failure_reason and verbose:
        print(f"\n[FAILURE REASON] Analysis produced no result: {failure_reason}.")

    return {
        'f_inter': f_inter,
        'n_valid_components': n_significant_components,
        'n_total_components': n_components_total,
        'total_basins': total_basins_all,
        'u_nontrivial': final_u_nt,
        'u_trivial': final_u_tr,
        'u_nonultrametric': final_u_non,
        'total_triplets': total_weight,
        'total_physical_modes': total_physical_modes,
        'failure_reason': failure_reason
    }


# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================
def run_sensitivity_analysis(K_sym, adjacency, free_energies, n_micro, temp_k,
                             min_basin_size, min_basins_per_comp, num_modes,
                             eigs_sigma, eigs_tol, dense_threshold,
                             gap_threshold, eps, delta, sens_list,
                             verbose=True):
    if verbose:
        print("\n" + "=" * 70)
        print("SENSITIVITY ANALYSIS TO PLATEAU THRESHOLD (EQ_THRESHOLD)")
        print("=" * 70)

        header = (f"{'EQ (kJ/mol)':<12} | {'f_inter':<8} | {'Comp.':<6} | "
                  f"{'Basins':<6} | {'u_nt (%)':<10} | {'u_tr (%)':<10} | {'u_non (%)':<10} | {'Triplets':<8}")
        print(header)
        print("-" * len(header))

    for eq_thresh in sens_list:
        res = analyze_landscape_components(
            K_sym, adjacency, free_energies, n_micro, temp_k,
            eq_thresh, min_basin_size, min_basins_per_comp,
            num_modes, eigs_sigma, eigs_tol, dense_threshold,
            gap_threshold, eps, delta,
            verbose=False
        )

        if verbose:
            if res is not None and res['total_triplets'] > 0:
                print(f"{eq_thresh:<12.2f} | {res['f_inter']:<8.4f} | "
                      f"{res['n_valid_components']:<6} | "
                      f"{res['total_basins']:<6} | "
                      f"{res['u_nontrivial']:>8.2f}   | "
                      f"{res['u_trivial']:>8.2f}   | "
                      f"{res['u_nonultrametric']:>8.2f}   | "
                      f"{res['total_triplets']:>6}")
            else:
                print(f"{eq_thresh:<12.2f} | {'---':<8} | {'---':<6} | "
                      f"{'---':<6} | {'---':>8}   | {'---':>8}   | {'---':>8}   | {'---':>6}")

    if verbose:
        print("-" * len(header))


# ============================================================================
# ALPHA SENSITIVITY ANALYSIS
# ============================================================================
def run_alpha_analysis(traj, n_states, radius_percentile, rmsd_mode,
                       original_center_indices, valid_clusters, filtered_labels,
                       free_energies, RT, basin_eq_threshold, min_basin_size,
                       min_basins_per_comp, num_modes, eigs_sigma, eigs_tol,
                       dense_threshold, gap_threshold, eps, delta, alpha_list,
                       verbose=True):
    if verbose:
        print("\n" + "=" * 70)
        print("SENSITIVITY ANALYSIS TO NEIGHBOR_ALPHA PARAMETER")
        print("=" * 70)

        header = (f"{'Alpha':<8} | {'f_inter':<8} | {'Comp.':<6} | "
                  f"{'Basins':<6} | {'u_nt (%)':<10} | {'u_tr (%)':<10} | {'u_non (%)':<10} | {'Triplets':<8}")
        print(header)
        print("-" * len(header))

    for alpha_val in alpha_list:
        adjacency = build_adaptive_topological_graph(
            traj, n_states, radius_percentile, rmsd_mode, alpha_val,
            original_center_indices, valid_clusters, filtered_labels,
            verbose=False
        )

        diff = np.abs(free_energies[:, None] - free_energies[None, :])
        rates = np.exp(-diff / (2.0 * RT))
        K_sym = np.where(adjacency, rates, 0.0)
        np.fill_diagonal(K_sym, 0.0)
        diag_values = -np.sum(K_sym, axis=1)
        np.fill_diagonal(K_sym, diag_values)

        res = analyze_landscape_components(
            K_sym, adjacency, free_energies, n_states, RT / KB_KJ,
            basin_eq_threshold, min_basin_size, min_basins_per_comp,
            num_modes, eigs_sigma, eigs_tol, dense_threshold,
            gap_threshold, eps, delta,
            verbose=False
        )

        if verbose:
            if res is not None and res['total_triplets'] > 0:
                print(f"{alpha_val:<8.2f} | {res['f_inter']:<8.4f} | "
                      f"{res['n_valid_components']:<6} | "
                      f"{res['total_basins']:<6} | "
                      f"{res['u_nontrivial']:>8.2f}   | "
                      f"{res['u_trivial']:>8.2f}   | "
                      f"{res['u_nonultrametric']:>8.2f}   | "
                      f"{res['total_triplets']:>6}")
            else:
                print(f"{alpha_val:<8.2f} | {'---':<8} | {'---':<6} | "
                      f"{'---':<6} | {'---':>8}   | {'---':>8}   | {'---':>8}   | {'---':>6}")

    if verbose:
        print("-" * len(header))


# ============================================================================
# NULL MODEL H1: ENERGY SHUFFLE
# ============================================================================
def run_null_h1_energy_shuffle(K_sym, adjacency, free_energies, n_micro,
                               temp_k, eq_threshold, min_basin_size,
                               min_basins_per_comp, num_modes,
                               eigs_sigma, eigs_tol, dense_threshold,
                               gap_threshold, eps, delta,
                               n_samples, base_seed, verbose=True):
    if verbose:
        print("\n" + "=" * 70)
        print("NULL HYPOTHESIS H1 TEST: ENERGY SHUFFLE")
        print("  (shuffling cluster free energies)")
        print("  Basins and connected components are FIXED")
        print(f"  Number of realizations: {n_samples}")
        print("=" * 70)

    components = find_connected_components(adjacency)
    valid_components_data = []

    for comp_idx, comp_micro_indices in enumerate(components):
        adj_c = adjacency[np.ix_(comp_micro_indices, comp_micro_indices)]
        fe_c = free_energies[comp_micro_indices]
        fe_c = fe_c - fe_c.min()

        basin_labels_local, n_basins_raw, _ = identify_basins_gradient_descent(
            fe_c, adj_c, eq_threshold
        )

        unique_basins, basin_counts = np.unique(basin_labels_local, return_counts=True)
        valid_basins = unique_basins[basin_counts >= min_basin_size]
        n_basins_c = len(valid_basins)

        if n_basins_c < min_basins_per_comp:
            continue

        basin_map = {old: new for new, old in enumerate(valid_basins)}
        final_basin_labels = np.array(
            [basin_map[b] if b in basin_map else -1 for b in basin_labels_local]
        )

        valid_components_data.append({
            'adj_c': adj_c,
            'basin_labels_c': final_basin_labels,
            'n_basins_c': n_basins_c,
            'comp_micro_indices': comp_micro_indices
        })

    if len(valid_components_data) == 0:
        if verbose:
            print("  ERROR: No significant components for H1.")
        return None

    if verbose:
        print(f"  Significant components (fixed): {len(valid_components_data)}")
        total_basins_fixed = sum(c['n_basins_c'] for c in valid_components_data)
        print(f"  Total basins (fixed): {total_basins_fixed}")

    u_nt_null = []
    u_tr_null = []
    u_non_null = []
    n_basins_null = []

    t0 = time.time()
    RT = KB_KJ * temp_k

    for s in range(n_samples):
        rng = np.random.RandomState(base_seed + s)
        permuted_fe = free_energies.copy()
        rng.shuffle(permuted_fe)

        weighted_u_nt = 0.0
        weighted_u_tr = 0.0
        weighted_u_non = 0.0
        total_weight = 0
        total_basins_this_sample = 0

        for comp_data in valid_components_data:
            comp_micro_indices = comp_data['comp_micro_indices']
            adj_c = comp_data['adj_c']
            basin_labels_c = comp_data['basin_labels_c']
            n_basins_c = comp_data['n_basins_c']

            fe_c_perm = permuted_fe[comp_micro_indices]
            fe_c_perm = fe_c_perm - fe_c_perm.min()

            diff = np.abs(fe_c_perm[:, None] - fe_c_perm[None, :])
            rates = np.exp(-diff / (2.0 * RT))
            K_sym_c_perm = np.where(adj_c, rates, 0.0)
            np.fill_diagonal(K_sym_c_perm, 0.0)
            diag_values = -np.sum(K_sym_c_perm, axis=1)
            np.fill_diagonal(K_sym_c_perm, diag_values)

            res = compute_kinetic_metric_and_ultrametricity(
                K_sym_c_perm, basin_labels_c, n_basins_c,
                num_modes, eigs_sigma, eigs_tol, dense_threshold,
                gap_threshold, eps, delta, quiet=True
            )

            if res is not None and res['n_triplets'] > 0:
                w = res['n_triplets']
                weighted_u_nt += res['u_nontrivial'] * w
                weighted_u_tr += res['u_trivial'] * w
                weighted_u_non += res['u_nonultrametric'] * w
                total_weight += w
                total_basins_this_sample += n_basins_c

        if total_weight > 0:
            u_nt_null.append(weighted_u_nt / total_weight)
            u_tr_null.append(weighted_u_tr / total_weight)
            u_non_null.append(weighted_u_non / total_weight)
            n_basins_null.append(total_basins_this_sample)

        if verbose and (s + 1) % max(1, n_samples // 10) == 0:
            elapsed = time.time() - t0
            print(f"  Completed {s+1}/{n_samples} realizations ({elapsed:.1f} sec)")

    elapsed = time.time() - t0

    if len(u_nt_null) == 0:
        if verbose:
            print("  ERROR: All H1 realizations failed.")
        return None

    u_nt_null = np.array(u_nt_null)
    u_tr_null = np.array(u_tr_null)
    u_non_null = np.array(u_non_null)
    n_basins_null = np.array(n_basins_null)

    mean_nt = np.mean(u_nt_null)
    std_nt = np.std(u_nt_null)
    mean_tr = np.mean(u_tr_null)
    std_tr = np.std(u_tr_null)
    mean_non = np.mean(u_non_null)
    std_non = np.std(u_non_null)

    if verbose:
        print(f"\nNULL MODEL H1 RESULTS (Energy Shuffle):")
        print(f"  Computation time: {elapsed:.1f} sec")
        print(f"  Successful realizations: {len(u_nt_null)} of {n_samples}")
        print(f"  Mean basin count: {np.mean(n_basins_null):.1f}")
        print(f"  Null distribution (mean +/- std):")
        print(f"    u_nt (non-trivial):    {mean_nt:.2f} +/- {std_nt:.2f} %")
        print(f"    u_tr (trivial):        {mean_tr:.2f} +/- {std_tr:.2f} %")
        print(f"    u_non (non-ultramet.): {mean_non:.2f} +/- {std_non:.2f} %")

    return {
        'u_nt_null': u_nt_null,
        'u_tr_null': u_tr_null,
        'u_non_null': u_non_null,
        'n_basins_null': n_basins_null,
        'mean_nt': mean_nt,
        'std_nt': std_nt,
        'mean_tr': mean_tr,
        'std_tr': std_tr,
        'mean_non': mean_non,
        'std_non': std_non,
        'n_valid': len(u_nt_null),
        'elapsed': elapsed
    }


def print_null_h1_results(results_h1, u_nt_real, u_tr_real, u_non_real, verbose=True):
    if results_h1 is None:
        if verbose:
            print("\nH1: no data to display.")
        return

    null_vals_nt = results_h1['u_nt_null']
    mean_null_nt = results_h1['mean_nt']

    deviation_real = abs(u_nt_real - mean_null_nt)
    deviations_null = np.abs(null_vals_nt - mean_null_nt)
    count_extreme = np.sum(deviations_null >= deviation_real)
    n_samples = len(null_vals_nt)
    p_value = (count_extreme + 1) / (n_samples + 1)

    if verbose:
        print("\n" + "=" * 70)
        print("H1 STATISTICAL SIGNIFICANCE (Energy Shuffle):")
        print("=" * 70)
        print(f"  {'Metric':<25} | {'Real':>10} | {'Null (mean+/-std)':>20}")
        print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*20}")
        print(f"  {'u_nt (non-trivial)':<25} | {u_nt_real:>9.2f}% | {results_h1['mean_nt']:>8.2f} +/- {results_h1['std_nt']:.2f}%")
        print(f"  {'u_tr (trivial)':<25} | {u_tr_real:>9.2f}% | {results_h1['mean_tr']:>8.2f} +/- {results_h1['std_tr']:.2f}%")
        print(f"  {'u_non (non-ultramet.)':<25} | {u_non_real:>9.2f}% | {results_h1['mean_non']:>8.2f} +/- {results_h1['std_non']:.2f}%")
        print(f"  ")
        print(f"  Number of realizations:       {results_h1['n_valid']}")
        print(f"  Two-sided p-value (by u_nt):  {p_value:.4f}")

        if p_value < 0.01:
            print(f"  CONCLUSION: Statistically significant at p < 0.01")
        elif p_value < 0.05:
            print(f"  CONCLUSION: Statistically significant at p < 0.05")
        else:
            print(f"  CONCLUSION: Not statistically significant (p >= 0.05)")
        print("=" * 70)


# ============================================================================
# MAIN ANALYSIS
# ============================================================================
def run_main_analysis(n_samples, base_seed,
                      pdb_id, use_gb99dms, temp_k, md_steps,
                      md_report_interval, use_gpu,
                      n_clusters, min_cluster_size,
                      kramers_temperature, neighbor_method,
                      eta_neighbor, neighbor_radius_percentile,
                      geometry_rmsd_mode, neighbor_alpha,
                      eta_neighbor_mode,
                      basin_eq_threshold, min_basin_size,
                      min_basins_per_comp, num_modes,
                      eigs_sigma, eigs_tol, eigs_dense_threshold,
                      spectral_gap_threshold,
                      ultrametric_epsilon, ultrametric_delta,
                      starting_structure_file,
                      add_hydrogens_to_custom_start,
                      minimize_starting_structure,
                      md_time_step_fs, md_rollback_steps, md_max_rollbacks,
                      md_time_step_reduction_factor, md_min_time_step_fs,
                      verbose=True):
    if verbose:
        print("\n" + "=" * 70)
        print("MAIN ANALYSIS: MULTIPLE RUNS")
        print("  (full cycle MD -> MSM -> Analysis with different seeds)")
        print(f"  Number of runs: {n_samples}")
        print("=" * 70)

    u_nt_list = []
    u_tr_list = []
    u_non_list = []
    n_basins_list = []
    f_inter_list = []
    failure_reasons = []

    t0_total = time.time()

    for run_idx in range(n_samples):
        current_seed = base_seed + run_idx

        if verbose:
            print(f"\n--- Run {run_idx+1}/{n_samples} (seed = {current_seed}) ---")

        try:
            dcd_file, pdb_file, energy_file = run_md_simulation(
                pdb_id, use_gb99dms, temp_k, md_steps, md_report_interval,
                use_gpu, current_seed, starting_structure_file,
                add_hydrogens_to_custom_start, minimize_starting_structure,
                md_time_step_fs, md_rollback_steps, md_max_rollbacks,
                md_time_step_reduction_factor, md_min_time_step_fs,
                verbose=verbose
            )

            if dcd_file is None:
                if verbose:
                    print(f"  [MAIN] Run {run_idx+1}: MD failed (skipped).")
                continue

            effective_eta = eta_neighbor
            if neighbor_method == 'energy' and eta_neighbor_mode == 'auto':
                RT_temp = KB_KJ * kramers_temperature
                _, _, fe_temp, _ = build_msm_and_graph(
                    dcd_file, pdb_file, energy_file,
                    n_clusters, min_cluster_size,
                    kramers_temperature, neighbor_method, 1e-6,
                    neighbor_radius_percentile, geometry_rmsd_mode, neighbor_alpha,
                    verbose=False
                )
                if fe_temp is not None:
                    effective_eta = auto_select_eta_neighbor(
                        fe_temp, RT_temp, min_basin_size,
                        min_basins_per_comp, basin_eq_threshold,
                        verbose=verbose
                    )

            K_sym, adjacency, free_energies, n_micro = build_msm_and_graph(
                dcd_file, pdb_file, energy_file,
                n_clusters, min_cluster_size,
                kramers_temperature, neighbor_method, effective_eta,
                neighbor_radius_percentile, geometry_rmsd_mode, neighbor_alpha,
                verbose=verbose
            )

            if K_sym is None:
                if verbose:
                    print(f"  [MAIN] Run {run_idx+1}: MSM construction failed (skipped).")
                continue

            results = analyze_landscape_components(
                K_sym, adjacency, free_energies, n_micro,
                kramers_temperature,
                basin_eq_threshold, min_basin_size, min_basins_per_comp,
                num_modes, eigs_sigma, eigs_tol, eigs_dense_threshold,
                spectral_gap_threshold,
                ultrametric_epsilon, ultrametric_delta,
                verbose=verbose
            )

            if results is not None and results['total_triplets'] > 0:
                u_nt_list.append(results['u_nontrivial'])
                u_tr_list.append(results['u_trivial'])
                u_non_list.append(results['u_nonultrametric'])
                n_basins_list.append(results['total_basins'])
                f_inter_list.append(results['f_inter'])

                if verbose:
                    print(f"  [MAIN] Run {run_idx+1}: u_nt={results['u_nontrivial']:.2f}%, "
                          f"u_tr={results['u_trivial']:.2f}%, u_non={results['u_nonultrametric']:.2f}%")
            else:
                reason = results.get('failure_reason', 'unknown') if results else 'unknown'
                failure_reasons.append(reason)
                if verbose:
                    print(f"  [MAIN] Run {run_idx+1}: failed to compute u_nt (skipped).")

        except Exception as e:
            if verbose:
                print(f"  [MAIN] ERROR in run {run_idx+1} (seed={current_seed}): {type(e).__name__}")
                print(f"         Message: {str(e)[:200]}")
                print(f"         Run skipped, proceeding to next realization.")
            continue

    elapsed_total = time.time() - t0_total

    if len(u_nt_list) == 0:
        if verbose:
            print("\nERROR: All MAIN_ANALYSIS runs failed.")
            if failure_reasons:
                unique_reasons = list(set(failure_reasons))
                print("Failure reasons by run:")
                for r in unique_reasons:
                    count = failure_reasons.count(r)
                    print(f"  - {r} (in {count} runs)")
        return None

    u_nt_list = np.array(u_nt_list)
    u_tr_list = np.array(u_tr_list)
    u_non_list = np.array(u_non_list)
    n_basins_list = np.array(n_basins_list)
    f_inter_list = np.array(f_inter_list)

    if verbose:
        print(f"\nMAIN ANALYSIS RESULTS:")
        print(f"  Total time: {elapsed_total:.1f} sec ({elapsed_total/60:.1f} min)")
        print(f"  Successful runs: {len(u_nt_list)} of {n_samples}")

    return {
        'u_nt_list': u_nt_list,
        'u_tr_list': u_tr_list,
        'u_non_list': u_non_list,
        'n_basins_list': n_basins_list,
        'f_inter_list': f_inter_list,
        'mean_nt': np.mean(u_nt_list),
        'std_nt': np.std(u_nt_list),
        'mean_tr': np.mean(u_tr_list),
        'std_tr': np.std(u_tr_list),
        'mean_non': np.mean(u_non_list),
        'std_non': np.std(u_non_list),
        'n_valid': len(u_nt_list),
        'elapsed': elapsed_total
    }


def run_main_analysis_from_files(K_sym, adjacency, free_energies, n_micro,
                                 kramers_temperature,
                                 basin_eq_threshold, min_basin_size,
                                 min_basins_per_comp, num_modes,
                                 eigs_sigma, eigs_tol, eigs_dense_threshold,
                                 spectral_gap_threshold,
                                 ultrametric_epsilon, ultrametric_delta,
                                 verbose=True):
    if verbose:
        print("\n" + "=" * 70)
        print("MAIN ANALYSIS FROM FILES")
        print("  (single run, since FROM_FILES_ANALYSIS = True)")
        print("  Using pre-built MSM (no re-clustering).")
        print("=" * 70)

    if K_sym is None:
        return None

    results = analyze_landscape_components(
        K_sym, adjacency, free_energies, n_micro,
        kramers_temperature,
        basin_eq_threshold, min_basin_size, min_basins_per_comp,
        num_modes, eigs_sigma, eigs_tol, eigs_dense_threshold,
        spectral_gap_threshold,
        ultrametric_epsilon, ultrametric_delta,
        verbose=verbose
    )

    return results


def print_main_analysis_results(results_main, verbose=True):
    if results_main is None:
        if verbose:
            print("\nMAIN_ANALYSIS: no data to display.")
        return

    if verbose:
        print("\n" + "=" * 70)
        print("MAIN ANALYSIS SUMMARY TABLE")
        print("=" * 70)

        header = f"{'Run':<8} | {'u_nt (%)':<10} | {'u_tr (%)':<10} | {'u_non (%)':<10} | {'Basins':<6} | {'f_inter':<8}"
        print(header)
        print("-" * len(header))

        for i in range(len(results_main['u_nt_list'])):
            print(f"{i+1:<8} | {results_main['u_nt_list'][i]:>8.2f}   | "
                  f"{results_main['u_tr_list'][i]:>8.2f}   | "
                  f"{results_main['u_non_list'][i]:>8.2f}   | "
                  f"{results_main['n_basins_list'][i]:>5} | "
                  f"{results_main['f_inter_list'][i]:>7.4f}")

        print("-" * len(header))

        mean_nt = results_main['mean_nt']
        std_nt = results_main['std_nt']
        cv = (std_nt / max(abs(mean_nt), 0.01)) * 100

        print(f"\nFINAL STATISTICS AND STABILITY REPORT:")
        print(f"  Successful runs:            {results_main['n_valid']}")
        print(f"  Mean u_nt:                  {mean_nt:.2f} %")
        print(f"  Standard deviation (STD):   {std_nt:.2f} %")
        print(f"  Coefficient of variation:   {cv:.1f} %")

        if cv < 5.0:
            print(f"  CONCLUSION: HIGHLY STABLE result (CV < 5%).")
        elif cv < 10.0:
            print(f"  CONCLUSION: Stable result (5% <= CV < 10%).")
        elif cv < 20.0:
            print(f"  CONCLUSION: Moderately stable (10% <= CV < 20%).")
        else:
            print(f"  CONCLUSION: UNSTABLE result (CV >= 20%). Increase trajectory length.")
        print("=" * 70)


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    v = VERBOSE

    print("=" * 70)
    print("ULTRAMETRICITY ANALYSIS OF PROTEIN ENERGY LANDSCAPE")
    print("Kinetic metric based on spectral decomposition (v1.0)")
    print("=" * 70)
    print(f"  Protein:              {PDB_ID}")
    print(f"  Force field:          {'GB99dms' if USE_GB99DMS else 'Amber14/OBC2'}")
    print(f"  MD Temperature:       {TEMPERATURE_K} K")
    print(f"  MD Steps:             {MD_STEPS}, save interval: {MD_REPORT_INTERVAL}")
    print(f"  MD Timestep (init):   {MD_TIME_STEP_FS} fs, rollback: {MD_ROLLBACK_STEPS} steps")
    print(f"  Kramers Temperature:  {KRAMERS_TEMPERATURE} K")
    print(f"  Neighbor Method:      {NEIGHBOR_METHOD}")

    if NEIGHBOR_METHOD == 'energy':
        print(f"  Eta Mode:             {ETA_NEIGHBOR_MODE}")
        if ETA_NEIGHBOR_MODE == 'manual':
            RT_print = KB_KJ * KRAMERS_TEMPERATURE
            df_max_print = -RT_print * np.log(ETA_NEIGHBOR)
            print(f"  eta (manual):         {ETA_NEIGHBOR:.0e} -> dF_max = {df_max_print:.1f} kJ/mol")
    elif NEIGHBOR_METHOD == 'adaptive_topology':
        print(f"  Radius Percentile:    {NEIGHBOR_RADIUS_PERCENTILE}%")
        print(f"  Neighbor Alpha:       {NEIGHBOR_ALPHA}")

    print(f"  RMSD Metric:          {'C-alpha' if GEOMETRY_RMSD_MODE == 'ca' else 'all atoms'}")
    print(f"  Microstates:          {N_CLUSTERS} (target), min size: {MIN_CLUSTER_SIZE}")
    print(f"  Plateau Threshold:    {BASIN_EQ_THRESHOLD} kJ/mol, min basin size: {MIN_BASIN_SIZE}")
    print(f"  Min Basins/Component: {MIN_BASINS_PER_COMPONENT}")
    print(f"  Eigenmodes:           {NUM_EIGENMODES} (gap threshold: {SPECTRAL_GAP_THRESHOLD:.0e})")
    print(f"  EIGS sigma:           {EIGS_SIGMA:.0e}, tol: {EIGS_TOL:.0e}, dense threshold: {EIGS_DENSE_THRESHOLD}")
    print(f"  Ultrametricity:       eps={ULTRAMETRIC_EPSILON}, delta={ULTRAMETRIC_DELTA}")
    print(f"  CPU Threads:          {num_cores} (limit: {MAX_CPU_THREADS})")
    print("-" * 70)
    print("  SELECTED EXECUTION MODES:")
    print(f"  FROM_FILES_ANALYSIS:  {'Yes' if FROM_FILES_ANALYSIS else 'No'}")
    print(f"  Sensitivity Analysis: {'Yes (list: ' + str(SENSITIVITY_RANGE) + ')' if SENSITIVITY_ANALYSIS else 'No'}")
    print(f"  Alpha Analysis:       {'Yes (list: ' + str(ALPHA_RANGE) + ')' if ALPHA_ANALYSIS else 'No'}")
    print(f"  H1 (Energy Shuffle):  {'Yes (' + str(NULL_H1_NUM_SAMPLES) + ')' if NULL_H1_ENERGY_SHUFFLE else 'No'}")
    print(f"  MAIN_ANALYSIS:        {'Yes (' + str(MAIN_NUM_SAMPLES) + ')' if MAIN_ANALYSIS else 'No'}")
    print(f"  GPU:                  {'Yes' if USE_GPU else 'No'}")
    print(f"  Verbose:              {'Yes' if v else 'No'}")
    print("=" * 70)

    sens_active = SENSITIVITY_ANALYSIS and isinstance(SENSITIVITY_RANGE, list) and len(SENSITIVITY_RANGE) > 0
    alpha_active = ALPHA_ANALYSIS and isinstance(ALPHA_RANGE, list) and len(ALPHA_RANGE) > 0

    if alpha_active and NEIGHBOR_METHOD != 'adaptive_topology':
        if v:
            print("\n[WARNING] ALPHA_ANALYSIS enabled but NEIGHBOR_METHOD != 'adaptive_topology'.")
            print("Alpha sensitivity analysis only applies to adaptive topological graphs.")
            print("ALPHA_ANALYSIS mode will be skipped.")
        alpha_active = False

    h1_active = NULL_H1_ENERGY_SHUFFLE
    main_active = MAIN_ANALYSIS

    # Mode conflict check
    if sens_active and alpha_active:
        print("\n" + "=" * 70)
        print("[ERROR] ANALYSIS MODE CONFLICT!")
        print("SENSITIVITY_ANALYSIS and ALPHA_ANALYSIS cannot be True simultaneously.")
        print("Please select only one sensitivity analysis mode.")
        print("=" * 70)
        return None

    if not (sens_active or alpha_active or h1_active or main_active):
        print("\n" + "=" * 70)
        print("ALL ANALYTICAL MODES ARE DISABLED.")
        print("Enable at least one of the following flags:")
        print("  SENSITIVITY_ANALYSIS = True (with non-empty SENSITIVITY_RANGE)")
        print("  ALPHA_ANALYSIS = True (with non-empty ALPHA_RANGE)")
        print("  NULL_H1_ENERGY_SHUFFLE = True")
        print("  MAIN_ANALYSIS = True")
        print("Program terminating.")
        print("=" * 70)
        return None

    total_start = time.time()

    if FROM_FILES_ANALYSIS:
        print("\n" + "=" * 70)
        print(">>> FROM_FILES_ANALYSIS MODE: LOADING DATA FROM DISK <<<")
        print("=" * 70)

        dcd_file, pdb_file, energy_file = load_trajectory_and_energies_from_files(
            PDB_ID, STARTING_STRUCTURE_FILE, USE_GB99DMS, USE_GPU,
            KRAMERS_TEMPERATURE, verbose=v
        )

        if dcd_file is None:
            print("\n[ERROR] Failed to load data from files.")
            print("Program terminated.")
            return None

        effective_eta = ETA_NEIGHBOR
        if NEIGHBOR_METHOD == 'energy' and ETA_NEIGHBOR_MODE == 'auto':
            RT_main = KB_KJ * KRAMERS_TEMPERATURE
            _, _, fe_for_auto, _ = build_msm_and_graph(
                dcd_file, pdb_file, energy_file,
                N_CLUSTERS, MIN_CLUSTER_SIZE,
                KRAMERS_TEMPERATURE, NEIGHBOR_METHOD, 1e-6,
                NEIGHBOR_RADIUS_PERCENTILE, GEOMETRY_RMSD_MODE, NEIGHBOR_ALPHA,
                verbose=False
            )
            if fe_for_auto is not None:
                effective_eta = auto_select_eta_neighbor(
                    fe_for_auto, RT_main, MIN_BASIN_SIZE,
                    MIN_BASINS_PER_COMPONENT, BASIN_EQ_THRESHOLD,
                    verbose=v
                )
            else:
                if v:
                    print("  [WARNING] Could not obtain free_energies for auto eta selection.")

        K_sym, adjacency, free_energies, n_micro = build_msm_and_graph(
            dcd_file, pdb_file, energy_file,
            N_CLUSTERS, MIN_CLUSTER_SIZE,
            KRAMERS_TEMPERATURE, NEIGHBOR_METHOD, effective_eta,
            NEIGHBOR_RADIUS_PERCENTILE, GEOMETRY_RMSD_MODE, NEIGHBOR_ALPHA,
            verbose=v
        )

        if K_sym is None:
            print("[ERROR] Failed to build MSM. Program terminated.")
            return None

        results_base = None
        u_nt_real = None
        u_tr_real = None
        u_non_real = None

        if h1_active or sens_active:
            results_base = analyze_landscape_components(
                K_sym, adjacency, free_energies, n_micro,
                KRAMERS_TEMPERATURE,
                BASIN_EQ_THRESHOLD, MIN_BASIN_SIZE, MIN_BASINS_PER_COMPONENT,
                NUM_EIGENMODES, EIGS_SIGMA, EIGS_TOL, EIGS_DENSE_THRESHOLD,
                SPECTRAL_GAP_THRESHOLD,
                ULTRAMETRIC_EPSILON, ULTRAMETRIC_DELTA,
                verbose=v
            )
            if results_base is not None and results_base['total_triplets'] > 0:
                u_nt_real = results_base['u_nontrivial']
                u_tr_real = results_base['u_trivial']
                u_non_real = results_base['u_nonultrametric']

        if sens_active:
            run_sensitivity_analysis(
                K_sym=K_sym,
                adjacency=adjacency,
                free_energies=free_energies,
                n_micro=n_micro,
                temp_k=KRAMERS_TEMPERATURE,
                min_basin_size=MIN_BASIN_SIZE,
                min_basins_per_comp=MIN_BASINS_PER_COMPONENT,
                num_modes=NUM_EIGENMODES,
                eigs_sigma=EIGS_SIGMA,
                eigs_tol=EIGS_TOL,
                dense_threshold=EIGS_DENSE_THRESHOLD,
                gap_threshold=SPECTRAL_GAP_THRESHOLD,
                eps=ULTRAMETRIC_EPSILON,
                delta=ULTRAMETRIC_DELTA,
                sens_list=SENSITIVITY_RANGE,
                verbose=v
            )

        if alpha_active:
            if v:
                print("\n" + "=" * 70)
                print(">>> LAUNCHING MODE: ALPHA SENSITIVITY ANALYSIS <<<")
                print("=" * 70)

            import mdtraj as md
            if v:
                print("  Loading trajectory for ALPHA_ANALYSIS...")

            traj = md.load(dcd_file, top=pdb_file)
            energies = np.loadtxt(energy_file)

            if GEOMETRY_RMSD_MODE == 'ca':
                cluster_atom_indices = [a.index for a in traj.topology.atoms if a.name == 'CA']
            else:
                cluster_atom_indices = None

            if v:
                print(f"  K-centers clustering (baseline for ALPHA cycle)...")

            cluster_obj = ClusterKCenters(traj, N_CLUSTERS, cluster_atom_indices)
            labels = cluster_obj.assign(traj)
            original_center_indices = cluster_obj.center_indices

            unique, counts = np.unique(labels, return_counts=True)
            valid_clusters = unique[counts >= MIN_CLUSTER_SIZE]
            label_map = {old: new for new, old in enumerate(valid_clusters)}
            filtered_labels = np.array([label_map[l] if l in label_map else -1 for l in labels])
            n_states = len(valid_clusters)

            RT = KB_KJ * KRAMERS_TEMPERATURE
            free_energies_alpha = np.zeros(n_states)
            for i in range(n_states):
                mask = (filtered_labels == i)
                if np.sum(mask) > 0:
                    cluster_energies = energies[mask]
                    e_min = cluster_energies.min()
                    boltzmann_sum = np.sum(np.exp(-(cluster_energies - e_min) / RT))
                    free_energies_alpha[i] = e_min - RT * np.log(max(boltzmann_sum, 1e-300))
            free_energies_alpha -= free_energies_alpha.min()

            run_alpha_analysis(
                traj, n_states, NEIGHBOR_RADIUS_PERCENTILE, GEOMETRY_RMSD_MODE,
                original_center_indices, valid_clusters, filtered_labels,
                free_energies_alpha, RT, BASIN_EQ_THRESHOLD, MIN_BASIN_SIZE,
                MIN_BASINS_PER_COMPONENT, NUM_EIGENMODES,
                EIGS_SIGMA, EIGS_TOL, EIGS_DENSE_THRESHOLD,
                SPECTRAL_GAP_THRESHOLD,
                ULTRAMETRIC_EPSILON, ULTRAMETRIC_DELTA,
                ALPHA_RANGE, verbose=v
            )

            del traj
            gc.collect()

        if h1_active:
            if results_base is not None and results_base['total_triplets'] > 0:
                results_h1 = run_null_h1_energy_shuffle(
                    K_sym, adjacency, free_energies, n_micro,
                    KRAMERS_TEMPERATURE,
                    BASIN_EQ_THRESHOLD, MIN_BASIN_SIZE, MIN_BASINS_PER_COMPONENT,
                    NUM_EIGENMODES, EIGS_SIGMA, EIGS_TOL, EIGS_DENSE_THRESHOLD,
                    SPECTRAL_GAP_THRESHOLD,
                    ULTRAMETRIC_EPSILON, ULTRAMETRIC_DELTA,
                    NULL_H1_NUM_SAMPLES, NULL_SEED_BASE,
                    verbose=v
                )
                print_null_h1_results(results_h1, u_nt_real, u_tr_real, u_non_real, verbose=v)
            else:
                print("[ERROR] Baseline calculation failed. H1 mode skipped.")

        if main_active:
            if MAIN_NUM_SAMPLES > 1 and v:
                print(f"\n[WARNING] In FROM_FILES_ANALYSIS mode, MAIN_NUM_SAMPLES "
                      f"(={MAIN_NUM_SAMPLES}) is forced to 1,")
                print(f"           since repeated analysis of the same trajectory "
                      f"with identical clustering parameters has no physical meaning.")

            results_main_single = run_main_analysis_from_files(
                K_sym, adjacency, free_energies, n_micro,
                KRAMERS_TEMPERATURE,
                BASIN_EQ_THRESHOLD, MIN_BASIN_SIZE,
                MIN_BASINS_PER_COMPONENT, NUM_EIGENMODES,
                EIGS_SIGMA, EIGS_TOL, EIGS_DENSE_THRESHOLD,
                SPECTRAL_GAP_THRESHOLD,
                ULTRAMETRIC_EPSILON, ULTRAMETRIC_DELTA,
                verbose=v
            )

            if results_main_single is not None and results_main_single['total_triplets'] > 0:
                if v:
                    print(f"\nMAIN ANALYSIS RESULT (single run):")
                    print(f"  u_nt (non-trivial):    {results_main_single['u_nontrivial']:.2f}%")
                    print(f"  u_tr (trivial):        {results_main_single['u_trivial']:.2f}%")
                    print(f"  u_non (non-ultramet.): {results_main_single['u_nonultrametric']:.2f}%")
                    print(f"  Total basins:          {results_main_single['total_basins']}")
                    print(f"  Fragmentation index:   {results_main_single['f_inter']:.4f}")
            else:
                print("[ERROR] Main analysis produced no result.")

    else:
        # Non-FROM_FILES modes: each mode runs its own MD
        if sens_active:
            print("\n" + "=" * 70)
            print(">>> LAUNCHING MODE 1: SENSITIVITY ANALYSIS <<<")
            print("=" * 70)

            dcd_file, pdb_file, energy_file = run_md_simulation(
                PDB_ID, USE_GB99DMS, TEMPERATURE_K,
                MD_STEPS, MD_REPORT_INTERVAL, USE_GPU, MD_SEED,
                STARTING_STRUCTURE_FILE, ADD_HYDROGENS_TO_CUSTOM_START,
                MINIMIZE_STARTING_STRUCTURE,
                MD_TIME_STEP_FS, MD_ROLLBACK_STEPS, MD_MAX_ROLLBACKS,
                MD_TIME_STEP_REDUCTION_FACTOR, MD_MIN_TIME_STEP_FS,
                verbose=v
            )

            if dcd_file is None:
                print("[ERROR] MD failed. Sensitivity mode skipped.")
            else:
                effective_eta = ETA_NEIGHBOR
                if NEIGHBOR_METHOD == 'energy' and ETA_NEIGHBOR_MODE == 'auto':
                    RT_main = KB_KJ * KRAMERS_TEMPERATURE
                    _, _, fe_for_auto, _ = build_msm_and_graph(
                        dcd_file, pdb_file, energy_file,
                        N_CLUSTERS, MIN_CLUSTER_SIZE,
                        KRAMERS_TEMPERATURE, NEIGHBOR_METHOD, 1e-6,
                        NEIGHBOR_RADIUS_PERCENTILE, GEOMETRY_RMSD_MODE, NEIGHBOR_ALPHA,
                        verbose=False
                    )
                    if fe_for_auto is not None:
                        effective_eta = auto_select_eta_neighbor(
                            fe_for_auto, RT_main, MIN_BASIN_SIZE,
                            MIN_BASINS_PER_COMPONENT, BASIN_EQ_THRESHOLD,
                            verbose=v
                        )
                    else:
                        if v:
                            print("  [WARNING] Could not obtain free_energies for auto eta selection.")

                K_sym, adjacency, free_energies, n_micro = build_msm_and_graph(
                    dcd_file, pdb_file, energy_file,
                    N_CLUSTERS, MIN_CLUSTER_SIZE,
                    KRAMERS_TEMPERATURE, NEIGHBOR_METHOD, effective_eta,
                    NEIGHBOR_RADIUS_PERCENTILE, GEOMETRY_RMSD_MODE, NEIGHBOR_ALPHA,
                    verbose=v
                )

                if K_sym is None:
                    print("[ERROR] MSM construction failed. Sensitivity mode skipped.")
                else:
                    run_sensitivity_analysis(
                        K_sym=K_sym,
                        adjacency=adjacency,
                        free_energies=free_energies,
                        n_micro=n_micro,
                        temp_k=KRAMERS_TEMPERATURE,
                        min_basin_size=MIN_BASIN_SIZE,
                        min_basins_per_comp=MIN_BASINS_PER_COMPONENT,
                        num_modes=NUM_EIGENMODES,
                        eigs_sigma=EIGS_SIGMA,
                        eigs_tol=EIGS_TOL,
                        dense_threshold=EIGS_DENSE_THRESHOLD,
                        gap_threshold=SPECTRAL_GAP_THRESHOLD,
                        eps=ULTRAMETRIC_EPSILON,
                        delta=ULTRAMETRIC_DELTA,
                        sens_list=SENSITIVITY_RANGE,
                        verbose=v
                    )

        if alpha_active:
            print("\n" + "=" * 70)
            print(">>> LAUNCHING MODE: ALPHA SENSITIVITY ANALYSIS <<<")
            print("=" * 70)

            dcd_file, pdb_file, energy_file = run_md_simulation(
                PDB_ID, USE_GB99DMS, TEMPERATURE_K,
                MD_STEPS, MD_REPORT_INTERVAL, USE_GPU, MD_SEED,
                STARTING_STRUCTURE_FILE, ADD_HYDROGENS_TO_CUSTOM_START,
                MINIMIZE_STARTING_STRUCTURE,
                MD_TIME_STEP_FS, MD_ROLLBACK_STEPS, MD_MAX_ROLLBACKS,
                MD_TIME_STEP_REDUCTION_FACTOR, MD_MIN_TIME_STEP_FS,
                verbose=v
            )

            if dcd_file is None:
                print("[ERROR] MD failed. ALPHA mode skipped.")
            else:
                import mdtraj as md
                traj = md.load(dcd_file, top=pdb_file)
                energies = np.loadtxt(energy_file)

                if GEOMETRY_RMSD_MODE == 'ca':
                    cluster_atom_indices = [a.index for a in traj.topology.atoms if a.name == 'CA']
                else:
                    cluster_atom_indices = None

                if v:
                    print(f"  K-centers clustering (baseline for ALPHA cycle)...")

                cluster_obj = ClusterKCenters(traj, N_CLUSTERS, cluster_atom_indices)
                labels = cluster_obj.assign(traj)
                original_center_indices = cluster_obj.center_indices

                unique, counts = np.unique(labels, return_counts=True)
                valid_clusters = unique[counts >= MIN_CLUSTER_SIZE]
                label_map = {old: new for new, old in enumerate(valid_clusters)}
                filtered_labels = np.array([label_map[l] if l in label_map else -1 for l in labels])
                n_states = len(valid_clusters)

                RT = KB_KJ * KRAMERS_TEMPERATURE
                free_energies_alpha = np.zeros(n_states)
                for i in range(n_states):
                    mask = (filtered_labels == i)
                    if np.sum(mask) > 0:
                        cluster_energies = energies[mask]
                        e_min = cluster_energies.min()
                        boltzmann_sum = np.sum(np.exp(-(cluster_energies - e_min) / RT))
                        free_energies_alpha[i] = e_min - RT * np.log(max(boltzmann_sum, 1e-300))
                free_energies_alpha -= free_energies_alpha.min()

                run_alpha_analysis(
                    traj, n_states, NEIGHBOR_RADIUS_PERCENTILE, GEOMETRY_RMSD_MODE,
                    original_center_indices, valid_clusters, filtered_labels,
                    free_energies_alpha, RT, BASIN_EQ_THRESHOLD, MIN_BASIN_SIZE,
                    MIN_BASINS_PER_COMPONENT, NUM_EIGENMODES,
                    EIGS_SIGMA, EIGS_TOL, EIGS_DENSE_THRESHOLD,
                    SPECTRAL_GAP_THRESHOLD,
                    ULTRAMETRIC_EPSILON, ULTRAMETRIC_DELTA,
                    ALPHA_RANGE, verbose=v
                )

                del traj
                gc.collect()

        if h1_active:
            print("\n" + "=" * 70)
            print(">>> LAUNCHING MODE 2: NULL HYPOTHESIS H1 TEST <<<")
            print("=" * 70)

            dcd_file, pdb_file, energy_file = run_md_simulation(
                PDB_ID, USE_GB99DMS, TEMPERATURE_K,
                MD_STEPS, MD_REPORT_INTERVAL, USE_GPU, MD_SEED,
                STARTING_STRUCTURE_FILE, ADD_HYDROGENS_TO_CUSTOM_START,
                MINIMIZE_STARTING_STRUCTURE,
                MD_TIME_STEP_FS, MD_ROLLBACK_STEPS, MD_MAX_ROLLBACKS,
                MD_TIME_STEP_REDUCTION_FACTOR, MD_MIN_TIME_STEP_FS,
                verbose=v
            )

            if dcd_file is None:
                print("[ERROR] MD failed. H1 mode skipped.")
            else:
                effective_eta = ETA_NEIGHBOR
                if NEIGHBOR_METHOD == 'energy' and ETA_NEIGHBOR_MODE == 'auto':
                    RT_main = KB_KJ * KRAMERS_TEMPERATURE
                    _, _, fe_for_auto, _ = build_msm_and_graph(
                        dcd_file, pdb_file, energy_file,
                        N_CLUSTERS, MIN_CLUSTER_SIZE,
                        KRAMERS_TEMPERATURE, NEIGHBOR_METHOD, 1e-6,
                        NEIGHBOR_RADIUS_PERCENTILE, GEOMETRY_RMSD_MODE, NEIGHBOR_ALPHA,
                        verbose=False
                    )
                    if fe_for_auto is not None:
                        effective_eta = auto_select_eta_neighbor(
                            fe_for_auto, RT_main, MIN_BASIN_SIZE,
                            MIN_BASINS_PER_COMPONENT, BASIN_EQ_THRESHOLD,
                            verbose=v
                        )
                    else:
                        if v:
                            print("  [WARNING] Could not obtain free_energies for auto eta selection.")

                K_sym, adjacency, free_energies, n_micro = build_msm_and_graph(
                    dcd_file, pdb_file, energy_file,
                    N_CLUSTERS, MIN_CLUSTER_SIZE,
                    KRAMERS_TEMPERATURE, NEIGHBOR_METHOD, effective_eta,
                    NEIGHBOR_RADIUS_PERCENTILE, GEOMETRY_RMSD_MODE, NEIGHBOR_ALPHA,
                    verbose=v
                )

                if K_sym is None:
                    print("[ERROR] MSM construction failed. H1 mode skipped.")
                else:
                    results_base = analyze_landscape_components(
                        K_sym, adjacency, free_energies, n_micro,
                        KRAMERS_TEMPERATURE,
                        BASIN_EQ_THRESHOLD, MIN_BASIN_SIZE, MIN_BASINS_PER_COMPONENT,
                        NUM_EIGENMODES, EIGS_SIGMA, EIGS_TOL, EIGS_DENSE_THRESHOLD,
                        SPECTRAL_GAP_THRESHOLD,
                        ULTRAMETRIC_EPSILON, ULTRAMETRIC_DELTA,
                        verbose=v
                    )

                    if results_base is not None and results_base['total_triplets'] > 0:
                        u_nt_real = results_base['u_nontrivial']
                        u_tr_real = results_base['u_trivial']
                        u_non_real = results_base['u_nonultrametric']

                        if v:
                            print(f"\nBASELINE CALCULATION FOR H1:")
                            print(f"  u_nt: {u_nt_real:.2f}%")

                        results_h1 = run_null_h1_energy_shuffle(
                            K_sym, adjacency, free_energies, n_micro,
                            KRAMERS_TEMPERATURE,
                            BASIN_EQ_THRESHOLD, MIN_BASIN_SIZE, MIN_BASINS_PER_COMPONENT,
                            NUM_EIGENMODES, EIGS_SIGMA, EIGS_TOL, EIGS_DENSE_THRESHOLD,
                            SPECTRAL_GAP_THRESHOLD,
                            ULTRAMETRIC_EPSILON, ULTRAMETRIC_DELTA,
                            NULL_H1_NUM_SAMPLES, NULL_SEED_BASE,
                            verbose=v
                        )
                        print_null_h1_results(results_h1, u_nt_real, u_tr_real, u_non_real, verbose=v)
                    else:
                        print("[ERROR] Baseline calculation failed. H1 mode skipped.")

        if main_active:
            print("\n" + "=" * 70)
            print(">>> LAUNCHING MODE 3: MAIN ANALYSIS <<<")
            print("=" * 70)

            results_main = run_main_analysis(
                MAIN_NUM_SAMPLES, NULL_SEED_BASE,
                PDB_ID, USE_GB99DMS, TEMPERATURE_K,
                MD_STEPS, MD_REPORT_INTERVAL, USE_GPU,
                N_CLUSTERS, MIN_CLUSTER_SIZE,
                KRAMERS_TEMPERATURE, NEIGHBOR_METHOD,
                ETA_NEIGHBOR, NEIGHBOR_RADIUS_PERCENTILE,
                GEOMETRY_RMSD_MODE, NEIGHBOR_ALPHA,
                ETA_NEIGHBOR_MODE,
                BASIN_EQ_THRESHOLD, MIN_BASIN_SIZE,
                MIN_BASINS_PER_COMPONENT, NUM_EIGENMODES,
                EIGS_SIGMA, EIGS_TOL, EIGS_DENSE_THRESHOLD,
                SPECTRAL_GAP_THRESHOLD,
                ULTRAMETRIC_EPSILON, ULTRAMETRIC_DELTA,
                STARTING_STRUCTURE_FILE, ADD_HYDROGENS_TO_CUSTOM_START,
                MINIMIZE_STARTING_STRUCTURE,
                MD_TIME_STEP_FS, MD_ROLLBACK_STEPS, MD_MAX_ROLLBACKS,
                MD_TIME_STEP_REDUCTION_FACTOR, MD_MIN_TIME_STEP_FS,
                verbose=v
            )
            print_main_analysis_results(results_main, verbose=v)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"TOTAL EXECUTION TIME: {total_elapsed:.1f} sec ({total_elapsed/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
