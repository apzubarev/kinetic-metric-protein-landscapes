#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ULTRAMETRIC ANALYSIS OF PROTEIN ENERGY LANDSCAPES
BASED ON KINETIC METRICS AND MARKOV STATE MODELS (MSM)
Author: A. P. Zubarev
Version: 1.12 (Fixed architecture: MD is launched only MAIN_NUM_SAMPLES times,
files are saved with suffixes, ALPHA_ANALYSIS uses prepared trajectories.
Table output is unconditional, added alpha indication for MAIN_ANALYSIS)
================================================================================
METHOD DESCRIPTION AND PROGRAM PURPOSE
--------------------------------------------------------------------------------
This program implements a method for testing the hypothesis of ultrametric
organization of protein energy landscapes. The method is based on analyzing
transition kinetics between macrostates (attraction basins) using a kinetic
metric constructed via spectral decomposition of the symmetrized Kramers
transition rate matrix and the Mahalanobis distance.

VERSION 1.12 FEATURES:
- Removed the redundant MAIN_ALPHA_SAMPLES parameter. The number of replicas
  for ALPHA_ANALYSIS is now exactly MAIN_NUM_SAMPLES.
- FUNDAMENTAL ARCHITECTURE FIX: Molecular dynamics (MD) and subsequent
  trajectory clustering are performed STRICTLY MAIN_NUM_SAMPLES times.
  Simulation 1 generates files: {base}_trajectory_1.dcd, {base}_1.pdb,
  {base}_energies_1.txt.
  Simulation 2 generates files: {base}_trajectory_2.dcd, {base}_2.pdb,
  {base}_energies_2.txt.
- ALPHA_ANALYSIS mode no longer starts MD. It takes already generated
  (or loaded from disk) MAIN_NUM_SAMPLES trajectories and applies all alpha
  values from ALPHA_RANGE to each of them. This is physically correct and
  saves hundreds of hours of computation.
- In FROM_FILES_ANALYSIS mode, the program automatically scans the folder
  for files with suffixes (_trajectory_1.dcd, _trajectory_2.dcd, etc.),
  determines the number of available samples, and performs analysis
  (MAIN and ALPHA) for each of them.
- FINAL TABLES ARE NOW ALWAYS PRINTED, even if VERBOSE = False.

The program performs the following computational stages:

1. MOLECULAR DYNAMICS (MD) STAGE
   Generation of an equilibrium conformational ensemble by Langevin dynamics
   using the OpenMM package. Supported implicit-solvent force fields are
   GB99dms and Amber14/OBC2.

   ROBUST STRUCTURE PREPARATION:
   Before MD, the structure is automatically cleaned: non-protein residues
   (ligands, ions, water) are removed, the longest polypeptide chain is
   selected, and missing atoms/hydrogens are added. This ensures
   compatibility with OpenMM force fields for arbitrary PDB entries.

   IN-MEMORY BUFFERING:
   Instead of writing directly to disk via DCDReporter, the program uses a
   custom MemoryReporter that accumulates coordinates and energies in Python
   lists. This solves the fundamental problem of "garbage" frames when NaNs
   occur: lists in RAM can be atomically truncated, whereas binary DCD files
   cannot be rolled back. A single DCD file is written to disk via mdtraj
   ONLY ONCE at the very end, after successful completion of the simulation.

   If the number of rollbacks exceeds the limit (MD_MAX_ROLLBACKS), the
   integration time step is automatically reduced and the simulation is
   restarted.

2. MARKOV STATE MODEL (MSM) CONSTRUCTION STAGE
   The continuous conformational space is discretized by clustering trajectory
   frames into microstates using the k-centers algorithm in RMSD metric.
   The atom selection for RMSD calculation is determined by the parameter
   GEOMETRY_RMSD_MODE:
   - 'ca': only C-alpha atoms (faster, reflects backbone folding).
   - 'all': RMSD over all atoms (stricter, accounts for side chains).

   IMPORTANT: The selected metric is used CONSISTENTLY both for clustering
   and for constructing the neighborhood graph, ensuring internal consistency
   of space discretization.

   For each microstate (cluster), the free energy is computed through the
   partition function over energies of frames inside that cluster.

   The neighborhood graph is built by one of two methods:
   - Energetic: two clusters are neighbors if the difference of their free
     energies does not exceed the threshold dF_max = -RT ln(eta). This
     criterion is invariant to energy shifts and provides correct
     connectivity at all energy levels.
   - Adaptive topological graph: for each cluster, an effective radius is
     computed (the specified percentile of distances from the center to
     cluster points). Two clusters are neighbors if the distance between
     their centers does not exceed alpha * (R_i + R_j), where alpha is
     NEIGHBOR_ALPHA.

   Based on free energies and the neighborhood graph, the symmetrized
   transition rate matrix K_sym is built. The off-diagonal elements K_sym_ij
   are computed by the Kramers formula: exp(-|F_i - F_j| / (2RT)) for
   neighboring microstates, otherwise 0.

   Macrostates (attraction basins) are identified by gradient descent on the
   free energy landscape. Attractors (local minima and plateaus) are
   determined automatically; each microstate is assigned to the basin reached
   by gradient descent. BASIN_EQ_THRESHOLD sets the energy equivalence
   threshold below which states belong to the same plateau.

   IMPORTANT ALGORITHM NOTE:
   1. Attractors are identified first (states that have no neighbors strictly
      lower than F_i - epsilon_eq).
   2. The equivalence graph analysis is performed ONLY among attractors to
      correctly merge degenerate minima/plateaus.
   3. Gradient descent is applied exclusively to unlabeled slope states.

   The analysis is performed strictly by connected components. For each
   component with number of basins >= MIN_BASINS_PER_COMPONENT,
   ultrametricity is computed independently. The final ultrametricity is
   computed as a weighted average. The fragmentation index f_inter is also
   computed.

3. SPECTRAL ANALYSIS AND ULTRAMETRICITY TESTING STAGE
   For each significant connected component, spectral decomposition of
   K_sym_c is performed. Noise modes are filtered by the spectral gap
   criterion. Basin characteristic vectors are projected onto physical
   eigenmodes. The distance between basins is the Mahalanobis distance:
   a weighted Euclidean norm of projection differences with weights
   1/|lambda_k|. Slower relaxation modes contribute more to the distance.

   Ultrametricity is tested by enumerating all triplets of basins. A triplet
   is considered nontrivially ultrametric if the two largest distances are
   approximately equal (relative difference <= epsilon), and the third one
   is significantly smaller (relative difference > delta). The fraction of
   such triplets is u_nt.

4. EXECUTION ARCHITECTURE AND MODES
   The program operates in one or more of the following modes:
   - FROM_FILES_ANALYSIS: Skip MD, load existing DCD/CIF/PDB files.
     The program automatically finds all samples (_1, _2, ...), clusters
     them, and builds MSMs. Then the enabled analytical modes are run.
   - SENSITIVITY_ANALYSIS: Run analysis for several BASIN_EQ_THRESHOLD
     values specified in SENSITIVITY_RANGE. Cannot be combined with
     ALPHA_ANALYSIS.
   - ALPHA_ANALYSIS: Run analysis for several NEIGHBOR_ALPHA values
     specified in ALPHA_RANGE. Cannot be combined with
     SENSITIVITY_ANALYSIS. The number of replicas for each alpha is exactly
     equal to the number of loaded or generated trajectories
     (MAIN_NUM_SAMPLES).
   - MAIN_ANALYSIS: Analysis with base parameters for all generated or
     loaded trajectories. Mean, standard deviation, and CV are computed to
     assess stability.

TECHNICAL REQUIREMENTS
--------------------------------------------------------------------------------
Execution environment: Windows 10/11, Linux, macOS. Python 3.10-3.12.
Required packages (pip install):
- openmm, pdbfixer, mdtraj, numpy, scipy, threadpoolctl, torch

HARDWARE REQUIREMENTS
--------------------------------------------------------------------------------
An NVIDIA CUDA GPU is recommended for MD. Automatic fallback to CPU.
MSM and spectral analysis stages are performed on CPU.

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
import re
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
# GROUP 1: WORK MODE SELECTION AND GENERAL SETTINGS
# ----------------------------------------------------------------------------

MAX_CPU_THREADS = 6
"""
Maximum number of CPU threads used for computations (OpenMP, BLAS, LAPACK).
Set this value lower than the total number of logical cores to leave
resources for the operating system and background tasks, avoiding system
overload. If set to 0 or None, the program automatically uses all available
logical cores (multiprocessing.cpu_count()).
Recommended value: 22 (for a 24-thread CPU) or 0 (auto).
"""

FROM_FILES_ANALYSIS = True
"""
Enable analysis-from-files mode (skip MD stage).
True  - The program skips the expensive molecular dynamics stage and loads
        ready files from disk:
        - {PDB_ID}_trajectory_N.dcd
        - {PDB_ID}_N.pdb (or .cif)
        - {PDB_ID}_energies_N.txt
        The program automatically scans the folder, finds all samples
        (N=1, 2, ...), clusters them, and performs analysis for each.
False - The program performs the full cycle starting from MD, generating
        MAIN_NUM_SAMPLES trajectories and saving them to disk with suffixes
        _1, _2, ...
Recommended value: False (for primary calculation),
True (for repeated analysis of ready trajectories).
"""

SENSITIVITY_ANALYSIS = False
"""
Enable sensitivity analysis of results to the energy equivalence threshold
BASIN_EQ_THRESHOLD.
If True, the program performs a computation loop for the set of EQ_THRESHOLD
values specified by SENSITIVITY_RANGE and prints a summary table.
This allows assessing the stability of the landscape hierarchical structure
with respect to this parameter without a separate preliminary "main run".

Logic details:
- An empty SENSITIVITY_RANGE = [] is equivalent to disabling the mode.
- A single-element list (e.g. [3.0]) is equivalent to a single main run
  from previous program versions.
- A non-empty list starts the computation loop and builds a table.

WARNING: Cannot be True at the same time as ALPHA_ANALYSIS.
Recommended value: True (for studying landscape stability).
"""

SENSITIVITY_RANGE = [1.0, 3.0, 5.0, 7.0]
"""
List of BASIN_EQ_THRESHOLD values (in kJ/mol) for sensitivity analysis.
Used ONLY when SENSITIVITY_ANALYSIS = True.
Format: explicit list of values, e.g. [1.0, 3.0, 5.0, 7.0].
The program will sequentially perform analysis for each value and print
a summary table.

Logic details:
- Empty list [] is equivalent to disabling the mode.
- Single-element list is equivalent to a single run.
- Non-empty list starts the computation loop and builds a table.
Recommended value: [2.0, 2.5, 3.0] (three values around base 2.5 kJ/mol).
For broader analysis: [1.0, 3.0, 5.0, 7.0, 10.0].
"""

ALPHA_ANALYSIS = True
"""
Enable sensitivity analysis to the sphere overlap parameter NEIGHBOR_ALPHA.
If True, the program performs a computation loop for the set of ALPHA_RANGE
values and prints a summary table.
This allows assessing the stability of the landscape hierarchical structure
with respect to the topological parameter without rerunning MD.

Logic details:
- Empty ALPHA_RANGE = [] is equivalent to disabling the mode.
- Single-element list (e.g. [0.6]) is equivalent to a single main run.
- Non-empty list starts the computation loop and builds a table.
- The number of independent replicas for each alpha is exactly equal to the
  number of generated or loaded trajectories (MAIN_NUM_SAMPLES).

WARNING: Cannot be True at the same time as SENSITIVITY_ANALYSIS.
Recommended value: False (or True for studying graph topology influence).
"""

ALPHA_RANGE = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
"""
List of NEIGHBOR_ALPHA values for sensitivity analysis.
Used ONLY when ALPHA_ANALYSIS = True and NEIGHBOR_METHOD =
'adaptive_topology'.
Format: explicit list of values, e.g. [0.5, 0.6, 0.7, 0.8, 1.0].
The program will sequentially perform analysis for each value and print
a summary table.

Logic details:
- Empty list [] is equivalent to disabling the mode.
- Single-element list is equivalent to a single run.
- Non-empty list starts the computation loop and builds a table.
Recommended value: [0.5, 0.6, 0.7, 0.8, 1.0].
"""

MAIN_ANALYSIS = False
"""
Enable Main Analysis mode.
True  - No preliminary main run is performed. The full computation cycle
        (MD -> clustering -> MSM -> analysis) is performed MAIN_NUM_SAMPLES
        times with different random seeds. For all runs, mean values,
        standard deviations, and the coefficient of variation of u_nt are
        computed. A final table and a detailed stability report are produced.
False - Main analysis mode is skipped.

Logic: This mode is the MAIN TOOL for obtaining statistically significant
results. The full computation volume is immediately distributed over
MAIN_NUM_SAMPLES independent runs.

MD errors (including NaN coordinates) are handled automatically: failed runs
are skipped, and the program continues.

Computational cost: about 10 minutes per run on GPU. With 10 runs the whole
test takes about 1.5-2 hours. Recommended for final calculations and
publication-quality results.

SPECIAL NOTE FOR FROM_FILES_ANALYSIS:
If FROM_FILES_ANALYSIS = True, the program automatically determines
MAIN_NUM_SAMPLES from the number of available files on disk.
Recommended value: True (for final calculations), False (for quick tests).
"""

MAIN_NUM_SAMPLES = 5
"""
Number of independent runs for Main Analysis (MAIN_ANALYSIS) and
ALPHA sensitivity analysis (ALPHA_ANALYSIS).
Determines the number of full program restarts with different seeds.
Each run includes the whole chain: MD -> clustering -> MSM -> analysis.
Mean, standard deviation, and a table are computed from the results.

SPECIAL NOTE: In FROM_FILES_ANALYSIS = True mode this parameter is ignored,
because the number of samples is determined automatically from files on disk.
Recommended value: 5-10 (compromise between statistical reliability and
computation time). For high-precision standard deviations, increase to 20-30.
Allowed range: 1 - 100. (With value 1 the mode is equivalent to a single
run.)
"""

NULL_SEED_BASE = 12345
"""
Base seed for random number generation in Main Analysis (MAIN_ANALYSIS).
Each replica uses seed = NULL_SEED_BASE + replica_index, ensuring
reproducibility of results when rerunning.
Recommended value: 12345 (any integer).
"""

USE_GPU = True
"""
Use GPU (CUDA) for molecular dynamics.
True  - The program tries to use an NVIDIA GPU. If GPU is unavailable
        (no driver, CUDA toolkit not installed) or a JIT compilation error
        occurs (e.g. missing nvrtc DLL), the program AUTOMATICALLY AND
        SEAMLESSLY switches to CPU with a warning.
False - The program guaranteed uses CPU only at all stages.
        Useful for machines without NVIDIA GPU or when CUDA errors occur.

Performance impact:
GPU: about 2-3 minutes for 500,000 MD steps.
CPU: about 20-30 minutes for 500,000 MD steps.
Recommended value: True (if GPU is available).
"""

VERBOSE = False
"""
Detailed console output.
True  - Detailed information about all stages is printed: intermediate MD
        results, clustering statistics, neighborhood graph properties, list
        of connected components, progress of null models.
False - Only key results are printed (final tables). Useful for batch runs
        or redirecting output to a file.
Recommended value: True.
"""

N_COMP_CON = 10
"""
Number of largest connected components whose information is printed in the
console table.
Recommended value: 10.
"""

# ----------------------------------------------------------------------------
# GROUP 2: MOLECULAR DYNAMICS (MD) PARAMETERS
# ----------------------------------------------------------------------------

PDB_ID = "1L2Y"
"""
Protein identifier in the RCSB Protein Data Bank.
Defines the initial structure for simulation if a custom starting structure
file (STARTING_STRUCTURE_FILE) is not specified.
The structure is automatically downloaded from RCSB PDB, cleaned from
foreign molecules (water, ligands), supplemented with missing atoms and
hydrogens at pH 7.0. Local energy minimization is performed before dynamics.

IN FROM_FILES_ANALYSIS MODE:
PDB_ID is used as the base name for searching files on disk:
{PDB_ID}_trajectory.dcd, {PDB_ID}.cif (or .pdb), {PDB_ID}_energies.txt.

Recommended test values:
"1L2Y" - Trp-cage (20 residues, fast folding)
"1UBQ" - ubiquitin (76 residues)
"1JSF" - protein with ligands (tests robust structure cleaning)
"2RVD" - short peptide (10 residues)
Allowed values: any valid four-character PDB code.
"""

STARTING_STRUCTURE_FILE = ""
"""
Path to a custom starting structure file (PDB or CIF format).
If left empty (""), the program automatically downloads the native structure
from RCSB PDB using PDB_ID.
Using an alternative starting structure (e.g. from another simulation or a
homology model) allows:
- avoiding artificial trapping in the deep native minimum;
- starting simulation from an unfolded or partially folded state;
- exploring the landscape near a conformation of interest.

If the file contains only heavy atoms, set ADD_HYDROGENS_TO_CUSTOM_START =
True.

IN FROM_FILES_ANALYSIS MODE:
If STARTING_STRUCTURE_FILE is not empty, its base name (without extension)
is used to search for the DCD file and energy file.
For example, for "myprotein.cif" the program will search:
- myprotein_trajectory.dcd
- myprotein_energies.txt (optional)
"""

ADD_HYDROGENS_TO_CUSTOM_START = False
"""
Whether to add hydrogen atoms to a custom starting structure.
False - used if the PDB/CIF file already contains hydrogens (e.g. prepared
        with pdbfixer or taken from a simulation with hydrogens).
True  - used if the file contains only heavy atoms; hydrogens will be added
        automatically according to the selected force field.
Recommended value: False.
"""

USE_GB99DMS = True
"""
Force field selection for molecular dynamics.
True  - Use GB99dms force field with implicit solvent.
        Optimized for reproducing properties of both ordered and disordered
        proteins. The GB99dms.xml file is automatically downloaded from the
        official repository on first use. Contains Debye-Hueckel screening
        to account for ionic strength.
False - Use standard Amber14-all force field with OBC2 implicit solvent
        model (obc2.xml). A more conservative choice, well tested in the
        literature.

Impact: The choice of force field determines the relief of the energy
landscape and, consequently, the set of visited conformations, minimum
depths, and relaxation times. GB99dms gives more realistic ensembles for
disordered proteins; Amber14/OBC2 may be preferable for compact globular
proteins.

IN FROM_FILES_ANALYSIS MODE:
This parameter is also used for automatic energy recalculation from a DCD
file if the *_energies.txt file is missing. The force field choice affects
computed energies and, therefore, cluster free energies and all subsequent
analysis results.
Recommended value: True (for most tasks).
"""

TEMPERATURE_K = 300.0
"""
Langevin dynamics temperature in kelvin.
Defines the thermostat temperature in the Langevin equation. Physically it
sets the scale of thermal fluctuations and viscous friction, i.e. the
effective temperature at which the canonical conformational ensemble is
generated.
At low temperatures (~300 K) the system may get stuck in the native basin
and not explore other landscape regions. At high temperatures (~500-1000 K)
the system overcomes energy barriers and visits a broad spectrum of
conformations, including partially and fully unfolded states. The analysis
temperature (KRAMERS_TEMPERATURE) can be chosen independently and set to
physiological (300 K).

WARNING: At temperatures above 1200 K, classical force fields may lose
physical adequacy (covalent bond breaking is possible). The program will
automatically reduce the integration step to prevent numerical instability
(NaN), but results at T > 1500 K should be interpreted with caution.

Recommended values:
300 K    - study of the native basin and nearby minima.
400-500 K - extended sampling of local minima.
800-1000 K - global landscape exploration, including unfolded states
             (recommended for ultrametricity analysis).
Allowed range: 200.0 - 2000.0 K.
"""

MD_STEPS = 10000000
"""
Total number of Langevin dynamics integration steps.
The actual integration step (in fs) is chosen automatically depending on
temperature to ensure numerical stability (2 fs for T < 1000 K, 1 fs for
1000 <= T < 1500 K, 0.5 fs for T >= 1500 K). Total simulation time scales
accordingly.

The number of saved frames is MD_STEPS / MD_REPORT_INTERVAL. For a
statistically meaningful ensemble (~10,000 frames) with standard saving
interval 50, about 500,000 steps are required.

Recommended values:
500,000 - minimum length for obtaining ~10,000 frames.
2,000,000 - recommended for increased statistical reliability.
Allowed range: 10,000 - 100,000,000.
"""

MD_REPORT_INTERVAL = 100
"""
Trajectory frame saving interval (in integration steps).
Every MD_REPORT_INTERVAL-th frame is written to the DCD file, and its
potential energy to the energy text file. Too frequent interval leads to
excessive data volume and correlated frames; too rare interval loses
information about fast transitions.
Recommended value: 50 (provides ~10,000 frames for 500,000 steps).
Allowed range: 10 - 10,000.
"""

MINIMIZE_STARTING_STRUCTURE = True
"""
Whether to perform local energy minimization of the starting structure
before starting molecular dynamics.
Minimization removes steric clashes and reduces initial potential energy.
This prevents simulation "explosion" at the first steps due to unrealistically
high forces.
Strongly recommended True. Disabling (False) makes sense only if the
starting structure has already been minimized or if it is required to start
dynamics from a non-equilibrium configuration.
Recommended value: True.
"""

MD_SEED = 42
"""
Seed for the pseudo-random number generator for Langevin dynamics.
Used to initialize the random number generator in the Langevin integrator
and to set initial velocities. Ensures trajectory reproducibility when
rerunning with the same parameters.
To obtain different trajectories with otherwise equal parameters, change
this value. In MAIN_ANALYSIS mode this seed is ignored because
NULL_SEED_BASE is used.
Recommended value: 42 (any integer).
"""

MD_TIME_STEP_FS = 2.0
"""
Initial Langevin dynamics integration step in femtoseconds.
Defines the base time step for numerical integration of equations of motion.
Value 2.0 fs is standard for systems with bond length constraints (e.g.
LINCS or SHAKE for bonds involving hydrogen).
If numerical instability (NaN) occurs and the rollback limit is exceeded,
this step will be automatically reduced.
Recommended value: 2.0.
Allowed range: 0.5 - 5.0.
"""

MD_ROLLBACK_STEPS = 5000
"""
Number of steps to roll back when numerical instability (NaN) occurs.
The simulation is split into chunks of this size. After successful chunk
completion, system coordinates and velocities are saved in RAM. If a NaN
error occurs during integration, the program restores the saved state,
recreates the simulation object (to clear CUDA context), and continues the
simulation, trying to overcome transient force fluctuations.
Recommended value: 1000 (or a multiple of the frame saving interval).
Allowed range: 100 - 10000.
"""

MD_MAX_ROLLBACKS = 5
"""
Maximum number of CONSECUTIVE allowed rollbacks without reducing the
integration step.
If the number of consecutive rollbacks exceeds this value, the program
concludes systematic instability at the current step, reduces the integration
step by MD_TIME_STEP_REDUCTION_FACTOR times, and starts the simulation from
scratch (from zero, after minimization).
Value 0 means rollbacks are not used, and at the first NaN the step is
reduced and the simulation restarts (as in previous versions).
Recommended value: 5.
Allowed range: 0 - 100.
"""

MD_TIME_STEP_REDUCTION_FACTOR = 2.0
"""
Integration step reduction factor when the rollback limit is exceeded.
If the number of rollbacks exceeds MD_MAX_ROLLBACKS, the current step is
divided by this number and the simulation starts over.
Recommended value: 2.0.
Allowed range: 1.1 - 10.0.
"""

MD_MIN_TIME_STEP_FS = 0.25
"""
Minimum allowed integration step in femtoseconds.
If, after successive reductions, the step falls below this value, the
simulation is considered physically unstable and is aborted with an error.
Recommended value: 0.25.
"""

# ----------------------------------------------------------------------------
# GROUP 3: MSM CONSTRUCTION AND NEIGHBORHOOD GRAPH PARAMETERS
# ----------------------------------------------------------------------------

KRAMERS_TEMPERATURE = 300.0
"""
Temperature in the Kramers formula (in kelvin).
Used ONLY at the transition rate matrix construction stage.
Defines the scale of exponential factors:
K_sym_ij ~ exp(-|F_i - F_j| / (2RT))
Physically it sets the temperature at which transition rates between
microstates are estimated. It may differ from the simulation temperature
(TEMPERATURE_K). This allows, for example, sampling the landscape at high
temperature (1000 K) while estimating barriers at physiological (300 K).
Recommended value: 300.0 K (physiological temperature).
Allowed range: 100.0 - 500.0 K.
"""

NEIGHBOR_METHOD = 'adaptive_topology'
"""
Method for defining microstate (cluster) neighborhood.
'energy'            - Energetic neighborhood. Two microstates are considered
                      neighbors if the difference of their free energies does
                      not exceed the threshold dF_max = -RT ln(eta).
                      The criterion is invariant to energy shift and provides
                      correct connectivity at all levels.
                      Requires ETA_NEIGHBOR_MODE and (in manual mode)
                      ETA_NEIGHBOR parameters.
'adaptive_topology' - Adaptive topological graph. For each cluster, an
                      effective radius is computed (the
                      NEIGHBOR_RADIUS_PERCENTILE-th percentile of distances
                      from the center to cluster points). Two clusters are
                      neighbors if the distance between their centers does
                      not exceed alpha * (R_i + R_j), where alpha is set by
                      NEIGHBOR_ALPHA.
                      The method is objective and does not require a manual
                      distance threshold.
Recommended value: 'adaptive_topology'.
"""

ETA_NEIGHBOR_MODE = 'manual'
"""
Mode for selecting the eta parameter for energetic neighborhood.
Used ONLY when NEIGHBOR_METHOD = 'energy'.
'manual' - Uses ETA_NEIGHBOR value specified by the user.
           Suitable when there is physical justification for a specific
           threshold of inter-cluster transition rate.
'auto'   - Automatic eta selection based on graph topology analysis.
           The algorithm scans eta values from a logarithmic grid (30 points
           in range [1e-12, 1e-2]) and selects the largest eta (strictest
           dF_max threshold) satisfying:
           (1) at least one significant connected component exists
               (C_sig >= 1);
           (2) fragmentation index f_inter <= 0.9.
           In case of ties, the variant with the maximum number of basins in
           the largest component (K_max) is chosen.
           This mode makes eta fully data-driven and removes the need for
           heuristic selection.
Recommended value: 'auto' (for objective analysis).
'manual' (for comparison with a fixed threshold).
"""

ETA_NEIGHBOR = 1e-6
"""
Dimensionless threshold for inter-cluster transition rate for energetic
neighborhood (manual mode).
Used ONLY when NEIGHBOR_METHOD = 'energy' AND ETA_NEIGHBOR_MODE = 'manual'.
Two microstates i and j are considered neighbors if:
|F_i - F_j| <= dF_max, where dF_max = -RT ln(ETA_NEIGHBOR)

Physical interpretation: eta sets the minimum significant transition rate
between microstates (in units of the frequency factor nu_0). The threshold
dF_max is computed automatically from eta and the Kramers temperature T.

Example dF_max values at T = 300 K (RT ~= 2.49 kJ/mol):
eta = 1e-6  ->  dF_max ~= 34.5 kJ/mol (~= 13.8 k_B T)
eta = 1e-4  ->  dF_max ~= 23.0 kJ/mol (~= 9.2 k_B T)
eta = 1e-8  ->  dF_max ~= 46.0 kJ/mol (~= 18.4 k_B T)

WARNING: In versions < 2.8, the criterion max(F_i, F_j) <= -RT ln(eta) was
used, which depended on the absolute energy value. The current criterion
|F_i - F_j| <= dF_max is physically correct and invariant.
Recommended value: 1e-6 (corresponds to ~14 k_B T at 300 K).
Allowed range: 1e-12 - 1e-3.
"""

NEIGHBOR_RADIUS_PERCENTILE = 95
"""
Percentile for computing the effective cluster radius (in percent).
Used ONLY when NEIGHBOR_METHOD = 'adaptive_topology'.
The effective cluster radius is defined as the specified percentile of
distances from the cluster center to all points assigned to that cluster.
Value 95 means that 95% of cluster points lie inside a sphere of this
radius. This cuts off single outliers (the 5% farthest points) and correctly
estimates the "sphere of influence" of a microstate in configuration space.
Recommended value: 95.
Allowed range: 50 - 99.
"""

NEIGHBOR_ALPHA = 0.6
"""
Scaling coefficient for the neighborhood criterion in the adaptive
topological graph.
Used ONLY when NEIGHBOR_METHOD = 'adaptive_topology'.
The neighborhood condition for two clusters i and j is:
d(c_i, c_j) <= alpha * (R_i + R_j)
where d is the distance between cluster centers, R_i and R_j are their
effective radii, alpha is this parameter.

Physical interpretation:
alpha = 1.0 - Standard sphere-touching criterion (analogous to Delaunay
              triangulation for finite-size objects). Clusters are neighbors
              if their "spheres of influence" touch or overlap.
              Recommended default value.
alpha > 1.0 - Graph becomes denser. A gap between influence spheres is
              allowed, proportional to alpha. Useful for merging weakly
              connected components or for insufficient sampling when radii
              may be underestimated. Typical values: 1.1 - 1.5.
alpha < 1.0 - Graph becomes sparser. Overlap of influence spheres is
              required. Useful for removing false connections between distant
              clusters that formally touch due to outliers in point
              distribution. Typical values: 0.8 - 0.95.

Impact on result: Increasing alpha increases the number of edges in the
neighborhood graph, which may merge connected components and increase the
number of basins in a component. Decreasing alpha breaks weak connections,
which may increase fragmentation index f_inter.
Recommended value: 1.0.
Allowed range: 0.5 - 2.0.
"""

GEOMETRY_RMSD_MODE = 'all'
"""
Unified RMSD mode for ALL geometric operations.
Defines the metric for both k-centers clustering and adaptive topological
graph construction. This ensures internal consistency of conformational
space discretization.
'ca'  - RMSD over C-alpha atoms only. Faster, recommended for most tasks.
        Reflects overall chain folding well.
'all' - Full-atom RMSD (all protein atoms). Stricter proximity definition,
        accounting for side-chain conformations, but computationally more
        expensive.
WARNING: When choosing 'all', clustering and graph construction time
increases. For long trajectories (>20,000 frames), 'ca' is recommended.
Recommended value: 'all'.
"""

N_CLUSTERS = 5000
"""
Target number of microstates (clusters) for conformational space
discretization.
Detailed resolution of the energy landscape requires a large number of
microstates. The actual number of significant clusters may be less than
N_CLUSTERS, because clusters with fewer frames than MIN_CLUSTER_SIZE are
discarded.
Recommended value: 1000 (for a trajectory of ~10,000 frames).
For trajectories of ~40,000 frames, 4000 is recommended.
Allowed range: 100 - 10,000.
"""

MIN_CLUSTER_SIZE = 10
"""
Minimum number of frames in a cluster to be considered a significant
microstate.
Smaller clusters are considered statistically insignificant (noise) and are
excluded from further analysis.
Recommended value: 5 (with 10,000 frames and N_CLUSTERS = 1000, the
expected average is 10 frames per cluster; threshold 5 cuts off clusters
with insufficient statistics).
Allowed range: 2 - 100.
"""

BASIN_EQ_THRESHOLD = 1.0
"""
Energy equivalence threshold (epsilon_eq) for determining plateaus and
local minima (in kJ/mol).
When determining attraction points (basins) by gradient descent:
- If the free energy difference between two neighboring microstates
  |F_i - F_j| <= epsilon_eq, they are considered energetically equivalent
  and may belong to the same plateau.
- If F_j < F_i - epsilon_eq, microstate j is considered strictly deeper,
  and gradient descent goes there.

Value ~3.0 kJ/mol corresponds approximately to 1 k_B T at 300 K, allowing
merging of degenerate minima whose energy difference does not exceed thermal
fluctuations.
Recommended value: 3.0 kJ/mol.
Allowed range: 0.1 - 20.0 kJ/mol.
"""

MIN_BASIN_SIZE = 20
"""
Minimum number of microstates in an attraction basin.
Smaller basins are discarded as statistically insignificant.
This prevents the appearance of "basins" of one or two microstates, which
may be artifacts of clustering or insufficient sampling.
Recommended value: 5.
Allowed range: 1 - 20.
"""

MIN_BASINS_PER_COMPONENT = 10
"""
Minimum number of attraction basins in a connected component of the
microstate graph.
Connected components in which, after gradient descent and filtering, fewer
basins remain than this value are discarded as statistically insignificant.
Ultrametricity analysis requires at least three basins to compute at least
one triplet, but for statistically reliable conclusions at least 10 basins
(120 triplets) are needed.
Recommended value: 10.
Allowed range: 3 - 100.
"""

# ----------------------------------------------------------------------------
# GROUP 4: SPECTRAL ANALYSIS AND ULTRAMETRICITY TESTING PARAMETERS
# ----------------------------------------------------------------------------

NUM_EIGENMODES = 50
"""
Maximum number of eigenmodes for spectral decomposition of K_sym.
After automatic filtering of noise modes by the spectral gap criterion,
n_phys physical modes remain from this number and are used to compute the
kinetic metric.
Recommended value: 50.
Allowed range: 10 - 500.
"""

EIGS_SIGMA = 1e-4
"""
Sigma shift for the Lanczos algorithm (shift-invert) when searching for
eigenvalues of K_sym near zero.
The K_sym matrix is negative semidefinite (all lambda <= 0). The
shift-invert algorithm (which='LM', sigma > 0) allows efficiently finding
eigenvalues closest to sigma, i.e. the smallest nonzero modes - the slowest
relaxation processes, which contribute most to the kinetic metric.

Impact of sigma on performance:
sigma = 1e-4 ... 1e-5 (recommended):
    Strengthens diagonal dominance of the shifted matrix (K - sigma*I),
    LU factorization (SuperLU) is done in fractions of a second.
sigma < 1e-8:
    The shifted matrix is almost singular, factorization may slow down by
    orders of magnitude (tens of seconds) due to numerical instability and
    the need for extensive pivoting.
sigma > 1e-2:
    The algorithm finds modes near sigma, which may not be the slowest;
    the result may lose physical meaning.
Recommended value: 1e-4.
Allowed range: 1e-6 - 1e-3.
"""

EIGS_TOL = 1e-8
"""
Convergence tolerance for the Lanczos algorithm (eigsh).
Defines the accuracy of eigenvalues and eigenvectors.
Smaller value gives more accurate result but requires more iterations.
Recommended value: 1e-8 (sufficient for most tasks).
Allowed range: 1e-10 - 1e-6.
"""

EIGS_DENSE_THRESHOLD = 6000
"""
Matrix size threshold for switching to dense diagonalization.
If the number of microstates in a connected component is below this
threshold, full dense matrix diagonalization (np.linalg.eigh) is used
instead of sparse Lanczos (eigsh). For small matrices dense diagonalization
is faster and more reliable; for large matrices sparse algorithm saves time.
Note: if the sparse algorithm does not converge (ArpackNoConvergence), the
program automatically switches to dense diagonalization regardless of this
threshold.
Recommended value: 6000.
Allowed range: 100 - 10000.
"""

SPECTRAL_GAP_THRESHOLD = 1e6
"""
Threshold for the ratio |lambda_k| / |lambda_{k-1}| to detect a spectral
gap. Eigenvalues are sorted by increasing absolute value. If for some k the
ratio exceeds the threshold, all modes with indices 0..k-1 are considered
numerical noise and excluded. Remaining modes (k..m-1) are considered
physical.
Typical values: 1e4 - 1e8. Threshold 1e6 reliably separates noise modes
(|lambda| ~ 10^-16) from physical (|lambda| ~ 10^-5 - 10^0).
Recommended value: 1e6.
"""

ULTRAMETRIC_EPSILON = 0.05
"""
Relative tolerance epsilon for testing equality of the two largest sides of
a triangle during triplet classification.
A triplet (x, y, z) with ordered distances d_min <= d_mid <= d_max is
considered nontrivially ultrametric if:
(d_max - d_mid) / d_mid <= epsilon AND (d_mid - d_min) / d_mid > delta
Must be strictly less than ULTRAMETRIC_DELTA.
Recommended value: 0.05 (5% relative tolerance).
Allowed range: 0.01 - 0.20.
"""

ULTRAMETRIC_DELTA = 0.1
"""
Minimum relative difference between the middle and smaller sides of a
triangle for classifying the triplet as nontrivially ultrametric.
Must be strictly greater than ULTRAMETRIC_EPSILON.
Recommended value: 0.1 (10% relative difference).
Allowed range: 0.05 - 0.50.
"""

# ============================================================================
# BLAS/LAPACK/OPENMP THREAD LIMITING (OPTIMIZED FOR STAGE 2 ACCELERATION)
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

    non_protein_residues = []
    for residue in modeller.topology.residues():
        if residue.name not in ALL_AMINO_ACIDS:
            non_protein_residues.append(residue)

    if non_protein_residues:
        if verbose:
            print(f"  [Modeller] Removing {len(non_protein_residues)} non-protein residues (ligands, water, etc.)...")
        modeller.delete(non_protein_residues)

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
                print(f"  [Modeller] Selecting chain '{longest_chain_id}'. Removing {len(residues_to_delete)} residues from other chains...")
            modeller.delete(residues_to_delete)

    if verbose:
        print("  [Modeller] Adding hydrogens according to force field...")

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
        print(f"  Final structure: {topology.getNumAtoms()} atoms, {sum(1 for _ in topology.residues())} residues")

    return topology, positions, ff


# ============================================================================
# LOAD TRAJECTORY AND ENERGIES FROM FILES (FOR FROM_FILES_ANALYSIS MODE)
# ============================================================================
def load_trajectory_and_energies_from_files(pdb_id, starting_structure_file,
                                            use_gb99dms, use_gpu,
                                            kramers_temperature, sample_idx=None, verbose=True):
    from openmm import app
    import mdtraj as md

    if verbose:
        print("\n" + "=" * 70)
        print("LOADING DATA FROM FILES (FROM_FILES_ANALYSIS mode)")
        print("=" * 70)

    base_name = os.path.splitext(os.path.basename(starting_structure_file))[0] if starting_structure_file and os.path.exists(starting_structure_file) else pdb_id
    suffix = f"_{sample_idx}" if sample_idx is not None else ""

    dcd_file = f"{base_name}_trajectory{suffix}.dcd"
    energy_file = f"{base_name}_energies{suffix}.txt"

    topology_file_for_mdtraj = None
    for ext in [".pdb", ".cif", "_topology.pdb", ""]:
        cand = f"{base_name}{suffix}{ext}"
        if os.path.exists(cand):
            topology_file_for_mdtraj = cand
            break

    if not topology_file_for_mdtraj:
        for ext in [".pdb", ".cif"]:
            cand = f"{base_name}{ext}"
            if os.path.exists(cand):
                topology_file_for_mdtraj = cand
                break

    if not os.path.exists(dcd_file):
        if verbose:
            print(f"\n[ERROR] Trajectory file not found: {dcd_file}")
            print("  FROM_FILES_ANALYSIS mode requires a DCD file.")
        return None, None, None

    if topology_file_for_mdtraj is None:
        if verbose:
            print(f"\n[ERROR] Topology file not found for base name: {base_name}{suffix}")
        return None, None, None

    if verbose:
        print(f"  Trajectory found: {dcd_file}")
        print(f"  Topology file for mdtraj: {topology_file_for_mdtraj}")

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
                    print(f"\n[WARNING] Mismatch: trajectory has {n_frames} frames, energy file has {len(energies)} values.")
                    print("  Energy file will be recalculated.")
                energy_file_exists = False
            else:
                if verbose:
                    print(f"  Energy file loaded: {energy_file}")
                    print(f"  Energies: min={energies.min():.1f}, max={energies.max():.1f}, mean={energies.mean():.1f} kJ/mol")
        except Exception as e:
            if verbose:
                print(f"\n[WARNING] Failed to read energy file: {type(e).__name__}")
                print("  Energy file will be recalculated.")
            energy_file_exists = False

    if not energy_file_exists:
        if verbose:
            print("\nRecalculating energies through OpenMM (this may take ~30-60 sec)...")
            print(f"  Force field: {'GB99dms' if use_gb99dms else 'Amber14/OBC2'}")

        try:
            topology_omm, _, ff = prepare_structure(topology_file_for_mdtraj, use_gb99dms, verbose)
        except ImportError:
            raise ImportError("pdbfixer is required: pip install pdbfixer")

        if topology_omm.getNumAtoms() != traj.n_atoms:
            if verbose:
                print(f"\n[ERROR] Atom count in cleaned structure ({topology_omm.getNumAtoms()}) does not match trajectory atom count ({traj.n_atoms}).")
                print("  Ensure the DCD file was created by this program, or provide a matching *_topology.pdb file.")
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
            positions = [openmm.Vec3(x, y, z) * unit.nanometer for x, y, z in coords_nm]
            context.setPositions(positions)
            state = context.getState(getEnergy=True)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            energies_list.append(energy)

            if verbose and (frame_idx + 1) % max(1, n_frames // 10) == 0:
                elapsed = time.time() - t0
                print(f"    Processed {frame_idx + 1}/{n_frames} frames ({elapsed:.1f} sec)")

        del context, integrator
        gc.collect()

        energies = np.array(energies_list)
        np.savetxt(energy_file, energies, fmt='%.6f')

        if verbose:
            elapsed = time.time() - t0
            print(f"  Energies recalculated in {elapsed:.1f} sec")
            print(f"  Saved: {energy_file}")
            print(f"  Energies: min={energies.min():.1f}, max={energies.max():.1f}, mean={energies.mean():.1f} kJ/mol")

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
            n_micro_c = len(comp_micro_indices)
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
            print("    [WARNING] No eta satisfies the criteria.")
            print(f"    Using smallest eta = {fallback_eta:.2e}")
            print(f"    (dF_max = {df_max_fallback:.1f} kJ/mol)")
        return fallback_eta

    candidates.sort(key=lambda x: (-x[0], -x[2]))
    best = candidates[0]
    best_eta = best[0]
    df_max_best = -RT * np.log(best_eta)

    if verbose:
        print(f"    Selected eta = {best_eta:.2e}")
        print(f"    dF_max = {df_max_best:.1f} kJ/mol ({df_max_best / RT:.1f} RT)")
        print(f"    C_sig = {best[1]}, K_max = {best[2]}, f_inter = {best[3]:.4f}")
        print(f"    Total suitable candidates: {len(candidates)} of {len(eta_grid)}")

    return best_eta


# ============================================================================
# STAGE 1: MOLECULAR DYNAMICS
# ============================================================================
def run_md_simulation(pdb_id, use_gb99dms, temp_k, n_steps, report_interval,
                      use_gpu, seed, starting_structure_file,
                      add_hydrogens_to_custom_start, minimize_starting_structure,
                      md_time_step_fs, md_rollback_steps, md_max_rollbacks,
                      md_time_step_reduction_factor, md_min_time_step_fs,
                      sample_idx=None, verbose=True):
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
                print(f"\nFound local CIF: {local_cif}")
        elif os.path.exists(local_pdb):
            input_file = local_pdb
            if verbose:
                print(f"\nFound local PDB: {local_pdb}")
        else:
            if verbose:
                print(f"\nLocal files not found. Downloading {pdb_id} from RCSB PDB...")

            import urllib.request
            cif_url = f"https://files.rcsb.org/download/{pdb_id}.cif"
            download_path = os.path.abspath(local_cif)

            try:
                urllib.request.urlretrieve(cif_url, download_path)
                input_file = download_path
                if verbose:
                    print(f"  CIF successfully downloaded: {download_path}")
            except Exception:
                if verbose:
                    print("  CIF download failed, trying PDB...")
                pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                download_path = os.path.abspath(local_pdb)
                urllib.request.urlretrieve(pdb_url, download_path)
                input_file = download_path
                if verbose:
                    print(f"  PDB successfully downloaded: {download_path}")

    try:
        topology, positions, ff = prepare_structure(input_file, use_gb99dms, verbose)
    except ImportError:
        raise ImportError("pdbfixer is required: pip install pdbfixer")

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
                print(f"  [WARNING] CUDA initialization error (missing nvrtc DLL or driver): {type(e).__name__}")
                print("             Automatic switch to CPU.")
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

    base_name = os.path.splitext(os.path.basename(starting_structure_file))[0] if starting_structure_file and os.path.exists(starting_structure_file) else pdb_id
    suffix = f"_{sample_idx}" if sample_idx is not None else ""

    final_dcd_file = f"{base_name}_trajectory{suffix}.dcd"
    energy_file = f"{base_name}_energies{suffix}.txt"
    pdb_file = f"{base_name}{suffix}.pdb"

    with open(pdb_file, 'w') as f:
        app.PDBFile.writeFile(topology, positions, f)

    if verbose:
        print(f"  Cleaned topology saved: {pdb_file}")

    for f in [final_dcd_file, energy_file]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

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
            self.energies.append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))

    while True:
        if current_dt_fs < md_min_time_step_fs:
            if verbose:
                print(f"\n[ERROR] Integration step fell below minimum ({md_min_time_step_fs} fs).")
                print("        The system is physically unstable.")
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
                    print("  Energy minimization of starting structure...")
                sim.minimizeEnergy(maxIterations=2000)

            sim.context.setVelocitiesToTemperature(temp_k * unit.kelvin, seed)

            reporter = MemoryReporter(report_interval)
            sim.reporters.append(reporter)

            total_time_ps = n_steps * dt
            n_frames = n_steps // report_interval

            if verbose:
                print(f"  Starting MD: {n_steps} steps ({total_time_ps:.1f} ps, dt {current_dt_fs} fs), will save ~{n_frames} frames")

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

                    if verbose and (total_steps_done % max(1, n_steps // 10) == 0 or total_steps_done == n_steps):
                        state = sim.context.getState(getEnergy=True)
                        pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                        ke = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
                        temp = (2.0 * ke) / (dof * KB_KJ) if dof > 0 else 0.0
                        print(f"{total_steps_done},{pe},{temp}")

                    state = sim.context.getState(getPositions=True, getVelocities=True)
                    pos_state = state.getPositions(asNumpy=True)
                    vel_state = state.getVelocities(asNumpy=True)

                    pos_values = pos_state.value_in_unit(unit.nanometer) if hasattr(pos_state, 'value_in_unit') else pos_state
                    vel_values = vel_state.value_in_unit(unit.nanometer / unit.picosecond) if hasattr(vel_state, 'value_in_unit') else vel_state

                    if np.isnan(pos_values).any() or np.isnan(vel_values).any():
                        if verbose:
                            print(f"\n[ROLLBACK] Hidden NaNs detected in successful step {total_steps_done}. Initiating rollback.")
                        raise ValueError("Hidden NaNs detected in getState")

                    resume_pos = pos_state
                    resume_vel = vel_state
                    frames_saved = len(reporter.positions)
                    consecutive_rollbacks = 0

                except Exception as e:
                    is_nan_error = ("NaN" in str(e) or
                                    isinstance(e, openmm.OpenMMException) or
                                    isinstance(e, ValueError))

                    if is_nan_error:
                        consecutive_rollbacks += 1

                        if consecutive_rollbacks > md_max_rollbacks:
                            if verbose:
                                print(f"\n[WARNING] Consecutive rollback limit ({md_max_rollbacks}) exceeded.")
                                print(f"          Reducing time step by factor {md_time_step_reduction_factor} and restarting.")
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
                            integrator.setRandomNumberSeed(seed + consecutive_rollbacks + int(time.time()) % 1000)

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
                print(f"  MD completed in {elapsed:.1f} sec ({elapsed / 60:.1f} min)")

            if len(reporter.positions) > 0:
                if verbose:
                    print(f"  Writing DCD file to disk ({len(reporter.positions)} frames)...")

                topology_mdtraj = md.load(pdb_file).topology

                positions_np = np.array(
                    [p.value_in_unit(unit.nanometer) if hasattr(p, 'value_in_unit') else p
                     for p in reporter.positions],
                    dtype=np.float32
                )

                traj = md.Trajectory(xyz=positions_np, topology=topology_mdtraj)
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
                print(f"          Message: {str(e)[:150]}")
            try:
                sim.reporters = []
                del sim, integrator
            except Exception:
                pass
            gc.collect()
            return None, None, None


# ============================================================================
# K-CENTERS CLUSTERIZATION
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
# STAGE 2: MSM CONSTRUCTION AND NEIGHBORHOOD GRAPH
# ============================================================================
def build_neighbor_graph_energy(free_energies, RT, eta, verbose=True):
    n_states = len(free_energies)
    delta_f_max = -RT * np.log(eta)

    diff = np.abs(free_energies[:, None] - free_energies[None, :])
    adjacency = diff <= delta_f_max
    np.fill_diagonal(adjacency, False)

    n_edges = np.sum(adjacency) // 2

    if verbose:
        print(f"  Energy neighborhood: dF_max = {delta_f_max:.1f} kJ/mol "
              f"(eta={eta:.0e}, T={RT / KB_KJ:.0f} K)")
        print(f"  Edges in graph: {n_edges} of {n_states * (n_states - 1) // 2} possible")

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
        print("  Adaptive topological graph: computing effective radii")
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
        print(f"  Effective radii: min={radii.min():.4f} nm, max={radii.max():.4f} nm, mean={radii.mean():.4f} nm")
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
        print(f"  Edges in graph: {n_edges} of {n_states * (n_states - 1) // 2} possible")

    return adjacency


def build_msm_and_graph(dcd_file, pdb_file, energy_file, n_clusters,
                        min_cluster_size, temp_k, neighbor_method, eta,
                        radius_percentile, rmsd_mode, neighbor_alpha,
                        verbose=True):
    import mdtraj as md

    if verbose:
        print("\n" + "=" * 70)
        print("STAGE 2: MSM CONSTRUCTION AND NEIGHBORHOOD GRAPH")
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
        print(f"  Energies: min={energies.min():.1f}, max={energies.max():.1f}, mean={energies.mean():.1f} kJ/mol")

    if rmsd_mode == 'ca':
        cluster_atom_indices = [a.index for a in traj.topology.atoms if a.name == 'CA']
        atom_desc = f"C-alpha ({len(cluster_atom_indices)})"
    else:
        cluster_atom_indices = None
        atom_desc = f"all atoms ({traj.n_atoms})"

    if rmsd_mode == 'ca' and len(cluster_atom_indices) < 3:
        raise ValueError(f"Too few C-alpha atoms: {len(cluster_atom_indices)} (need >= 3)")

    if verbose:
        print(f"  k-centers clustering: k={n_clusters}, atoms: {atom_desc}")

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
            print(f"  ERROR: Significant clusters {len(valid_clusters)} (need >= 3).")
        return None, None, None, None

    label_map = {old: new for new, old in enumerate(valid_clusters)}
    filtered_labels = np.array(
        [label_map[l] if l in label_map else -1 for l in labels]
    )

    n_states = len(valid_clusters)

    if verbose:
        print(f"  Significant microstates: {n_states} "
              f"(discarded {len(unique) - n_states} underpopulated)")

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

    diff = np.abs(free_energies[:, None] - free_energies[None, :])
    rates = np.exp(-diff / (2.0 * RT))

    K_sym = np.where(adjacency, rates, 0.0)
    np.fill_diagonal(K_sym, 0.0)

    max_F = np.maximum(free_energies[:, None], free_energies[None, :])
    K_ji = np.exp(-(max_F - free_energies[:, None]) / RT)
    K_ji = np.where(adjacency, K_ji, 0.0)
    np.fill_diagonal(K_ji, 0.0)

    diag_values = -np.sum(K_ji, axis=1)
    np.fill_diagonal(K_sym, diag_values)

    n_nonzero = np.count_nonzero(K_sym) - n_states

    graph = csr_matrix(adjacency)
    n_components, _ = connected_components(csgraph=graph, directed=False, return_labels=True)

    if verbose:
        print(f"  K_sym matrix: {n_states}x{n_states}, "
              f"nonzero off-diagonal elements: {n_nonzero}")
        print(f"  Number of connected components in neighborhood graph: {n_components}")

    del traj
    gc.collect()

    return K_sym, adjacency, free_energies, n_states


# ============================================================================
# CONNECTED COMPONENT AND BASIN ANALYSIS
# ============================================================================
def find_connected_components(adjacency):
    graph = csr_matrix(adjacency)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    components = [np.where(labels == i)[0] for i in range(n_components)]
    components.sort(key=len, reverse=True)
    return components


def identify_basins_gradient_descent(free_energies, adjacency, eq_threshold):
    n_states = len(free_energies)

    diff = free_energies[:, None] - free_energies[None, :]
    strictly_lower_neighbors = adjacency & (diff > eq_threshold)
    n_lower = np.sum(strictly_lower_neighbors, axis=1)

    is_attractor = (n_lower == 0)
    attractor_indices = np.where(is_attractor)[0]
    n_attractors = len(attractor_indices)

    basin_labels = np.full(n_states, -1, dtype=int)
    basin_count = 0
    n_plateaus = 0

    if n_attractors > 0:
        att_adj = adjacency[np.ix_(attractor_indices, attractor_indices)]
        att_energies = free_energies[attractor_indices]

        att_diff = np.abs(att_energies[:, None] - att_energies[None, :])
        eq_adj_att = att_adj & (att_diff <= eq_threshold)
        np.fill_diagonal(eq_adj_att, False)

        eq_graph_att = csr_matrix(eq_adj_att)
        n_plateaus, plateau_labels = connected_components(csgraph=eq_graph_att, directed=False, return_labels=True)

        basin_labels[attractor_indices] = plateau_labels
        basin_count = n_plateaus

    sorted_indices = np.argsort(free_energies)

    for i in sorted_indices:
        if basin_labels[i] != -1:
            continue

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

    n_basins_before_filter = basin_count

    stats = {
        'n_attractors': n_attractors,
        'n_plateaus': n_plateaus,
        'n_basins_before_filter': n_basins_before_filter
    }

    return basin_labels, basin_count, stats


# ============================================================================
# STAGE 3: KINETIC METRIC AND ULTRAMETRICITY TESTING
# ============================================================================
def compute_kinetic_metric_and_ultrametricity(K_sym_c, basin_labels_c,
                                              n_basins_c, free_energies_c, temp_k,
                                              num_modes, eigs_sigma, eigs_tol,
                                              dense_threshold, gap_threshold, eps, delta,
                                              quiet=False):
    n_micro = K_sym_c.shape[0]
    RT = KB_KJ * temp_k

    if n_micro < 2 or n_basins_c < 3:
        return None

    w = np.exp(-free_energies_c / RT)

    k_request = min(num_modes + 1, n_micro - 1)
    if k_request <= 0:
        return None

    eigenvalues = None
    eigenvectors = None
    eigenvalues_scaled = None

    scale_factor = np.max(np.abs(np.diag(K_sym_c)))
    if scale_factor < 1e-100:
        scale_factor = 1.0

    K_sym_scaled = K_sym_c / scale_factor
    sigma_scaled = eigs_sigma / scale_factor

    if n_micro >= dense_threshold:
        try:
            eigenvalues_scaled, eigenvectors = eigsh(
                K_sym_scaled, k=k_request, which='LM',
                sigma=sigma_scaled, tol=eigs_tol
            )
        except ArpackNoConvergence:
            if not quiet:
                print("    [WARNING] ARPACK did not converge. Switching to dense diagonalization.")
            eigenvalues_scaled = None
            eigenvectors = None
        except Exception:
            eigenvalues_scaled = None
            eigenvectors = None

    if eigenvalues_scaled is None or eigenvectors is None:
        evals_full, evecs_full = np.linalg.eigh(K_sym_scaled)
        idx = np.argsort(np.abs(evals_full))[:k_request]
        eigenvalues_scaled = evals_full[idx]
        eigenvectors = evecs_full[:, idx]

    sort_idx = np.argsort(np.abs(eigenvalues_scaled))
    eigenvalues_scaled = eigenvalues_scaled[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    eigenvalues = eigenvalues_scaled * scale_factor
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

    abs_lambda = np.abs(phys_eigenvalues)
    valid_lambda = abs_lambda > 1e-300

    if not np.any(valid_lambda):
        return None

    phys_eigenvalues = phys_eigenvalues[valid_lambda]
    phys_eigenvectors = phys_eigenvectors[:, valid_lambda]
    weights = 1.0 / abs_lambda[valid_lambda]

    n_phys = len(phys_eigenvalues)

    if n_phys == 0:
        return None

    if not quiet:
        print(f"    Physical modes: {n_phys} (noise modes discarded: {num_noise})")

    chi = np.zeros((n_basins_c, n_micro), dtype=np.float64)

    for a in range(n_basins_c):
        mask = (basin_labels_c == a)
        if np.sum(mask) > 0:
            W_a = np.sum(w[mask])
            if W_a > 1e-300:
                chi[a, mask] = np.sqrt(w[mask]) / W_a

    proj = chi @ phys_eigenvectors
    proj_scaled = proj * np.sqrt(weights)

    D = cdist(proj_scaled, proj_scaled, metric='euclidean')

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
        fe_max_c = np.max(fe_c)
        fe_min_c = np.min(fe_c)
        fe_mean_c = np.mean(fe_c)
        fe_std_c = np.std(fe_c)

        fe_c = fe_c - fe_c.min()

        basin_labels_local, n_basins_raw, stats = identify_basins_gradient_descent(
            fe_c, adj_c, eq_threshold
        )

        unique_basins, basin_counts = np.unique(basin_labels_local, return_counts=True)
        valid_basins = unique_basins[basin_counts >= min_basin_size]
        n_basins_c = len(valid_basins)

        status_str = "ACCEPTED" if n_basins_c >= min_basins_per_comp else "REJECTED"

        all_components_stats.append({
            'comp_idx': comp_idx,
            'n_micro_c': n_micro_c,
            'n_edges_c': n_edges_c,
            'n_attractors': stats['n_attractors'],
            'n_plateaus': stats['n_plateaus'],
            'n_basins_before_filter': n_basins_raw,
            'n_basins_after_filter': n_basins_c,
            'fe_max': fe_max_c,
            'fe_min': fe_min_c,
            'fe_mean': fe_mean_c,
            'fe_std': fe_std_c,
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
            'fe_c': fe_c,
            'triplets_c': triplets_c,
            'n_micro_c': n_micro_c,
            'n_edges_c': n_edges_c,
            'comp_idx': comp_idx,
            'micro_indices': comp_micro_indices
        })

    if verbose:
        print(f"\n{'=' * 145}")
        print("BASIN FORMATION DIAGNOSTICS (first " + str(N_COMP_CON) + " connected components):")
        print(f"{'=' * 145}")
        print(f"{'No.':<4} | {'Micro':<6} | {'Edges':<8} | {'Local min.':<11} | {'Plateaus':<14} | {'Basins':<14} | {'Basins':<15} | {'F_max':<8} | {'F_min':<8} | {'F_mean+/-std':<14} | Status")
        print(f"{'':<4} | {'':<6} | {'':<8} | {'(before merge)':<11} | {'(after merge)':<14} | {'(after descent)':<14} | {'(after filter)':<15} | {'(kJ/mol)':<8} | {'(kJ/mol)':<8} | {'(kJ/mol)':<14} |")
        print("-" * 145)

        limit = min(len(all_components_stats), N_COMP_CON)
        for i in range(limit):
            s = all_components_stats[i]
            fe_str = f"{s['fe_mean']:.1f}+/-{s['fe_std']:.1f}"
            print(f"{s['comp_idx'] + 1:<4} | {s['n_micro_c']:<6} | {s['n_edges_c']:<8} | "
                  f"{s['n_attractors']:<11} | {s['n_plateaus']:<14} | {s['n_basins_before_filter']:<14} | "
                  f"{s['n_basins_after_filter']:<15} | {s['fe_max']:<8.1f} | {s['fe_min']:<8.1f} | {fe_str:<14} | {s['status']}")

        if len(all_components_stats) > limit:
            print(f"... (shown {limit} of {len(all_components_stats)} components)")

        print("-" * 145)

    n_significant_components = len(valid_components_data)

    if total_basins_all >= 3:
        total_triplets_possible = (total_basins_all * (total_basins_all - 1) *
                                   (total_basins_all - 2) // 6)
        f_inter = (1.0 - sum_triplets_all / total_triplets_possible
                   if total_triplets_possible > 0 else 1.0)
    else:
        f_inter = 1.0

    if verbose:
        print("\nSUMMARY BY COMPONENTS:")
        print(f"  Total connected components: {n_components_total}")
        print(f"  Significant components: {n_significant_components}")
        print(f"  Total basins in significant components: {total_basins_all}")

        if n_significant_components > 1:
            print(f"  Fragmentation index f_inter: {f_inter:.4f}")
            if f_inter > 0.8:
                print("  [WARNING] High fragmentation. Results reflect hierarchy within components, not the global landscape.")
        elif n_significant_components == 1:
            print("  Fragmentation index f_inter: 0.0000 (all basins in one component)")
        else:
            print("  Fragmentation index f_inter: not computed (no significant basins)")

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
            comp_data['fe_c'],
            temp_k,
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

    if verbose and n_significant_components > 0:
        print("\nTABLE OF SIGNIFICANT CONNECTED COMPONENTS:")
        header = (f"{'No.':<4} | {'Clusters':<9} | {'Edges':<8} | {'Triplets':<8} | "
                  f"{'u_nt (%)':<9} | {'u_tr (%)':<9} | {'u_non (%)':<9}")
        print(header)
        print("-" * len(header))

        limit = min(n_significant_components, 10)
        for i in range(limit):
            cd = valid_components_data[i]
            u_nt_str = f"{cd.get('u_nt', 0.0):.2f}" if cd.get('u_nt') is not None else "--"
            u_tr_str = f"{cd.get('u_tr', 0.0):.2f}" if cd.get('u_tr') is not None else "--"
            u_non_str = f"{cd.get('u_non', 0.0):.2f}" if cd.get('u_non') is not None else "--"

            print(f"{cd['comp_idx'] + 1:<4} | {cd['n_micro_c']:<9} | {cd['n_edges_c']:<8} | {cd['triplets_c']:<8} | "
                  f"{u_nt_str:<9} | {u_tr_str:<9} | {u_non_str:<9}")

        if n_significant_components > 10:
            print(f"... (shown 10 of {n_significant_components} significant components)")

        print("-" * len(header))

    if total_weight > 0:
        final_u_nt = weighted_u_nt / total_weight
        final_u_tr = weighted_u_tr / total_weight
        final_u_non = weighted_u_non / total_weight
    else:
        final_u_nt = final_u_tr = final_u_non = 0.0

    failure_reason = None

    if n_significant_components == 0:
        if n_components_total == 0:
            failure_reason = "neighbor graph contains no connected components"
        elif total_valid_basins_all_components == 0:
            failure_reason = f"no component contains basins with size >= {min_basin_size}"
        elif max_basins_in_any_component < min_basins_per_comp:
            failure_reason = (f"maximum number of basins in a component ({max_basins_in_any_component}) "
                              f"is below MIN_BASINS_PER_COMPONENT ({min_basins_per_comp})")
        else:
            failure_reason = "no significant connected components found"
    elif total_weight == 0:
        if total_basins_all < 3:
            failure_reason = f"total number of basins in all significant components ({total_basins_all}) is less than 3 (cannot form triplets)"
        else:
            failure_reason = "failed to compute kinetic metric for any basin triplet (possibly all modes filtered as noise)"

    if failure_reason and verbose:
        print(f"\n[FAILURE REASON] Analysis produced no result: {failure_reason}.")

    valid_micro_indices = []
    for comp_data in valid_components_data:
        valid_micro_indices.extend(comp_data['micro_indices'])

    if len(valid_micro_indices) > 0:
        valid_fes = free_energies[valid_micro_indices]
        fe_mean_valid = float(np.mean(valid_fes))
        fe_std_valid = float(np.std(valid_fes))
    else:
        fe_mean_valid = np.nan
        fe_std_valid = np.nan

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
        'failure_reason': failure_reason,
        'fe_mean_valid': fe_mean_valid,
        'fe_std_valid': fe_std_valid
    }


# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================
def run_sensitivity_analysis(K_sym, adjacency, free_energies, n_micro, temp_k,
                             min_basin_size, min_basins_per_comp, num_modes,
                             eigs_sigma, eigs_tol, dense_threshold,
                             gap_threshold, eps, delta, sens_list,
                             verbose=True):
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

        if res is not None and res['total_triplets'] > 0:
            print(f"{eq_thresh:<12.2f} | {res['f_inter']:<8.4f} | "
                  f"{res['n_valid_components']:<6} | "
                  f"{res['total_basins']:<6} | "
                  f"{res['u_nontrivial']:>8.2f}   | "
                  f"{res['u_trivial']:>8.2f}   | "
                  f"{res['u_nonultrametric']:>8.2f}   | "
                  f"{res['total_triplets']:>6}")
        else:
            print(f"{eq_thresh:<12.2f} | {'--':<8} | {'--':<6} | "
                  f"{'--':<6} | {'--':>8}   | {'--':>8}   | {'--':>8}   | {'--':>6}")

    print("-" * len(header))


# ============================================================================
# COMMON FUNCTIONS FOR TRAJECTORY PROCESSING AND MSM CONSTRUCTION
# ============================================================================
def process_trajectory(dcd_file, pdb_file, energy_file, n_clusters, min_cluster_size, kramers_temp, rmsd_mode, verbose=True):
    import mdtraj as md

    if verbose:
        print(f"  Loading and clustering: {dcd_file}")

    traj = md.load(dcd_file, top=pdb_file)
    energies = np.loadtxt(energy_file)

    if rmsd_mode == 'ca':
        cluster_atom_indices = [a.index for a in traj.topology.atoms if a.name == 'CA']
    else:
        cluster_atom_indices = None

    cluster_obj = ClusterKCenters(traj, n_clusters, cluster_atom_indices)
    labels = cluster_obj.assign(traj)
    original_center_indices = cluster_obj.center_indices

    unique, counts = np.unique(labels, return_counts=True)
    valid_clusters = unique[counts >= min_cluster_size]

    if len(valid_clusters) < 3:
        if verbose:
            print(f"  [WARNING] Significant clusters {len(valid_clusters)} (need >= 3).")
        del traj
        gc.collect()
        return None

    label_map = {old: new for new, old in enumerate(valid_clusters)}
    filtered_labels = np.array([label_map[l] if l in label_map else -1 for l in labels])

    n_states = len(valid_clusters)

    RT = KB_KJ * kramers_temp
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
        print(f"  Microstates: {n_states}, F: [{free_energies.min():.1f}, {free_energies.max():.1f}] kJ/mol")

    return {
        'traj': traj,
        'pdb_file': pdb_file,
        'energy_file': energy_file,
        'n_states': n_states,
        'original_center_indices': original_center_indices,
        'valid_clusters': valid_clusters,
        'filtered_labels': filtered_labels,
        'free_energies': free_energies,
        'RT': RT
    }


def build_msm_from_sd(sd, eta, alpha, neighbor_method, radius_percentile, rmsd_mode, verbose=True):
    if neighbor_method == 'energy':
        adjacency = build_neighbor_graph_energy(sd['free_energies'], sd['RT'], eta, verbose=verbose)
    else:
        adjacency = build_adaptive_topological_graph(
            sd['traj'], sd['n_states'], radius_percentile, rmsd_mode, alpha,
            sd['original_center_indices'], sd['valid_clusters'], sd['filtered_labels'],
            verbose=verbose
        )

    diff = np.abs(sd['free_energies'][:, None] - sd['free_energies'][None, :])
    rates = np.exp(-diff / (2.0 * sd['RT']))

    K_sym = np.where(adjacency, rates, 0.0)
    np.fill_diagonal(K_sym, 0.0)

    max_F = np.maximum(sd['free_energies'][:, None], sd['free_energies'][None, :])
    K_ji = np.exp(-(max_F - sd['free_energies'][:, None]) / sd['RT'])
    K_ji = np.where(adjacency, K_ji, 0.0)
    np.fill_diagonal(K_ji, 0.0)

    diag_values = -np.sum(K_ji, axis=1)
    np.fill_diagonal(K_sym, diag_values)

    return K_sym, adjacency


# ============================================================================
# NEIGHBOR_ALPHA SENSITIVITY ANALYSIS (FIXED: NO REPEATED MD)
# ============================================================================
def run_alpha_analysis(samples_data, alpha_list, basin_eq_threshold, min_basin_size,
                       min_basins_per_comp, num_modes, eigs_sigma, eigs_tol,
                       dense_threshold, gap_threshold, eps, delta,
                       neighbor_method, eta, radius_percentile, rmsd_mode,
                       verbose=True):
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS TO NEIGHBOR_ALPHA PARAMETER")
    print(f"  Number of trajectories for each alpha: {len(samples_data)}")
    print("=" * 70)

    header1 = (f"{'Alpha':<8} | {'f_inter':<8} | {'Comp.(sign.)':<12} | "
               f"{'Basins':<12} | {'u_nt (%)':<14} | {'u_tr (%)':<14} | {'u_non (%)':<14} | {'Triplets':<12}")
    print(header1)
    print("-" * len(header1))

    results_list = []
    metric_keys = [
        'f_inter', 'n_valid_components', 'total_basins', 'u_nontrivial',
        'u_trivial', 'u_nonultrametric', 'total_triplets', 'n_edges',
        'n_components', 'fe_mean_valid', 'fe_std_valid'
    ]

    def fmt(val, std, width):
        if np.isnan(val):
            return f"{'--':<{width}}"
        if len(samples_data) > 1 and std > 0:
            return f"{val:.2f}+/-{std:.2f}".center(width)
        return f"{val:.2f}".center(width)

    for alpha_idx, alpha_val in enumerate(alpha_list):
        sample_results = []

        for sd in samples_data:
            K_sym, adjacency = build_msm_from_sd(
                sd, eta, alpha_val, neighbor_method, radius_percentile, rmsd_mode, verbose=False
            )

            res = analyze_landscape_components(
                K_sym, adjacency, sd['free_energies'], sd['n_states'], sd['RT'] / KB_KJ,
                basin_eq_threshold, min_basin_size, min_basins_per_comp,
                num_modes, eigs_sigma, eigs_tol, dense_threshold,
                gap_threshold, eps, delta,
                verbose=False
            )

            n_edges = np.sum(adjacency) // 2
            n_possible_edges = sd['n_states'] * (sd['n_states'] - 1) // 2

            if res is not None and res['total_triplets'] > 0:
                sample_results.append({
                    'f_inter': res['f_inter'],
                    'n_valid_components': res['n_valid_components'],
                    'total_basins': res['total_basins'],
                    'u_nontrivial': res['u_nontrivial'],
                    'u_trivial': res['u_trivial'],
                    'u_nonultrametric': res['u_nonultrametric'],
                    'total_triplets': res['total_triplets'],
                    'n_states': sd['n_states'],
                    'fe_min': np.min(sd['free_energies']),
                    'fe_max': np.max(sd['free_energies']),
                    'n_edges': n_edges,
                    'n_possible_edges': n_possible_edges,
                    'n_components': res['n_total_components'],
                    'fe_mean_valid': res['fe_mean_valid'],
                    'fe_std_valid': res['fe_std_valid']
                })

        if len(sample_results) == 0:
            print(f"{alpha_val:<8.2f} | {'--':<8} | {'--':<12} | {'--':<12} | {'--':<14} | {'--':<14} | {'--':<14} | {'--':<12}")

            agg = {
                'alpha': alpha_val,
                'n_states': np.nan,
                'fe_min': np.nan,
                'fe_max': np.nan,
                'n_possible_edges': np.nan
            }

            for key in metric_keys:
                agg[f'{key}_mean'] = np.nan
                agg[f'{key}_std'] = np.nan

            results_list.append(agg)
        else:
            agg = {}

            for key in metric_keys:
                values = [r[key] for r in sample_results if not np.isnan(r[key])]
                if len(values) > 0:
                    agg[f'{key}_mean'] = np.mean(values)
                    agg[f'{key}_std'] = np.std(values) if len(values) > 1 else 0.0
                else:
                    agg[f'{key}_mean'] = np.nan
                    agg[f'{key}_std'] = np.nan

            agg['alpha'] = alpha_val
            agg['n_states'] = sample_results[0]['n_states']
            agg['fe_min'] = np.mean([r['fe_min'] for r in sample_results])
            agg['fe_max'] = np.mean([r['fe_max'] for r in sample_results])
            agg['n_possible_edges'] = sample_results[0]['n_possible_edges']

            results_list.append(agg)

            f_inter_str = fmt(agg['f_inter_mean'], agg['f_inter_std'], 8)
            n_valid_str = fmt(agg['n_valid_components_mean'], agg['n_valid_components_std'], 12)
            basins_str = fmt(agg['total_basins_mean'], agg['total_basins_std'], 12)
            u_nt_str = fmt(agg['u_nontrivial_mean'], agg['u_nontrivial_std'], 14)
            u_tr_str = fmt(agg['u_trivial_mean'], agg['u_trivial_std'], 14)
            u_non_str = fmt(agg['u_nonultrametric_mean'], agg['u_nonultrametric_std'], 14)
            triplets_str = fmt(agg['total_triplets_mean'], agg['total_triplets_std'], 12)

            print(f"{alpha_val:<8.2f} | {f_inter_str} | {n_valid_str} | {basins_str} | {u_nt_str} | {u_tr_str} | {u_non_str} | {triplets_str}")

    print("-" * len(header1))

    print("\n" + "=" * 70)
    print("GRAPH AND FREE ENERGY STATISTICS FOR EACH ALPHA")
    if len(samples_data) > 1:
        print("  (values: mean +/- STD across trajectories)")
    print("=" * 70)

    header2 = (f"{'Alpha':<8} | {'Micro':<6} | {'F_min':<8} | {'F_max':<8} | "
               f"{'Edges':<14} | {'Possible edges':<12} | {'Comp.(total)':<14} | {'F_mean+/-std (valid)':<20}")
    print(header2)
    print("-" * len(header2))

    for r in results_list:
        fe_min_str = f"{r['fe_min']:.1f}" if not np.isnan(r['fe_min']) else "--"
        fe_max_str = f"{r['fe_max']:.1f}" if not np.isnan(r['fe_max']) else "--"

        if len(samples_data) > 1 and not np.isnan(r['n_edges_mean']):
            edges_str = f"{r['n_edges_mean']:.0f}+/-{r['n_edges_std']:.0f}"
        elif not np.isnan(r['n_edges_mean']):
            edges_str = f"{r['n_edges_mean']:.0f}"
        else:
            edges_str = "--"

        if len(samples_data) > 1 and not np.isnan(r['n_components_mean']):
            comp_str = f"{r['n_components_mean']:.1f}+/-{r['n_components_std']:.1f}"
        elif not np.isnan(r['n_components_mean']):
            comp_str = f"{r['n_components_mean']:.0f}"
        else:
            comp_str = "--"

        if not np.isnan(r['fe_mean_valid_mean']):
            if len(samples_data) > 1 and r['fe_mean_valid_std'] > 0:
                fe_valid_str = f"{r['fe_mean_valid_mean']:.1f}+/-{r['fe_mean_valid_std']:.1f}"
            else:
                fe_valid_str = f"{r['fe_mean_valid_mean']:.1f}"
        else:
            fe_valid_str = "--"

        n_states_print = r['n_states']
        if not np.isnan(n_states_print):
            n_states_print = f"{int(n_states_print)}"
        else:
            n_states_print = "--"

        poss_edges_print = r['n_possible_edges']
        if not np.isnan(poss_edges_print):
            poss_edges_print = f"{int(poss_edges_print)}"
        else:
            poss_edges_print = "--"

        print(f"{r['alpha']:<8.2f} | {n_states_print:<6} | {fe_min_str:<8} | {fe_max_str:<8} | "
              f"{edges_str:<14} | {poss_edges_print:<12} | {comp_str:<14} | {fe_valid_str:<20}")

    print("-" * len(header2))


# ============================================================================
# MAIN ANALYSIS (FIXED: USES PREPARED TRAJECTORIES)
# ============================================================================
def run_main_analysis(samples_data, kramers_temperature, basin_eq_threshold, min_basin_size,
                      min_basins_per_comp, num_modes, eigs_sigma, eigs_tol, eigs_dense_threshold,
                      spectral_gap_threshold, ultrametric_epsilon, ultrametric_delta,
                      neighbor_method, eta_neighbor, eta_neighbor_mode,
                      neighbor_alpha, radius_percentile, rmsd_mode,
                      verbose=True):
    print("\n" + "=" * 70)
    print("MAIN ANALYSIS (MAIN_ANALYSIS)")
    print(f"  Analysis of {len(samples_data)} independent trajectories with base parameters.")
    print("=" * 70)

    u_nt_list = []
    u_tr_list = []
    u_non_list = []
    n_basins_list = []
    f_inter_list = []
    failure_reasons = []

    for i, sd in enumerate(samples_data):
        if verbose:
            print(f"\n--- Analysis of trajectory {i + 1}/{len(samples_data)} ---")

        effective_eta = eta_neighbor

        if neighbor_method == 'energy' and eta_neighbor_mode == 'auto':
            effective_eta = auto_select_eta_neighbor(
                sd['free_energies'], sd['RT'], min_basin_size,
                min_basins_per_comp, basin_eq_threshold,
                verbose=verbose
            )

        K_sym, adjacency = build_msm_from_sd(
            sd, effective_eta, neighbor_alpha, neighbor_method, radius_percentile, rmsd_mode, verbose=verbose
        )

        results = analyze_landscape_components(
            K_sym, adjacency, sd['free_energies'], sd['n_states'],
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
                print(f"  [MAIN] Trajectory {i + 1}: u_nt={results['u_nontrivial']:.2f}%, "
                      f"u_tr={results['u_trivial']:.2f}%, u_non={results['u_nonultrametric']:.2f}%")
        else:
            reason = results.get('failure_reason', 'unknown reason') if results else 'unknown reason'
            failure_reasons.append(reason)

            if verbose:
                print(f"  [MAIN] Trajectory {i + 1}: failed to compute u_nt (skipped).")

    if len(u_nt_list) == 0:
        if verbose:
            print("\nERROR: All MAIN_ANALYSIS trajectories failed to produce a result.")
            if failure_reasons:
                unique_reasons = list(set(failure_reasons))
                print("Failure reasons by trajectories:")
                for r in unique_reasons:
                    count = failure_reasons.count(r)
                    print(f"  - {r} (in {count} trajectories)")
        return None

    u_nt_list = np.array(u_nt_list)
    u_tr_list = np.array(u_tr_list)
    u_non_list = np.array(u_non_list)
    n_basins_list = np.array(n_basins_list)
    f_inter_list = np.array(f_inter_list)

    stats_dict = {
        'u_nt_list': u_nt_list, 'u_tr_list': u_tr_list, 'u_non_list': u_non_list,
        'n_basins_list': n_basins_list, 'f_inter_list': f_inter_list,
        'mean_nt': np.mean(u_nt_list), 'std_nt': np.std(u_nt_list),
        'mean_tr': np.mean(u_tr_list), 'std_tr': np.std(u_tr_list),
        'mean_non': np.mean(u_non_list), 'std_non': np.std(u_non_list),
        'n_valid': len(u_nt_list), 'elapsed': 0
    }

    return stats_dict


def print_main_analysis_results(results_main, neighbor_alpha, verbose=True):
    if results_main is None:
        if verbose:
            print("\nMAIN_ANALYSIS: no data to print.")
        return

    print("\n" + "=" * 70)
    print("SUMMARY TABLE OF MAIN ANALYSIS RESULTS (MAIN_ANALYSIS)")
    print(f"  (Per-trajectory details for base NEIGHBOR_ALPHA = {neighbor_alpha})")
    print("=" * 70)

    header = f"{'Trajectory':<12} | {'u_nt (%)':<10} | {'u_tr (%)':<10} | {'u_non (%)':<10} | {'Basins':<6} | {'f_inter':<8}"
    print(header)
    print("-" * len(header))

    for i in range(len(results_main['u_nt_list'])):
        print(f"{i + 1:<12} | {results_main['u_nt_list'][i]:>8.2f}   | "
              f"{results_main['u_tr_list'][i]:>8.2f}   | "
              f"{results_main['u_non_list'][i]:>8.2f}   | "
              f"{results_main['n_basins_list'][i]:>5} | "
              f"{results_main['f_inter_list'][i]:>7.4f}")

    print("-" * len(header))

    mean_nt = results_main['mean_nt']
    std_nt = results_main['std_nt']
    cv = (std_nt / max(abs(mean_nt), 0.01)) * 100

    print("\nFINAL STATISTICS AND STABILITY REPORT:")
    print(f"  Number of successful trajectories: {results_main['n_valid']}")
    print(f"  Mean u_nt:                         {mean_nt:.2f} %")
    print(f"  Standard deviation (STD):          {std_nt:.2f} %")
    print(f"  Coefficient of variation (CV):     {cv:.1f} %")

    if cv < 5.0:
        print("  CONCLUSION: Result is HIGHLY STABLE (CV < 5%).")
    elif cv < 10.0:
        print("  CONCLUSION: Result is stable (5% <= CV < 10%).")
    elif cv < 20.0:
        print("  CONCLUSION: Result is moderately stable (10% <= CV < 20%).")
    else:
        print("  CONCLUSION: Result is UNSTABLE (CV >= 20%). Increase trajectory length.")

    print("=" * 70)


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    v = VERBOSE

    print("=" * 70)
    print("ULTRAMETRICITY ANALYSIS OF PROTEIN ENERGY LANDSCAPE")
    print("Kinetic metric based on spectral decomposition (v1.12)")
    print("=" * 70)

    print(f"  Protein:              {PDB_ID}")
    print(f"  Force field:          {'GB99dms' if USE_GB99DMS else 'Amber14/OBC2'}")
    print(f"  MD temperature:       {TEMPERATURE_K} K")
    print(f"  MD steps:             {MD_STEPS}, save interval: {MD_REPORT_INTERVAL}")
    print(f"  MD step (initial):    {MD_TIME_STEP_FS} fs, rollback: {MD_ROLLBACK_STEPS} steps")
    print(f"  Kramers temperature:  {KRAMERS_TEMPERATURE} K")
    print(f"  Neighborhood method:  {NEIGHBOR_METHOD}")

    if NEIGHBOR_METHOD == 'energy':
        print(f"  Eta mode:             {ETA_NEIGHBOR_MODE}")
        if ETA_NEIGHBOR_MODE == 'manual':
            RT_print = KB_KJ * KRAMERS_TEMPERATURE
            df_max_print = -RT_print * np.log(ETA_NEIGHBOR)
            print(f"  eta (manual):         {ETA_NEIGHBOR:.0e} -> dF_max = {df_max_print:.1f} kJ/mol")
    elif NEIGHBOR_METHOD == 'adaptive_topology':
        print(f"  Radius percentile:    {NEIGHBOR_RADIUS_PERCENTILE}%")
        print(f"  Neighbor alpha:       {NEIGHBOR_ALPHA}")

    print(f"  RMSD metric:          {'C-alpha' if GEOMETRY_RMSD_MODE == 'ca' else 'all atoms'}")
    print(f"  Microstates:          {N_CLUSTERS} (target), min size: {MIN_CLUSTER_SIZE}")
    print(f"  Plateau threshold:    {BASIN_EQ_THRESHOLD} kJ/mol, min basin size: {MIN_BASIN_SIZE}")
    print(f"  Min basins per comp.: {MIN_BASINS_PER_COMPONENT}")
    print(f"  Modes:                {NUM_EIGENMODES} (gap threshold: {SPECTRAL_GAP_THRESHOLD:.0e})")
    print(f"  EIGS sigma:           {EIGS_SIGMA:.0e}, tol: {EIGS_TOL:.0e}, dense_threshold: {EIGS_DENSE_THRESHOLD}")
    print(f"  Ultrametricity:       epsilon={ULTRAMETRIC_EPSILON}, delta={ULTRAMETRIC_DELTA}")
    print(f"  CPU threads:          {num_cores} (limit: {MAX_CPU_THREADS})")

    print("-" * 70)
    print("  SELECTED WORK MODES:")
    print(f"  FROM_FILES_ANALYSIS: {'Yes' if FROM_FILES_ANALYSIS else 'No'}")
    print(f"  Sensitivity analysis: {'Yes (list: ' + str(SENSITIVITY_RANGE) + ')' if SENSITIVITY_ANALYSIS else 'No'}")
    print(f"  ALPHA analysis:      {'Yes (list: ' + str(ALPHA_RANGE) + ')' if ALPHA_ANALYSIS else 'No'}")
    print(f"  MAIN_ANALYSIS:       {'Yes (' + str(MAIN_NUM_SAMPLES) + ')' if MAIN_ANALYSIS else 'No'}")
    print(f"  GPU:                 {'Yes' if USE_GPU else 'No'}")
    print(f"  Verbose:             {'Yes' if v else 'No'}")
    print("=" * 70)

    sens_active = SENSITIVITY_ANALYSIS and isinstance(SENSITIVITY_RANGE, list) and len(SENSITIVITY_RANGE) > 0
    alpha_active = ALPHA_ANALYSIS and isinstance(ALPHA_RANGE, list) and len(ALPHA_RANGE) > 0

    if alpha_active and NEIGHBOR_METHOD != 'adaptive_topology':
        if v:
            print("\n[WARNING] ALPHA_ANALYSIS is enabled, but NEIGHBOR_METHOD != 'adaptive_topology'.")
            print("Alpha sensitivity analysis makes sense only for adaptive topological graph.")
            print("ALPHA_ANALYSIS mode will be skipped.")
        alpha_active = False

    main_active = MAIN_ANALYSIS

    if sens_active and alpha_active:
        print("\n" + "=" * 70)
        print("[ERROR] ANALYSIS MODE CONFLICT!")
        print("SENSITIVITY_ANALYSIS and ALPHA_ANALYSIS cannot be True at the same time.")
        print("Please select only one sensitivity analysis mode.")
        print("=" * 70)
        return None

    if not (sens_active or alpha_active or main_active):
        print("\n" + "=" * 70)
        print("ALL ANALYTICAL MODES ARE DISABLED.")
        print("Enable at least one of the flags:")
        print("  SENSITIVITY_ANALYSIS = True (with non-empty SENSITIVITY_RANGE)")
        print("  ALPHA_ANALYSIS = True (with non-empty ALPHA_RANGE)")
        print("  MAIN_ANALYSIS = True")
        print("Program terminated.")
        print("=" * 70)
        return None

    total_start = time.time()

    base_name = os.path.splitext(os.path.basename(STARTING_STRUCTURE_FILE))[0] if STARTING_STRUCTURE_FILE and os.path.exists(STARTING_STRUCTURE_FILE) else PDB_ID

    samples_data = []

    if FROM_FILES_ANALYSIS:
        print("\n" + "=" * 70)
        print(">>> FROM_FILES_ANALYSIS MODE: LOADING DATA FROM DISK <<<")
        print("=" * 70)

        pattern = f"{base_name}_trajectory_*.dcd"
        files = glob.glob(pattern)

        samples = []
        for f in files:
            match = re.search(r'_trajectory_(\d+)\.dcd$', f)
            if match:
                samples.append(int(match.group(1)))

        samples.sort()

        if not samples:
            if os.path.exists(f"{base_name}_trajectory.dcd"):
                samples = [None]
            else:
                print(f"\n[ERROR] No trajectory files found for {base_name}.")
                print("Program terminated.")
                return None

        print(f"Samples detected: {len(samples)}")

        for sample_idx in samples:
            if v:
                if sample_idx is not None:
                    print(f"\n--- Loading and clustering sample {sample_idx} ---")
                else:
                    print("\n--- Loading and clustering single sample ---")

            dcd_file, pdb_file, energy_file = load_trajectory_and_energies_from_files(
                PDB_ID, STARTING_STRUCTURE_FILE, USE_GB99DMS, USE_GPU,
                KRAMERS_TEMPERATURE, sample_idx=sample_idx, verbose=v
            )

            if dcd_file is None:
                continue

            sd = process_trajectory(dcd_file, pdb_file, energy_file, N_CLUSTERS, MIN_CLUSTER_SIZE, KRAMERS_TEMPERATURE, GEOMETRY_RMSD_MODE, verbose=v)

            if sd is not None:
                sd['sample_idx'] = sample_idx
                samples_data.append(sd)
            else:
                if v:
                    print(f"  [WARNING] Sample {sample_idx} discarded (not enough clusters).")
    else:
        num_sims_to_run = MAIN_NUM_SAMPLES if (main_active or alpha_active) else 1
        print(f"\n>>> TRAJECTORY GENERATION: {num_sims_to_run} simulations <<<")

        for i in range(1, num_sims_to_run + 1):
            sample_idx = i if num_sims_to_run > 1 else None
            current_seed = NULL_SEED_BASE + (i - 1) if num_sims_to_run > 1 else MD_SEED

            if v:
                print(f"\n--- Simulation {i}/{num_sims_to_run} (seed = {current_seed}) ---")

            dcd_file, pdb_file, energy_file = run_md_simulation(
                PDB_ID, USE_GB99DMS, TEMPERATURE_K, MD_STEPS, MD_REPORT_INTERVAL, USE_GPU,
                current_seed, STARTING_STRUCTURE_FILE, ADD_HYDROGENS_TO_CUSTOM_START,
                MINIMIZE_STARTING_STRUCTURE, MD_TIME_STEP_FS, MD_ROLLBACK_STEPS, MD_MAX_ROLLBACKS,
                MD_TIME_STEP_REDUCTION_FACTOR, MD_MIN_TIME_STEP_FS,
                sample_idx=sample_idx, verbose=v
            )

            if dcd_file is None:
                continue

            sd = process_trajectory(dcd_file, pdb_file, energy_file, N_CLUSTERS, MIN_CLUSTER_SIZE, KRAMERS_TEMPERATURE, GEOMETRY_RMSD_MODE, verbose=v)

            if sd is not None:
                sd['sample_idx'] = sample_idx
                samples_data.append(sd)
            else:
                if v:
                    print(f"  [WARNING] Simulation {i} discarded (not enough clusters).")

    if not samples_data:
        print("\n[ERROR] No successful trajectories available for analysis.")
        return None

    print(f"\nTotal samples prepared for analysis: {len(samples_data)}")

    if sens_active:
        sd0 = samples_data[0]

        effective_eta = ETA_NEIGHBOR
        if NEIGHBOR_METHOD == 'energy' and ETA_NEIGHBOR_MODE == 'auto':
            effective_eta = auto_select_eta_neighbor(
                sd0['free_energies'], sd0['RT'], MIN_BASIN_SIZE,
                MIN_BASINS_PER_COMPONENT, BASIN_EQ_THRESHOLD, verbose=v
            )

        K_sym, adjacency = build_msm_from_sd(
            sd0, effective_eta, NEIGHBOR_ALPHA, NEIGHBOR_METHOD, NEIGHBOR_RADIUS_PERCENTILE, GEOMETRY_RMSD_MODE, verbose=v
        )

        run_sensitivity_analysis(
            K_sym=K_sym,
            adjacency=adjacency,
            free_energies=sd0['free_energies'],
            n_micro=sd0['n_states'],
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
            print(">>> STARTING MODE: ALPHA SENSITIVITY ANALYSIS <<<")
            print("=" * 70)

        run_alpha_analysis(
            samples_data=samples_data,
            alpha_list=ALPHA_RANGE,
            basin_eq_threshold=BASIN_EQ_THRESHOLD,
            min_basin_size=MIN_BASIN_SIZE,
            min_basins_per_comp=MIN_BASINS_PER_COMPONENT,
            num_modes=NUM_EIGENMODES,
            eigs_sigma=EIGS_SIGMA,
            eigs_tol=EIGS_TOL,
            dense_threshold=EIGS_DENSE_THRESHOLD,
            gap_threshold=SPECTRAL_GAP_THRESHOLD,
            eps=ULTRAMETRIC_EPSILON,
            delta=ULTRAMETRIC_DELTA,
            neighbor_method=NEIGHBOR_METHOD,
            eta=ETA_NEIGHBOR,
            radius_percentile=NEIGHBOR_RADIUS_PERCENTILE,
            rmsd_mode=GEOMETRY_RMSD_MODE,
            verbose=v
        )

    if main_active:
        if v:
            print("\n" + "=" * 70)
            print(">>> STARTING MODE: MAIN ANALYSIS (MAIN_ANALYSIS) <<<")
            print("=" * 70)

        results_main = run_main_analysis(
            samples_data=samples_data,
            kramers_temperature=KRAMERS_TEMPERATURE,
            basin_eq_threshold=BASIN_EQ_THRESHOLD,
            min_basin_size=MIN_BASIN_SIZE,
            min_basins_per_comp=MIN_BASINS_PER_COMPONENT,
            num_modes=NUM_EIGENMODES,
            eigs_sigma=EIGS_SIGMA,
            eigs_tol=EIGS_TOL,
            eigs_dense_threshold=EIGS_DENSE_THRESHOLD,
            spectral_gap_threshold=SPECTRAL_GAP_THRESHOLD,
            ultrametric_epsilon=ULTRAMETRIC_EPSILON,
            ultrametric_delta=ULTRAMETRIC_DELTA,
            neighbor_method=NEIGHBOR_METHOD,
            eta_neighbor=ETA_NEIGHBOR,
            eta_neighbor_mode=ETA_NEIGHBOR_MODE,
            neighbor_alpha=NEIGHBOR_ALPHA,
            radius_percentile=NEIGHBOR_RADIUS_PERCENTILE,
            rmsd_mode=GEOMETRY_RMSD_MODE,
            verbose=v
        )

        if results_main is not None:
            print_main_analysis_results(results_main, NEIGHBOR_ALPHA, verbose=v)

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 70}")
    print(f"TOTAL EXECUTION TIME: {total_elapsed:.1f} sec ({total_elapsed / 60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
