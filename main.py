
"""
================================================================================
ULTRAMETRIC ANALYSIS OF PROTEIN ENERGY LANDSCAPES
BASED ON KINETIC METRICS AND MARKOV STATE MODELS (MSM)
Author: A. P. Zubarev
Version: 1.22
================================================================================
DESCRIPTION OF THE METHOD AND PURPOSE OF THE PROGRAM
--------------------------------------------------------------------------------
This program implements a method for testing the hypothesis of ultrametric
organization of protein energy landscapes. The method is based on the analysis
of transition kinetics between macrostates (basins of attraction) using
a kinetic metric constructed via spectral decomposition of the symmetrized
Kramers transition rate matrix and the Mahalanobis distance.
The program performs the following computational stages:
1. MOLECULAR DYNAMICS (MD) STAGE
Generation of an equilibrium conformational ensemble using Langevin dynamics
with the OpenMM package. Implicit solvent force fields GB99dms and Amber14/OBC2
are supported.
ROBUST STRUCTURE PREPARATION:
Before running MD, the structure is automatically cleaned: non-protein residues
(ligands, ions, water) are removed, the longest polypeptide chain is selected,
and missing atoms/hydrogens are added. This ensures compatibility with OpenMM
force fields for any PDB records.
IN-MEMORY BUFFERING:
Instead of writing directly to disk via DCDReporter, the program uses a custom
MemoryReporter that accumulates coordinates and energies in Python lists.
This solves the fundamental problem of "garbage" frames when NaNs occur:
lists in RAM can be atomically truncated, whereas binary DCD files cannot be
rolled back. The single DCD file is written to disk via mdtraj ONLY ONCE at
the very end, after successful completion of the simulation.
If the number of rollbacks exceeds the limit (MD_MAX_ROLLBACKS), the integration
time step is automatically reduced, and the simulation is restarted from scratch.
2. MARKOV STATE MODEL (MSM) CONSTRUCTION STAGE
The continuous conformational space is discretized by clustering trajectory
frames into microstates using the k-centers algorithm in the RMSD metric.
The choice of atoms for RMSD calculation is determined by the GEOMETRY_RMSD_MODE
parameter:
- 'ca': only C-alpha atoms (faster, reflects backbone folding).
- 'all': RMSD over all atoms (stricter, accounts for side chains).
IMPORTANT: The selected metric is used UNIFORMLY for both clustering and
building the neighbor graph, ensuring internal consistency of space discretization.
For each microstate (cluster), the free energy is calculated via the cluster
population (number of frames) using the formula F_i = -RT ln(N_i / N_total).
Since sampling is performed from the Gibbs distribution at temperature T,
the cluster population already contains information about both energy and entropy,
ensuring a thermodynamically correct calculation of free energy without
double Boltzmann weighting.
The neighbor graph is built using one of two methods:
- Energetic: two clusters are neighbors if the difference in their free energies
does not exceed the threshold dF_max = -RT ln(eta). This criterion is invariant
to energy shifts and ensures correct connectivity at all energy levels.
- Adaptive topological graph: for each cluster, an effective radius is calculated
(a specified percentile of distances from the center to the cluster points).
Two clusters are neighbors if the distance between their centers does not exceed
(R_i + R_j).
Based on the free energies and the neighbor graph, a symmetrized transition rate
matrix K_sym is constructed. The elements K_sym_ij are calculated using the
Kramers formula: exp(-|F_i - F_j| / (2RT)) for neighboring microstates, else 0.
Macrostates (basins of attraction) are identified by gradient descent on the
free energy landscape. Attractors (local minima and plateaus) are identified
automatically; each microstate is assigned to the basin reached by gradient descent.
BASIN_EQ_THRESHOLD sets the energy equivalence threshold below which states
belong to the same plateau.
IMPORTANT NOTE ON THE ALGORITHM:
1. First, attractors are identified (states that have no neighbors strictly
below F_i - epsilon_eq).
2. Equivalence graph analysis is performed ONLY among attractors to correctly
merge degenerate minima/plateaus.
3. Gradient descent is applied exclusively to unlabeled slope states.
The analysis is performed strictly by connected components. For each component
with a number of basins >= MIN_BASINS_PER_COMPONENT, ultrametricity is calculated
independently. The final ultrametricity is calculated as a weighted average.
The fragmentation index f_inter is also calculated.
3. SPECTRAL ANALYSIS AND ULTRAMETRICITY TESTING STAGE
For each significant connected component, spectral decomposition of K_sym_c
is performed. Noise modes are filtered out using the spectral gap criterion.
Characteristic vectors of basins are projected onto physical eigenmodes.
The distance between basins is the Mahalanobis distance: the weighted Euclidean
norm of the differences in projections with weights 1/|lambda_k|.
Slower relaxation modes contribute more to the distance.
Ultrametricity is tested by enumerating all triplets of basins. A triplet is
considered non-trivially ultrametric if the two largest distances are approximately
equal (relative difference <= epsilon), and the third is significantly smaller
(relative difference > delta). The fraction of such triplets is u_nt.
4. EXECUTION ARCHITECTURE AND MODES
The program operates in one or more of the following modes:
- FROM_FILES_ANALYSIS: Skip MD, load existing DCD/CIF/PDB files. The program
automatically finds all samples (_1, _2, ...), clusters them, and builds the MSM.
Then the enabled analytical modes are launched sequentially.
There is a FROM_FILES_POOL_ENABLE mode. If enabled along with FROM_FILES_ANALYSIS,
all found trajectories are combined into one large pool trajectory, and the analysis
is performed as for a single sample (without calculating standard deviations across
trajectories).
- SENSITIVITY_ANALYSIS: Run analysis for multiple values of BASIN_EQ_THRESHOLD
specified in SENSITIVITY_RANGE.
- MAIN_ANALYSIS: Analysis with base parameters for all generated or loaded
trajectories. Mean, standard deviation, and CV are calculated to assess stability.
5. DIAGNOSTICS OF SKIPPED TRAJECTORIES
The program always (regardless of the VERBOSE value) outputs information about
trajectories that were excluded from the analysis. For each value of the analyzed
parameter, the number of successful trajectories and a list of skipped samples
with the reason for skipping are printed. A "Success" column is added to the
result tables, showing the ratio of successful trajectories to the total number
of loaded or generated trajectories. This allows unambiguous interpretation of
statistical values as calculated only over successful trajectories.
6. DETAILED OUTPUT FOR EACH TRAJECTORY
When VERBOSE = True, the program outputs detailed tables separately for each
trajectory (sample) before forming summary tables. For each trajectory, it prints:
- details of building the neighbor graph;
- diagnostics of basin formation by connected components;
- table of significant connected components;
- individual result table for this trajectory.
When VERBOSE = False, the program prints only mandatory skip diagnostics and
summary tables.
TECHNICAL REQUIREMENTS
--------------------------------------------------------------------------------
Runtime environment: Windows 10/11, Linux, macOS. Python 3.10-3.12.
Required packages (pip install):
- openmm, pdbfixer, mdtraj, numpy, scipy, threadpoolctl, torch
HARDWARE REQUIREMENTS
--------------------------------------------------------------------------------
GPU (NVIDIA CUDA) is recommended for MD. Automatic fallback to CPU.
MSM and spectral analysis stages are executed on CPU.
USER PARAMETERS
--------------------------------------------------------------------------------
All customizable parameters are grouped below with detailed comments.
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
# IMPORTANT FOR WINDOWS: SUPPORT FOR CUDA JIT COMPILATION (NVRTC)
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
# GROUP 1: CHOICE OF OPERATING MODES AND GENERAL SETTINGS
# ----------------------------------------------------------------------------
MAX_CPU_THREADS = 20
"""
Maximum number of CPU threads used for computations (OpenMP, BLAS, LAPACK).
Set this value lower than the total number of logical cores of your processor
to leave resources for the operating system and background tasks, avoiding
system hangs and overloads.
If set to 0 or None, the program automatically uses all available logical cores
(multiprocessing.cpu_count()).
Recommended value: 22 (for a 24-thread processor) or 0 (for auto).
"""

FROM_FILES_ANALYSIS = True
"""
Enable analysis mode from ready-made files (skip MD stage).
True  — The program skips the expensive molecular dynamics stage and
loads ready-made files from disk:
- {PDB_ID}_trajectory_N.dcd
- {PDB_ID}_N.pdb (or .cif)
- {PDB_ID}_energies_N.txt
The program automatically scans the folder, finds all samples (N=1, 2, ...),
clusters them, and performs analysis for each.
False — The program executes the full cycle, starting with MD, generating
MAIN_NUM_SAMPLES trajectories and saving them to disk with suffixes _1, _2, ...
Recommended value: False (for initial calculation),
True (for re-analysis of a ready trajectory).
"""

FROM_FILES_POOL_ENABLE = False
"""
Enable pooling mode (combining all loaded trajectories into one).
Used ONLY when FROM_FILES_ANALYSIS = True.
True  — The program loads all found trajectory files (_1, _2, ...),
combines them into one large trajectory, and performs analysis (clustering,
MSM construction, ultrametricity testing) as for a single sample.
In this mode, result tables are output without calculating standard deviation
(STD), since the analysis is performed for a single combined sample.
If METRIC_OUTPUT = True, the kinetic metric matrix is saved to the working
directory with the name seed_pooled_alpha_...npy.
False — The program processes each trajectory independently, calculating
mean values and STD across the ensemble of trajectories (standard behavior).
Recommended value: False (for statistical analysis over independent realizations),
True (to maximize data volume and detect rare events, if trajectories represent
fragments of a single process).
"""

SENSITIVITY_ANALYSIS = False
"""
Enable sensitivity analysis of results to the value of the energy equivalence
threshold BASIN_EQ_THRESHOLD.
If True, the program executes a computation loop for a set of EQ_THRESHOLD values
specified by the SENSITIVITY_RANGE list, and outputs a summary table.
This allows assessing the stability of the landscape's hierarchical structure
to the choice of this parameter without needing a preliminary "main calculation".
Logic features:
- An empty list SENSITIVITY_RANGE = [] is equivalent to disabling the mode.
- A list of one element (e.g., [3.0]) is equivalent to a single main calculation
from previous versions of the program.
- A non-empty list triggers the computation loop and table formation.
Recommended value: True (to study landscape stability).
"""

SENSITIVITY_RANGE = [1.0, 3.0, 5.0, 7.0]
"""
List of energy equivalence threshold values BASIN_EQ_THRESHOLD (in kJ/mol)
for sensitivity analysis.
Used ONLY when SENSITIVITY_ANALYSIS = True.
Format: explicit list of values, e.g., [1.0, 3.0, 5.0, 7.0].
The program will sequentially perform analysis for each value in the list
and output a summary table of results.
Logic features:
- An empty list [] is equivalent to disabling the mode (SENSITIVITY_ANALYSIS = False).
- A list of one element (e.g., [3.0]) is equivalent to a single main calculation
from previous versions of the program.
- A non-empty list triggers the computation loop and table formation.
Recommended value: [2.0, 2.5, 3.0] (three values around the base 2.5 kJ/mol).
For broader analysis: [1.0, 3.0, 5.0, 7.0, 10.0].
"""

MAIN_ANALYSIS = True
"""
Enable Main Analysis mode.
True  — Preliminary main calculation is NOT performed. The full computational
cycle (MD -> clustering -> MSM -> analysis) is executed MAIN_NUM_SAMPLES
times with different random seeds. Based on the results of all runs,
mean values, standard deviations (STD), and the coefficient of variation of u_nt
are calculated. A final table and a detailed stability report are generated.
False — Main analysis mode is skipped.
Logic: This mode is the MAIN TOOL for obtaining statistically significant results.
The entire computational volume is immediately distributed over MAIN_NUM_SAMPLES
independent runs.
MD errors (including NaN coordinates) are handled automatically: failed runs
are skipped, and the program continues working.
Computational complexity: ~10 minutes per run (on GPU). With 10 runs, the entire
test takes ~1.5–2 hours. It is recommended to enable for final calculations
and publication of results.
FEATURE IN FROM_FILES_ANALYSIS MODE:
If FROM_FILES_ANALYSIS = True, the program automatically determines MAIN_NUM_SAMPLES
by the number of available files on disk.
Recommended value: True (for final calculations), False (for quick tests).
"""

MAIN_NUM_SAMPLES = 10
"""
Number of independent runs for Main Analysis mode (MAIN_ANALYSIS).
Determines the number of full restarts of the program with different seeds.
Each run includes the entire chain: MD -> clustering -> MSM -> analysis.
Based on the results, mean, STD are calculated and a table is formed.
FEATURE: In FROM_FILES_ANALYSIS = True mode, this parameter is ignored,
since the number of samples is determined automatically by files on disk.
Recommended value: 5–10 (compromise between statistical reliability and
computation time). To obtain high-precision STD, it can be increased to 20-30.
Valid range: 1 – 100. (With a value of 1, the mode is equivalent to a single run).
"""

NULL_SEED_BASE = 12345
"""
Base seed for random number generation in main analysis (MAIN_ANALYSIS).
Each realization uses seed = NULL_SEED_BASE + realization_number,
which ensures reproducibility of results upon re-run.
Recommended value: 12345 (any integer).
"""

USE_GPU = True
"""
Use GPU (CUDA) for molecular dynamics.
True  — The program attempts to use an NVIDIA GPU. If the GPU is unavailable
(no driver, CUDA toolkit not installed) or a JIT compilation error occurs
(e.g., missing nvrtc DLL), the program AUTOMATICALLY AND SEAMLESSLY
switches to CPU with a warning.
False — The program is guaranteed to use only CPU at all stages.
Useful for running on machines without an NVIDIA graphics card or when
CUDA errors occur.
Performance impact:
GPU: ~2–3 minutes for 500,000 MD steps.
CPU: ~20–30 minutes for 500,000 MD steps.
Recommended value: True (if GPU is available).
"""

VERBOSE = True
"""
Verbose console output.
True  — Detailed information about the progress of all stages is output:
intermediate MD results, clustering statistics, neighbor graph properties,
list of connected components, null model progress.
When VERBOSE=True in multi-trajectory analysis mode, the program outputs
detailed tables separately for each trajectory (sample), including diagnostics
of connected components and an individual result table.
When FROM_FILES_POOL_ENABLE=True, the program outputs results for a single
combined trajectory without STD.
False — Only key results (summary tables) and mandatory diagnostics of skipped
trajectories are output. Useful for batch runs or redirecting output to a file.
Recommended value: True.
"""

N_COMP_CON = 10
"""
Number of largest connected components whose information is output to the console table.
Recommended value: 10.
"""

METRIC_OUTPUT = False
"""
Whether to output kinetic metric matrices to files.
If True, then for each seed the program saves the kinetic metric matrix
(Mahalanobis distances between basins of attraction) for the largest connected
component of the microstate graph to a numpy format file (.npy).
The file is saved to the program folder with the name:
"seed_seed_number.npy"
For example: "seed_1.npy".
In FROM_FILES_POOL_ENABLE mode, the file is saved as "seed_pooled.npy".
If False, matrices are not saved and the program works as before.
Recommended value: False (enable only if necessary to obtain matrices
for external analysis).
"""

# ----------------------------------------------------------------------------
# GROUP 2: MOLECULAR DYNAMICS (MD) PARAMETERS
# ----------------------------------------------------------------------------
PDB_ID = "1L2Y"
"""
Protein identifier in the RCSB Protein Data Bank database.
Determines the starting structure for simulation if a custom starting structure
file (STARTING_STRUCTURE_FILE) is not specified.
The structure is automatically loaded from RCSB PDB, cleaned of foreign molecules
(water, ligands), supplemented with missing atoms and hydrogen atoms at pH 7.0.
Before running dynamics, local energy minimization is performed.
IN FROM_FILES_ANALYSIS MODE:
PDB_ID is used as the base name for searching files on disk:
{PDB_ID}_trajectory.dcd, {PDB_ID}.cif (or .pdb), {PDB_ID}_energies.txt.
Recommended values for testing:
"1L2Y" — Trp-cage (20 residues, folds quickly)
"1UBQ" — ubiquitin (76 residues)
"1JSF" — protein with ligands (tests robust structure cleaning)
"2RVD" — short peptide (10 residues)
Valid values: any valid four-character PDB code.
"""

STARTING_STRUCTURE_FILE = ""
"""
Path to the custom starting structure file (PDB or CIF format).
If the string is left empty (""), the program automatically loads the native
structure from RCSB PDB by the PDB_ID identifier.
Using an alternative starting structure (e.g., a structure obtained from another
simulation, or a model built by homology) allows:
- avoiding artificial trapping in a deep native minimum;
- starting the simulation from an unfolded or partially folded state;
- exploring the landscape in the vicinity of the conformation of interest.
If the file contains only heavy atoms, set ADD_HYDROGENS_TO_CUSTOM_START = True.
IN FROM_FILES_ANALYSIS MODE:
If STARTING_STRUCTURE_FILE is not empty, its base name (without extension)
is used to search for the DCD file and energy file.
For example, for "myprotein.cif" the program will search for:
- myprotein_trajectory.dcd
- myprotein_energies.txt (optional)
"""

ADD_HYDROGENS_TO_CUSTOM_START = False
"""
Whether to add hydrogen atoms to the custom starting structure.
False — used if the PDB/CIF file already contains hydrogens (e.g., prepared
via pdbfixer or taken from a simulation with hydrogens).
True  — used if the file contains only heavy atoms; hydrogens will be added
automatically according to the selected force field.
Recommended value: False.
"""

USE_GB99DMS = True
"""
Choice of force field for molecular dynamics.
True  — Use the GB99dms force field with implicit solvent.
Optimized to reproduce properties of both ordered and disordered proteins.
The GB99dms.xml file is automatically loaded from the official repository
upon first run. Contains Debye-Hückel screening to account for ionic strength.
False — Use the standard Amber14-all force field with the OBC2 implicit solvent
model (obc2.xml). A more conservative choice, well-tested in the literature.
Impact: The choice of force field determines the relief of the energy landscape
and, consequently, the set of visited conformations, depth of minima, and
relaxation times. GB99dms gives more realistic ensembles for disordered proteins;
Amber14/OBC2 may be preferable for compact globular proteins.
IN FROM_FILES_ANALYSIS MODE:
This parameter is also used when automatically recalculating energies from the
DCD file if the *_energies.txt file is missing. The choice of force field affects
the calculated energies and, consequently, trajectory diagnostics and all
subsequent analysis results.
Recommended value: True (for most tasks).
"""

TEMPERATURE_K = 300.0
"""
Unified system temperature in Kelvin.
Used as the thermostat temperature in the Langevin equation when generating
the conformation ensemble and simultaneously as the temperature in the Kramers
formula when building the transition rate matrix. The coincidence of these
temperatures ensures thermodynamic consistency of calculating free energies
and transition rates: the free energy of a cluster is calculated via its
population at the same temperature at which sampling was performed.
Physically sets the scale of thermal fluctuations and viscous friction, i.e.,
the effective temperature at which the canonical conformation ensemble is generated.
At low temperatures (~300 K) the system may get "stuck" in the native well and
not explore other regions of the landscape. At high temperatures (~500–1000 K)
the system overcomes energy barriers and visits a wide spectrum of conformations,
including partially and fully unfolded states.
WARNING: At temperatures above 1200 K, classical force fields may lose physical
adequacy (covalent bond breaks are possible). The program will automatically
reduce the integration step to prevent numerical instability (NaN), but results
at T > 1500 K should be interpreted with caution.
Recommended values:
300 K  — exploring the native well and nearest minima.
400–500 K — extended sampling of local minima.
800–1000 K — global exploration of the landscape, including unfolded states
(recommended for ultrametricity analysis).
Valid range: 200.0 – 2000.0 K.
"""

MD_STEPS = 10000000
"""
Total number of Langevin dynamics integration steps.
The actual integration step (in fs) is chosen automatically depending on the
temperature to ensure numerical stability (2 fs for T < 1000 K, 1 fs for
1000 <= T < 1500 K, 0.5 fs for T >= 1500 K). The total simulation time is
scaled accordingly.
The number of saved frames is MD_STEPS / MD_REPORT_INTERVAL. For a statistically
significant ensemble (~10,000 frames) with a standard save interval of 50,
~500,000 steps are required.
Recommended values:
500,000 — minimum length to obtain ~10,000 frames.
2,000,000 — recommended to increase statistical reliability.
Valid range: 10,000 – 100,000,000.
"""

MD_REPORT_INTERVAL = 100
"""
Interval for saving trajectory frames (in integration steps).
Every MD_REPORT_INTERVAL-th frame is written to the DCD file, and its potential
energy to the energy text file. Too frequent an interval leads to excessive
data volume and correlated frames; too rare leads to loss of information about
fast transitions.
Recommended value: 50 (provides ~10,000 frames for 500,000 steps).
Valid range: 10 – 10,000.
"""

MINIMIZE_STARTING_STRUCTURE = True
"""
Whether to perform local energy minimization of the starting structure before
running molecular dynamics.
Minimization eliminates steric clashes and reduces the initial potential energy.
This prevents the simulation from "exploding" in the first steps due to
unrealistically high forces.
Highly recommended True. Disabling (False) makes sense only if the starting
structure has already been minimized or if it is required to start dynamics
from a non-equilibrium configuration.
Recommended value: True.
"""

MD_SEED = 43
"""
Seed for the pseudo-random number generator for Langevin dynamics.
Used to initialize the random number generator in the Langevin integrator
and to set initial velocities. Ensures trajectory reproducibility upon repeated
runs with the same parameters.
To obtain different trajectories with all other parameters equal, change this
value. In MAIN_ANALYSIS mode, this seed is ignored, as NULL_SEED_BASE is used.
Recommended value: 42 (any integer).
"""

MD_TIME_STEP_FS = 2.0
"""
Initial Langevin dynamics integration step in femtoseconds.
Determines the base time step for numerical integration of the equations of motion.
A value of 2.0 fs is standard for systems with bond length constraints
(e.g., LINCS or SHAKE for bonds with hydrogen).
If numerical instability (NaN) occurs and the rollback limit is exceeded,
this step will be automatically reduced.
Recommended value: 2.0.
Valid range: 0.5 – 5.0.
"""

MD_ROLLBACK_STEPS = 5000
"""
Number of steps to roll back when numerical instability (NaN) occurs.
The simulation is broken into chunks of this size. After successful execution
of a chunk, the system's coordinates and velocities are saved to RAM. If a NaN
error occurs during integration, the program restores the saved state, recreates
the simulation object (to clear the CUDA context), and continues the simulation,
trying to overcome short-term force fluctuations.
Recommended value: 1000 (or a multiple of the frame save interval).
Valid range: 100 – 10000.
"""

MD_MAX_ROLLBACKS = 5
"""
Maximum number of CONSECUTIVE allowable rollbacks without reducing the integration step.
If the number of consecutive rollbacks exceeds this value, the program concludes
that there is systematic instability at the current step, reduces the integration
step by a factor of MD_TIME_STEP_REDUCTION_FACTOR, and starts the simulation
from scratch (from zero, after minimization).
A value of 0 means that rollbacks are not used, and at the first NaN the step
is reduced and the simulation is restarted (as in previous versions).
Recommended value: 5.
Valid range: 0 – 100.
"""

MD_TIME_STEP_REDUCTION_FACTOR = 2.0
"""
Integration step reduction factor when the rollback limit is exceeded.
If the number of rollbacks exceeds MD_MAX_ROLLBACKS, the current step is divided
by this number, and the simulation starts from scratch.
Recommended value: 2.0.
Valid range: 1.1 – 10.0.
"""

MD_MIN_TIME_STEP_FS = 0.25
"""
Minimum allowable integration step in femtoseconds.
If, as a result of consecutive reductions, the step falls below this value,
the simulation is deemed physically unstable and aborted with an error returned.
Recommended value: 0.25.
"""

# ----------------------------------------------------------------------------
# GROUP 3: MSM AND NEIGHBOR GRAPH CONSTRUCTION PARAMETERS
# ----------------------------------------------------------------------------
SCALE_F = 1.0
"""
Scaling factor for free energies.
The formula for calculating the free energy of a cluster takes the form:
F_i = -SCALE_F * RT * ln(N_i / N_total).
By default, it is 1.0, which corresponds to the standard Boltzmann distribution.
Changing this parameter allows you to investigate how the degree of landscape
ultrametricity depends on the effective scale of energy fluctuations (analogous
to changing the beta parameter in Theorem 2).
Increasing SCALE_F is equivalent to increasing the spread of free energies,
which should enhance ultrametricity.
Recommended value: 1.0 (for physical free energies).
Valid range: 0.1 – 10.0.
"""

NEIGHBOR_METHOD = 'adaptive_topology'
"""
Method for determining the neighborhood of microstates (clusters).
'energy'            — Energetic neighborhood. Two microstates are considered
neighbors if the difference in their free energies does not exceed the threshold
ΔF_max = -RT ln(eta). The criterion is invariant to energy shift and ensures
correct connectivity at all levels. Requires setting the ETA_NEIGHBOR_MODE
parameters and (in manual mode) ETA_NEIGHBOR.
'adaptive_topology' — Adaptive topological graph. For each cluster, an effective
radius is calculated (NEIGHBOR_RADIUS_PERCENTILE-th percentile of distances from
the center to the cluster points). Two clusters are considered neighbors if the
distance between their centers does not exceed (R_i + R_j). The method is
objective and does not require a manual distance threshold.
Recommended value: 'adaptive_topology'.
"""

ETA_NEIGHBOR_MODE = 'manual'
"""
Mode for selecting the η parameter for energetic neighborhood.
Used ONLY when NEIGHBOR_METHOD = 'energy'.
'manual' — The ETA_NEIGHBOR value specified by the user is used. Suitable when
there is a physical justification for a specific threshold of inter-cluster
transition rate.
'auto'   — Automatic selection of η based on graph topology analysis. The algorithm
iterates over η values from a logarithmic grid (30 points in the range [1e-12, 1e-2])
and selects the largest η (strictest threshold ΔF_max) at which the conditions
are met:
(1) there is at least one significant connected component (C_sig >= 1);
(2) fragmentation index f_inter <= 0.9.
If conditions are equal, the option with the maximum number of basins in the
largest component (K_max) is selected. This mode makes the η parameter entirely
data-driven and eliminates the need for heuristic selection.
Recommended value: 'auto' (for objective analysis).
'manual' (for comparison with a fixed threshold).
"""

ETA_NEIGHBOR = 1e-6
"""
Dimensionless threshold of inter-cluster transition rate for energetic neighborhood
(manual mode).
Used ONLY when NEIGHBOR_METHOD = 'energy' AND ETA_NEIGHBOR_MODE = 'manual'.
Two microstates i and j are considered neighbors if:
|F_i - F_j| <= ΔF_max,   where   ΔF_max = -RT ln(ETA_NEIGHBOR)
Physical interpretation: η sets the minimum significant transition rate between
microstates (in units of the frequency factor ν₀). The threshold ΔF_max is
calculated automatically from η and the Kramers temperature T.
Examples of ΔF_max values at T = 300 K (RT ≈ 2.49 kJ/mol):
η = 1e-6  →  ΔF_max ≈ 34.5 kJ/mol (≈ 13.8 k_B T)
η = 1e-4  →  ΔF_max ≈ 23.0 kJ/mol (≈ 9.2 k_B T)
η = 1e-8  →  ΔF_max ≈ 46.0 kJ/mol (≈ 18.4 k_B T)
WARNING: In versions < 2.8, the criterion max(F_i, F_j) <= -RT ln(η) was used,
which depended on the absolute energy value. The current criterion
|F_i - F_j| <= ΔF_max is physically correct and invariant.
Recommended value: 1e-6 (corresponds to ~14 k_B T at 300 K).
Valid range: 1e-12 – 1e-3.
"""

NEIGHBOR_RADIUS_PERCENTILE = 95
"""
Percentile for calculating the effective radius of a cluster (in percent).
Used ONLY when NEIGHBOR_METHOD = 'adaptive_topology'.
The effective radius of a cluster is defined as the specified percentile of
distances from the center of the cluster to all points that fell into this cluster.
A value of 95 means that 95% of the cluster points lie inside a sphere of this
radius. This allows cutting off single outliers (the 5% furthest points) and
correctly estimating the "sphere of influence" of the microstate in configuration
space.
Recommended value: 95.
Valid range: 50 – 99.
"""

GEOMETRY_RMSD_MODE = 'ca'
"""
Unified RMSD calculation mode for ALL geometric operations.
Determines the metric for both k-centers clustering and building the adaptive
topological graph. This ensures internal consistency of conformational space
discretization.
'ca'  — RMSD only by C-alpha atoms. Faster, recommended for most tasks.
Well reflects the overall folding of the chain.
'all' — Full-atom RMSD (all protein atoms). A stricter definition of proximity,
accounting for side chain conformations, but computationally more expensive.
WARNING: When choosing 'all', the time for clustering and graph building increases.
For long trajectories (>20,000 frames), 'ca' is recommended.
Recommended value: 'all'.
"""

N_CLUSTERS = 5000
"""
Target number of microstates (clusters) for discretizing the conformational space.
A large number of microstates is required for detailed resolution of the energy
landscape. The actual number of significant clusters may be less than N_CLUSTERS,
since clusters with a number of frames less than MIN_CLUSTER_SIZE are discarded.
Recommended value: 1000 (for a trajectory of ~10,000 frames).
For trajectories of ~40,000 frames, 4000 is recommended.
Valid range: 100 – 10,000.
"""

MIN_CLUSTER_SIZE = 5
"""
Minimum number of frames in a cluster to recognize it as a significant microstate.
Smaller clusters are considered statistically insignificant (noise) and are
excluded from further analysis.
Recommended value: 5 (with 10,000 frames and N_CLUSTERS = 1000, an average of
10 frames per cluster is expected; a threshold of 5 cuts off clusters with
insufficient statistics).
Valid range: 2 – 100.
"""

BASIN_EQ_THRESHOLD = 1.0
"""
Energy equivalence threshold (ε_eq) for determining plateaus and local minima
(in kJ/mol).
When determining points of attraction (basins) by the gradient descent method:
- If the difference in free energies of two neighboring microstates
|F_i - F_j| <= ε_eq, they are considered energetically equivalent and can belong
to the same plateau.
- If F_j < F_i - ε_eq, microstate j is considered strictly deeper, and gradient
descent goes into it.
A value of ~3.0 kJ/mol corresponds to approximately 1 k_B T at 300 K, which
allows merging degenerate minima, the energy difference between which does not
exceed thermal fluctuations.
Recommended value: 3.0 kJ/mol.
Valid range: 0.1 – 20.0 kJ/mol.
"""

BASIN_EQ_THRESHOLD = BASIN_EQ_THRESHOLD * SCALE_F
"""
Scaling factor for the energy equivalence threshold coefficient (ε_eq) for determining plateaus and local
minima (in kJ/mol).
"""

MIN_BASIN_SIZE = 20
"""
Minimum number of microstates in a basin of attraction.
Smaller basins are discarded as statistically insignificant.
This prevents the appearance of "basins" of one or two microstates, which may
be artifacts of clustering or insufficient sampling.
Recommended value: 5.
Valid range: 1 – 20.
"""

MIN_BASINS_PER_COMPONENT = 5
"""
Minimum number of basins of attraction in a connected component of the microstate
graph.
Connected components in which, after gradient descent and filtering, fewer basins
remain than this value are discarded as statistically insignificant. Ultrametricity
analysis requires at least three basins to calculate at least one triplet, but
to obtain statistically reliable conclusions, at least 10 basins (120 triplets)
are needed.
Recommended value: 10.
Valid range: 3 – 100.
"""

# ----------------------------------------------------------------------------
# GROUP 4: SPECTRAL ANALYSIS AND ULTRAMETRICITY TESTING PARAMETERS
# ----------------------------------------------------------------------------
NUM_EIGENMODES = 50
"""
Maximum number of eigenmodes for spectral decomposition of the K_sym matrix.
From this number, after automatic filtering of noise modes by the spectral gap
criterion, n_phys physical modes remain, which are used to calculate the kinetic
metric.
Recommended value: 50.
Valid range: 10 – 500.
"""

EIGS_SIGMA = 1e-4
"""
Shift sigma for the Lanczos algorithm (shift-invert) when searching for eigenvalues
of the K_sym matrix near zero.
The K_sym matrix is negative semi-definite (all λ ≤ 0). The shift-invert algorithm
(which='LM', sigma > 0) allows efficiently finding eigenvalues closest to sigma,
i.e., the smallest non-zero modes in absolute value — the slowest relaxation
processes, which make the greatest contribution to the kinetic metric.
Impact of sigma value on performance:
sigma = 1e-4 ... 1e-5 (recommended):
Enhances the diagonal dominance of the shifted matrix (K - sigma*I),
LU factorization (SuperLU) is performed in fractions of a second.
sigma < 1e-8:
The shifted matrix is almost singular, factorization can slow down by orders
of magnitude (tens of seconds) due to numerical instability and the need to
select a large number of pivots.
sigma > 1e-2:
The algorithm finds modes near sigma, which may not be the slowest; the result
may lose physical meaning.
Recommended value: 1e-4.
Valid range: 1e-6 – 1e-3.
"""

EIGS_TOL = 1e-8
"""
Convergence tolerance for the Lanczos algorithm (eigsh).
Determines the accuracy with which eigenvalues and eigenvectors are calculated.
A smaller value gives a more accurate result but requires more iterations.
Recommended value: 1e-8 (sufficient for most tasks).
Valid range: 1e-10 – 1e-6.
"""

EIGS_DENSE_THRESHOLD = 6000
"""
Matrix size threshold for switching to dense diagonalization.
If the number of microstates in the connected component is less than this threshold,
instead of the sparse Lanczos algorithm (eigsh), full diagonalization of the dense
matrix (np.linalg.eigh) is used. For small matrices, dense diagonalization is
faster and more reliable; for large ones, the sparse algorithm significantly saves
time.
Note: if the sparse algorithm does not converge (ArpackNoConvergence), the program
automatically switches to dense diagonalization regardless of this threshold.
Recommended value: 6000.
Valid range: 100 – 10000.
"""

SPECTRAL_GAP_THRESHOLD = 1e6
"""
Threshold of the ratio |λ_k| / |λ_{k-1}| for detecting a spectral gap.
Eigenvalues are sorted in ascending order of absolute value. If for some k the
ratio exceeds the threshold, all modes with indices 0..k-1 are considered numerical
noise and excluded from analysis. The remaining modes (k..m-1) are considered
physical.
Typical values: 1e4 – 1e8. A threshold of 1e6 reliably separates noise modes
(|λ| ~ 10^-16) from physical ones (|λ| ~ 10^-5 – 10^0).
Recommended value: 1e6.
"""

ULTRAMETRIC_EPSILON = 0.05
"""
Relative accuracy ε for testing the equality of the two largest sides of a triangle
when classifying triplets.
A triplet (x, y, z) with ordered distances d_min ≤ d_mid ≤ d_max is considered
non-trivially ultrametric if:
(d_max - d_mid) / d_mid ≤ ε   AND   (d_mid - d_min) / d_mid > δ
Must be strictly less than ULTRAMETRIC_DELTA.
Recommended value: 0.05 (5% relative error).
Valid range: 0.01 – 0.20.
"""

ULTRAMETRIC_DELTA = 0.1
"""
Minimum relative difference between the middle and smaller sides of a triangle
for classifying a triplet as non-trivially ultrametric.
Must be strictly greater than ULTRAMETRIC_EPSILON.
Recommended value: 0.1 (10% relative difference).
Valid range: 0.05 – 0.50.
"""

# ============================================================================
# BLAS/LAPACK/OPENMP THREAD LIMITATION (OPTIMIZED TO SPEED UP STAGE 2)
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
"""Universal gas constant in kcal/(mol K)."""
KB_KJ = 0.008314462618
"""Universal gas constant in kJ/(mol K)."""

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

    if verbose: print("  [PDBFixer] Loading and fixing structure...")
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
        if verbose: print(f"  [Modeller] Removing {len(non_protein_residues)} non-protein residues (ligands, water, etc.)...")
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
            if verbose: print(f"  [Modeller] Selecting chain '{longest_chain_id}'. Removing {len(residues_to_delete)} residues from other chains...")
            modeller.delete(residues_to_delete)

    if verbose: print(f"  [Modeller] Adding hydrogens taking into account the force field...")
    if use_gb99dms:
        gb_path = "GB99dms.xml"
        if not os.path.exists(gb_path):
            if verbose: print("  Downloading GB99dms.xml...")
            import urllib.request
            url = "https://raw.githubusercontent.com/greener-group/GB99dms/main/GB99dms.xml"
            urllib.request.urlretrieve(url, gb_path)
        ff = ForceField(gb_path)
    else:
        ff = ForceField("amber14-all.xml", "implicit/obc2.xml")

    modeller.addHydrogens(ff, pH=7.0)
    topology = modeller.topology
    positions = modeller.positions
    if verbose: print(f"  Final structure: {topology.getNumAtoms()} atoms, {sum(1 for _ in topology.residues())} residues")

    return topology, positions, ff

# ============================================================================
# LOADING TRAJECTORY AND ENERGIES FROM FILES (FOR FROM_FILES_ANALYSIS MODE)
# ============================================================================
def load_trajectory_and_energies_from_files(pdb_id, starting_structure_file,
                                            use_gb99dms, use_gpu,
                                            temperature_k, sample_idx=None, verbose=True):
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
            print(f"  FROM_FILES_ANALYSIS mode requires a DCD file.")
        return None, None, None
    if verbose: print(f"  Trajectory found: {dcd_file}")
    if verbose: print(f"  Topology file for mdtraj: {topology_file_for_mdtraj}")

    energy_file_exists = os.path.exists(energy_file)
    try:
        traj = md.load(dcd_file, top=topology_file_for_mdtraj)
        n_frames = traj.n_frames
        if verbose: print(f"  Trajectory loaded: {n_frames} frames, {traj.n_atoms} atoms")
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
                          f"energies have {len(energies)} values.")
                    print(f"  Energy file will be recalculated.")
                energy_file_exists = False
            else:
                if verbose:
                    print(f"  Energy file loaded: {energy_file}")
                    print(f"  Energies: min={energies.min():.1f}, max={energies.max():.1f}, "
                          f"mean={energies.mean():.1f} kJ/mol")
        except Exception as e:
            if verbose:
                print(f"\n[WARNING] Failed to read energy file: {type(e).__name__}")
                print(f"  Energy file will be recalculated.")
            energy_file_exists = False

    if not energy_file_exists:
        if verbose:
            print(f"\nRecalculating energies via OpenMM (this may take ~30-60 sec)...")
            print(f"  Force field: {'GB99dms' if use_gb99dms else 'Amber14/OBC2'}")
        try:
            topology_omm, _, ff = prepare_structure(topology_file_for_mdtraj, use_gb99dms, verbose)
        except ImportError:
            raise ImportError("pdbfixer must be installed: pip install pdbfixer")

        if topology_omm.getNumAtoms() != traj.n_atoms:
            if verbose:
                print(f"\n[ERROR] Number of atoms in cleaned structure ({topology_omm.getNumAtoms()}) "
                      f"does not match number of atoms in trajectory ({traj.n_atoms}).")
                print(f"  Make sure the DCD file was created by this program, or provide "
                      f"the corresponding *_topology.pdb file.")
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
# AUTOMATIC SELECTION OF ETA PARAMETER
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
        print("\nAutomatic selection of η:")
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
            print(f"    [WARNING] No η satisfies the criteria.")
            print(f"    Using the smallest η = {fallback_eta:.2e}")
            print(f"    (ΔF_max = {df_max_fallback:.1f} kJ/mol)")
        return fallback_eta

    candidates.sort(key=lambda x: (-x[0], -x[2]))
    best = candidates[0]
    best_eta = best[0]
    df_max_best = -RT * np.log(best_eta)
    if verbose:
        print(f"    Selected η = {best_eta:.2e}")
        print(f"    ΔF_max = {df_max_best:.1f} kJ/mol ({df_max_best/RT:.1f} RT)")
        print(f"    C_sig = {best[1]}, K_max = {best[2]}, f_inter = {best[3]:.4f}")
        print(f"    Total suitable candidates: {len(candidates)} out of {len(eta_grid)}")

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
        if verbose: print(f"\nLoading custom structure: {input_file}")
    else:
        local_cif = f"{pdb_id}.cif"
        local_pdb = f"{pdb_id}.pdb"
        input_file = None
        if os.path.exists(local_cif):
            input_file = local_cif
            if verbose: print(f"\nLocal CIF found: {local_cif}")
        elif os.path.exists(local_pdb):
            input_file = local_pdb
            if verbose: print(f"\nLocal PDB found: {local_pdb}")
        else:
            if verbose:
                print(f"\nLocal files not found. Downloading {pdb_id} from RCSB PDB...")
            import urllib.request
            cif_url = f"https://files.rcsb.org/download/{pdb_id}.cif"
            download_path = os.path.abspath(local_cif)
            try:
                urllib.request.urlretrieve(cif_url, download_path)
                input_file = download_path
                if verbose: print(f"  CIF successfully downloaded: {download_path}")
            except Exception:
                if verbose: print(f"  CIF download failed, trying PDB...")
                pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                download_path = os.path.abspath(local_pdb)
                urllib.request.urlretrieve(pdb_url, download_path)
                input_file = download_path
                if verbose: print(f"  PDB successfully downloaded: {download_path}")

    try:
        topology, positions, ff = prepare_structure(input_file, use_gb99dms, verbose)
    except ImportError:
        raise ImportError("pdbfixer must be installed: pip install pdbfixer")

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
            if verbose: print("  Platform: CUDA (mixed precision)")
        except Exception as e:
            if verbose:
                print(f"  [WARNING] CUDA initialization error (missing nvrtc DLL or driver): {type(e).__name__}")
                print(f"             Automatic fallback to CPU.")
            platform = openmm.Platform.getPlatformByName("CPU")
            props = {"Threads": "1"}
            if verbose: print("  Platform: CPU (fallback)")
    else:
        platform = openmm.Platform.getPlatformByName("CPU")
        props = {"Threads": "1"}
        if verbose: print("  Platform: CPU (GPU disabled by user)")

    current_dt_fs = md_time_step_fs

    base_name = os.path.splitext(os.path.basename(starting_structure_file))[0] if starting_structure_file and os.path.exists(starting_structure_file) else pdb_id
    suffix = f"_{sample_idx}" if sample_idx is not None else ""

    final_dcd_file = f"{base_name}_trajectory{suffix}.dcd"
    energy_file = f"{base_name}_energies{suffix}.txt"
    pdb_file = f"{base_name}{suffix}.pdb"

    with open(pdb_file, 'w') as f:
        app.PDBFile.writeFile(topology, positions, f)
    if verbose: print(f"  Cleaned topology saved: {pdb_file}")

    for f in [final_dcd_file, energy_file]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

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
                if verbose: print("  Minimizing energy of starting structure...")
                sim.minimizeEnergy(maxIterations=2000)

            sim.context.setVelocitiesToTemperature(temp_k * unit.kelvin, seed)

            reporter = MemoryReporter(report_interval)
            sim.reporters.append(reporter)

            total_time_ps = n_steps * dt
            n_frames = n_steps // report_interval

            print(f"  Starting MD: {n_steps} steps ({total_time_ps:.1f} ps, step {current_dt_fs} fs), "
                  f"~{n_frames} frames will be saved")

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
                    pos_np = state.getPositions(asNumpy=True)
                    vel_np = state.getVelocities(asNumpy=True)

                    if np.isnan(pos_np).any() or np.isnan(vel_np).any():
                        if verbose:
                            print(f"\n[ROLLBACK] Hidden NaNs detected in successful step {total_steps_done}. Initiating rollback.")
                        raise ValueError("Hidden NaNs detected in getState")

                    resume_pos = pos_np
                    resume_vel = vel_np
                    frames_saved = len(reporter.positions)
                    consecutive_rollbacks = 0

                except Exception as e:
                    is_nan_error = "NaN" in str(e) or isinstance(e, openmm.OpenMMException) or isinstance(e, ValueError)
                    if is_nan_error:
                        consecutive_rollbacks += 1
                        if consecutive_rollbacks > md_max_rollbacks:
                            if verbose:
                                print(f"\n[WARNING] Consecutive rollback limit exceeded ({md_max_rollbacks}).")
                                print(f"             Reducing step by a factor of {md_time_step_reduction_factor} and restarting.")
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
                if verbose: print("  WARNING: Position/energy lists are empty!")

            return final_dcd_file, pdb_file, energy_file

        except Exception as e:
            if verbose:
                print(f"\n[WARNING] Critical error: {type(e).__name__}")
                print(f"             Message: {str(e)[:150]}")
            try:
                sim.reporters = []
                del sim, integrator
            except:
                pass
            gc.collect()
            return None, None, None

# ============================================================================
# K-CENTERS CLUSTERING
# ============================================================================
class ClusterKCenters:
    def __init__(self, traj, n_clusters, atom_indices):
        self.n_clusters = n_clusters
        if atom_indices is not None:
            if len(atom_indices) == 0 or len(atom_indices) == traj.n_atoms:
                atom_indices = None
            else:
                traj = traj.atom_slice(atom_indices)
                atom_indices = None
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
        print(f"  Energetic neighborhood: ΔF_max = {delta_f_max:.1f} kJ/mol "
              f"(η={eta:.0e}, T={RT/KB_KJ:.0f} K)")
        print(f"  Edges in graph: {n_edges} out of {n_states*(n_states-1)//2} possible")

    return adjacency

def build_adaptive_topological_graph(traj, n_states, radius_percentile,
                                     rmsd_mode,
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
        if len(atom_indices) == 0 or len(atom_indices) == traj.n_atoms:
            atom_indices = None
    else:
        atom_indices = None

    if verbose:
        n_atoms = len(atom_indices) if atom_indices is not None else traj.n_atoms
        print(f"  Adaptive topological graph: calculating effective radii")
        print(f"    percentile: {radius_percentile}%, atoms: {n_atoms}")

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
        print(f"  Building graph: d(c_i, c_j) ≤ (R_i + R_j)")

    adjacency = np.zeros((n_states, n_states), dtype=bool)
    for i in range(n_states):
        rmsds_to_i = md.rmsd(
            centers_traj, centers_traj, i,
            atom_indices=atom_indices, precentered=False
        )
        thresholds =  radii[i] + radii
        neighbors_mask = rmsds_to_i <= thresholds
        neighbors_mask[i] = False
        adjacency[i, neighbors_mask] = True
        adjacency[neighbors_mask, i] = True

    n_edges = np.sum(adjacency) // 2
    if verbose:
        print(f"  Edges in graph: {n_edges} out of {n_states*(n_states-1)//2} possible")

    return adjacency

def build_msm_and_graph(dcd_file, pdb_file, energy_file, n_clusters,
                        min_cluster_size, temp_k, neighbor_method, eta,
                        radius_percentile, rmsd_mode,
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
            pass
        elif neighbor_method == 'energy':
            print(f"  Eta (rate threshold): {eta:.0e}")
            print(f"  Temperature (unified): {temp_k} K")

    traj = md.load(dcd_file, top=pdb_file)
    n_frames = traj.n_frames

    if energy_file is None or not os.path.exists(energy_file):
        raise ValueError("Energy file not found.")
    energies = np.loadtxt(energy_file)
    if len(energies) != n_frames:
        raise ValueError(
            f"Mismatch in number of frames: trajectory {n_frames}, energies {len(energies)}"
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
        raise ValueError(f"Too few C-alpha atoms: {len(cluster_atom_indices)} (need ≥ 3)")

    if rmsd_mode == 'ca':
        traj_rmsd = traj.atom_slice(cluster_atom_indices)
        cluster_atom_indices = None
    else:
        traj_rmsd = traj

    if verbose:
        print(f"  k-centers clustering: k={n_clusters}, atoms: {atom_desc}")

    t0 = time.time()
    cluster_obj = ClusterKCenters(traj_rmsd, n_clusters, cluster_atom_indices)
    labels = cluster_obj.assign(traj_rmsd)
    original_center_indices = cluster_obj.center_indices
    elapsed = time.time() - t0

    if verbose:
        print(f"  Clustering completed in {elapsed:.1f} sec")

    unique, counts = np.unique(labels, return_counts=True)
    valid_clusters = unique[counts >= min_cluster_size]
    if len(valid_clusters) < 3:
        if verbose:
            print(f"  ERROR: Significant clusters {len(valid_clusters)} (need ≥ 3).")
        return None, None, None, None

    label_map = {old: new for new, old in enumerate(valid_clusters)}
    filtered_labels = np.array(
        [label_map[l] if l in label_map else -1 for l in labels]
    )
    n_states = len(valid_clusters)

    if verbose:
        print(f"  Significant microstates: {n_states} "
              f"(discarded {len(unique) - n_states} sparsely populated)")

    RT = KB_KJ * temp_k
    free_energies = np.zeros(n_states)
    for i in range(n_states):
        counts = np.sum(filtered_labels == i)
        free_energies[i] = -SCALE_F * RT * np.log(max(counts, 1) / n_frames)
    free_energies -= free_energies.min()

    if verbose:
        print(f"  Free energies (relat.): min={free_energies.min():.1f}, "
              f"max={free_energies.max():.1f} kJ/mol")

    if neighbor_method == 'energy':
        adjacency = build_neighbor_graph_energy(free_energies, RT, eta, verbose=verbose)
    elif neighbor_method == 'adaptive_topology':
        adjacency = build_adaptive_topological_graph(
            traj_rmsd, n_states, radius_percentile, rmsd_mode,
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
        print(f"  Matrix K_sym: {n_states}×{n_states}, "
              f"non-zero off-diagonal elements: {n_nonzero}")
        print(f"  Number of connected components of neighbor graph: {n_components}")

    del traj, traj_rmsd
    gc.collect()

    return K_sym, adjacency, free_energies, n_states

# ============================================================================
# ANALYSIS OF CONNECTED COMPONENTS AND BASINS
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
    w = np.exp(-free_energies_c / RT)

    k_request = min(num_modes + 1, n_micro - 1)
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
        except ArpackNoConvergence as e:
            if not quiet:
                print(f"    [WARNING] ARPACK did not converge. Switching to dense diagonalization.")
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
    n_phys = len(phys_eigenvalues)

    if n_phys == 0:
        return None
    if not quiet:
        print(f"    Physical modes: {n_phys} (noise discarded: {num_noise})")

    chi = np.zeros((n_basins_c, n_micro), dtype=np.float64)
    for a in range(n_basins_c):
        mask = (basin_labels_c == a)
        if np.sum(mask) > 0:
            W_a = np.sum(w[mask])
            if W_a > 1e-300:
                chi[a, mask] = np.sqrt(w[mask]) / W_a

    proj = chi @ phys_eigenvectors
    weights = 1.0 / np.abs(phys_eigenvalues)
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

        status_str = "ACCEPTED" if n_basins_c >= min_basins_per_comp else "DISCARDED"

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
        print(f"\n{'='*145}")
        print("DIAGNOSTICS OF BASIN FORMATION (first "+str(N_COMP_CON)+" connected components):")
        print(f"{'='*145}")
        print(f"{'No.':<4} | {'Micro.':<6} | {'Edges':<8} | {'Loc. min.':<11} | {'Plateaus':<14} | {'Basins':<14} | {'Basins':<15} | {'F_max':<8} | {'F_min':<8} | {'F_mean±std':<14} | Status")
        print(f"{'':<4} | {'':<6} | {'':<8} | {'(pre-merge)':<11} | {'(post-merge)':<14} | {'(post-desc)':<14} | {'(post-filter)':<15} | {'(kJ/mol)':<8} | {'(kJ/mol)':<8} | {'(kJ/mol)':<14} |")
        print("-" * 145)
        limit = min(len(all_components_stats), N_COMP_CON)
        for i in range(limit):
            s = all_components_stats[i]
            fe_str = f"{s['fe_mean']:.1f}±{s['fe_std']:.1f}"
            print(f"{s['comp_idx']+1:<4} | {s['n_micro_c']:<6} | {s['n_edges_c']:<8} | "
                  f"{s['n_attractors']:<11} | {s['n_plateaus']:<14} | {s['n_basins_before_filter']:<14} | "
                  f"{s['n_basins_after_filter']:<15} | {s['fe_max']:<8.1f} | {s['fe_min']:<8.1f} | {fe_str:<14} | {s['status']}")
        if len(all_components_stats) > 10:
            print(f"... (showing 10 out of {len(all_components_stats)} components)")
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
        print(f"\nTOTAL BY COMPONENTS:")
        print(f"  Total connected components: {n_components_total}")
        print(f"  Significant components: {n_significant_components}")
        print(f"  Total basins in significant components: {total_basins_all}")
        if n_significant_components > 1:
            print(f"  Fragmentation index f_inter: {f_inter:.4f}")
            if f_inter > 0.8:
                print(f"  [WARNING] High fragmentation. Results reflect")
                print(f"  hierarchy within components, not the global landscape.")
        elif n_significant_components == 1:
            print(f"  Fragmentation index f_inter: 0.0000 (all basins in one component)")
        else:
            print(f"  Fragmentation index f_inter: not calculated (no significant basins)")

    weighted_u_nt = 0.0
    weighted_u_tr = 0.0
    weighted_u_non = 0.0
    total_weight = 0
    total_physical_modes = 0
    largest_D = None
    max_micro = -1

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
            comp_data['distance_matrix'] = res['distance_matrix']
            if comp_data['n_micro_c'] > max_micro:
                max_micro = comp_data['n_micro_c']
                largest_D = res['distance_matrix']

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
        print(f"\nTABLE OF SIGNIFICANT CONNECTED COMPONENTS:")
        header = (f"{'No.':<4} | {'Clusters':<9} | {'Edges':<8} | {'Triplets':<8} | "
                  f"{'u_nt (%)':<9} | {'u_tr (%)':<9} | {'u_non (%)':<9}")
        print(header)
        print("-" * len(header))
        limit = min(n_significant_components, 10)
        for i in range(limit):
            cd = valid_components_data[i]
            u_nt_str = f"{cd.get('u_nt', 0.0):.2f}" if cd.get('u_nt') is not None else "—"
            u_tr_str = f"{cd.get('u_tr', 0.0):.2f}" if cd.get('u_tr') is not None else "—"
            u_non_str = f"{cd.get('u_non', 0.0):.2f}" if cd.get('u_non') is not None else "—"
            print(f"{cd['comp_idx']+1:<4} | {cd['n_micro_c']:<9} | {cd['n_edges_c']:<8} | {cd['triplets_c']:<8} | "
                  f"{u_nt_str:<9} | {u_tr_str:<9} | {u_non_str:<9}")
        if n_significant_components > 10:
            print(f"... (showing 10 out of {n_significant_components} significant components)")
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
            failure_reason = "neighbor graph does not contain any connected components"
        elif total_valid_basins_all_components == 0:
            failure_reason = f"no basins of size >= {min_basin_size} found in any component"
        elif max_basins_in_any_component < min_basins_per_comp:
            failure_reason = (f"maximum number of basins in a component ({max_basins_in_any_component}) "
                              f"is less than the threshold MIN_BASINS_PER_COMPONENT ({min_basins_per_comp})")
        else:
            failure_reason = "no significant connected components found"
    elif total_weight == 0:
        if total_basins_all < 3:
            failure_reason = f"total number of basins in all significant components ({total_basins_all}) is less than 3 (impossible to build triplets)"
        else:
            failure_reason = "failed to compute kinetic metric for any triplet of basins (possibly all modes filtered as noise)"

    if failure_reason and verbose:
        print(f"\n[ERROR REASON] Analysis yielded no result: {failure_reason}.")

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
        'fe_std_valid': fe_std_valid,
        'largest_D': largest_D
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
    if verbose:
        print("  Verbose output: for each EQ value, component tables will be printed.")
    print("=" * 70)
    header = (f"{'EQ (kJ/mol)':<12} | {'f_inter':<8} | {'Comp.':<6} | "
              f"{'Basins':<6} | {'u_nt (%)':<10} | {'u_tr (%)':<10} | {'u_non (%)':<10} | {'Triplets':<8}")
    print(header)
    print("-" * len(header))

    for eq_thresh in sens_list:
        if verbose:
            print("\n" + "=" * 70)
            print(f"DETAILED ANALYSIS FOR EQ = {eq_thresh:.2f} kJ/mol")
            print("=" * 70)

        res = analyze_landscape_components(
            K_sym, adjacency, free_energies, n_micro, temp_k,
            eq_thresh, min_basin_size, min_basins_per_comp,
            num_modes, eigs_sigma, eigs_tol, dense_threshold,
            gap_threshold, eps, delta,
            verbose=verbose
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
            if res is None:
                reason = "analysis function returned None"
            else:
                reason = res.get('failure_reason', "no significant triplets")
            print(f"  [SKIP] EQ={eq_thresh:.2f}: {reason}")
            print(f"{eq_thresh:<12.2f} | {'—':<8} | {'—':<6} | "
                  f"{'—':<6} | {'—':>8}   | {'—':>8}   | {'—':>8}   | {'—':>6}")
    print("-" * len(header))

# ============================================================================
# GENERAL FUNCTIONS FOR TRAJECTORY PROCESSING AND MSM CONSTRUCTION
# ============================================================================
def process_trajectory(dcd_file, pdb_file, energy_file, n_clusters, min_cluster_size, temp_k, rmsd_mode, verbose=True, traj_obj=None, energies_obj=None):
    import mdtraj as md
    if traj_obj is None:
        if verbose: print(f"  Loading and clustering: {dcd_file}")
        traj = md.load(dcd_file, top=pdb_file)
        energies = np.loadtxt(energy_file)
    else:
        if verbose: print(f"  Clustering combined trajectory ({traj_obj.n_frames} frames)...")
        traj = traj_obj
        energies = energies_obj

    if rmsd_mode == 'ca':
        cluster_atom_indices = [a.index for a in traj.topology.atoms if a.name == 'CA']
        if len(cluster_atom_indices) < 3:
            if verbose:
                print(f"  [WARNING] Too few C-alpha atoms: {len(cluster_atom_indices)} (need ≥ 3).")
            del traj
            gc.collect()
            return None
        traj_rmsd = traj.atom_slice(cluster_atom_indices)
        cluster_atom_indices = None
    else:
        traj_rmsd = traj
        cluster_atom_indices = None

    cluster_obj = ClusterKCenters(traj_rmsd, n_clusters, cluster_atom_indices)
    labels = cluster_obj.assign(traj_rmsd)
    original_center_indices = cluster_obj.center_indices

    unique, counts = np.unique(labels, return_counts=True)
    valid_clusters = unique[counts >= min_cluster_size]

    if len(valid_clusters) < 3:
        if verbose: print(f"  [WARNING] Significant clusters {len(valid_clusters)} (need ≥ 3).")
        del traj, traj_rmsd
        gc.collect()
        return None

    label_map = {old: new for new, old in enumerate(valid_clusters)}
    filtered_labels = np.array([label_map[l] if l in label_map else -1 for l in labels])
    n_states = len(valid_clusters)

    n_frames = traj.n_frames
    RT = KB_KJ * temp_k
    free_energies = np.zeros(n_states)
    for i in range(n_states):
        counts = np.sum(filtered_labels == i)
        free_energies[i] = -SCALE_F * RT * np.log(max(counts, 1) / n_frames)
    free_energies -= free_energies.min()

    if verbose: print(f"  Microstates: {n_states}, F: [{free_energies.min():.1f}, {free_energies.max():.1f}] kJ/mol")

    if traj_rmsd is not traj:
        del traj
        gc.collect()

    return {
        'traj': traj_rmsd,
        'pdb_file': pdb_file,
        'energy_file': energy_file,
        'n_states': n_states,
        'original_center_indices': original_center_indices,
        'valid_clusters': valid_clusters,
        'filtered_labels': filtered_labels,
        'free_energies': free_energies,
        'RT': RT
    }

def build_msm_from_sd(sd, eta, neighbor_method, radius_percentile, rmsd_mode, verbose=True):
    if neighbor_method == 'energy':
        adjacency = build_neighbor_graph_energy(sd['free_energies'], sd['RT'], eta, verbose=verbose)
    else:
        adjacency = build_adaptive_topological_graph(
            sd['traj'], sd['n_states'], radius_percentile, rmsd_mode,
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
# MAIN ANALYSIS (MAIN_ANALYSIS)
# ============================================================================
def run_main_analysis(samples_data, temperature_k, basin_eq_threshold, min_basin_size,
                      min_basins_per_comp, num_modes, eigs_sigma, eigs_tol, eigs_dense_threshold,
                      spectral_gap_threshold, ultrametric_epsilon, ultrametric_delta,
                      neighbor_method, eta_neighbor, eta_neighbor_mode,
                      radius_percentile, rmsd_mode,
                      verbose=True):
    print("\n" + "=" * 70)
    print("MAIN ANALYSIS (MAIN_ANALYSIS)")
    print(f"  Analyzing {len(samples_data)} independent trajectories with base parameters.")
    if verbose:
        print("  Verbose output: separate tables will be printed for each trajectory.")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

    u_nt_list = []
    u_tr_list = []
    u_non_list = []
    n_basins_list = []
    f_inter_list = []
    failure_reasons = []
    skipped_samples = []
    sample_results = []

    for i, sd in enumerate(samples_data):
        sample_idx = sd.get('sample_idx', i + 1)
        if sample_idx is None:
            sample_idx = i + 1

        if verbose:
            print("\n" + "=" * 70)
            print(f"DETAILED TRAJECTORY ANALYSIS: seed={sample_idx}")
            print("=" * 70)

        effective_eta = eta_neighbor
        if neighbor_method == 'energy' and eta_neighbor_mode == 'auto':
            effective_eta = auto_select_eta_neighbor(
                sd['free_energies'], sd['RT'], min_basin_size,
                min_basins_per_comp, basin_eq_threshold,
                verbose=verbose
            )

        K_sym, adjacency = build_msm_from_sd(
            sd, effective_eta, neighbor_method, radius_percentile, rmsd_mode, verbose=verbose
        )

        results = analyze_landscape_components(
            K_sym, adjacency, sd['free_energies'], sd['n_states'],
            temperature_k,
            basin_eq_threshold, min_basin_size, min_basins_per_comp,
            num_modes, eigs_sigma, eigs_tol, eigs_dense_threshold,
            spectral_gap_threshold,
            ultrametric_epsilon, ultrametric_delta,
            verbose=verbose
        )

        n_edges = np.sum(adjacency) // 2
        n_possible_edges = sd['n_states'] * (sd['n_states'] - 1) // 2

        if results is None:
            reason = "analysis function returned None"
        elif results['total_triplets'] <= 0:
            reason = results.get('failure_reason', 'unknown reason')
        else:
            reason = None

        if verbose:
            print(f"\nINDIVIDUAL RESULT TABLE FOR TRAJECTORY seed={sample_idx}:")
            ind_header = f"{'Metric':<45} | {'Value':<30}"
            print(ind_header)
            print("-" * len(ind_header))
            if reason is None:
                print(f"{'Status':<45} | {'SUCCESS':<30}")
                print(f"{'Number of microstates':<45} | {sd['n_states']:<30}")
                print(f"{'Edges in graph':<45} | {n_edges:<30}")
                print(f"{'Possible edges':<45} | {n_possible_edges:<30}")
                print(f"{'Connected components (total)':<45} | {results['n_total_components']:<30}")
                print(f"{'Significant components':<45} | {results['n_valid_components']:<30}")
                print(f"{'Basins in significant components':<45} | {results['total_basins']:<30}")
                print(f"{'f_inter':<45} | {results['f_inter']:<30.4f}")
                print(f"{'u_nt, %':<45} | {results['u_nontrivial']:<30.2f}")
                print(f"{'u_tr, %':<45} | {results['u_trivial']:<30.2f}")
                print(f"{'u_non, %':<45} | {results['u_nonultrametric']:<30.2f}")
                print(f"{'Triplets':<45} | {int(results['total_triplets']):<30}")

                fe_mean_val = results.get('fe_mean_valid', np.nan)
                if not np.isnan(fe_mean_val):
                    print(f"{'F_mean (valid), kJ/mol':<45} | {fe_mean_val:<30.1f}")

                fe_std_val = results.get('fe_std_valid', np.nan)
                if not np.isnan(fe_std_val):
                    print(f"{'STD_F (valid), kJ/mol':<45} | {fe_std_val:<30.1f}")
                    rt_val = KB_KJ * TEMPERATURE_K
                    print(f"{'STD_F/RT':<45} | {fe_std_val/rt_val:<30.2f}")
            else:
                print(f"{'Status':<45} | {'SKIPPED':<30}")
                print(f"{'Reason':<45} | {reason:<30}")
            print("-" * len(ind_header))

        if reason is None:
            u_nt_list.append(results['u_nontrivial'])
            u_tr_list.append(results['u_trivial'])
            u_non_list.append(results['u_nonultrametric'])
            n_basins_list.append(results['total_basins'])
            f_inter_list.append(results['f_inter'])

            sample_results.append({
                'f_inter': results['f_inter'],
                'n_valid_components': results['n_valid_components'],
                'total_basins': results['total_basins'],
                'u_nontrivial': results['u_nontrivial'],
                'u_trivial': results['u_trivial'],
                'u_nonultrametric': results['u_nonultrametric'],
                'total_triplets': results['total_triplets'],
                'n_states': sd['n_states'],
                'fe_min': np.min(sd['free_energies']),
                'fe_max': np.max(sd['free_energies']),
                'n_edges': n_edges,
                'n_possible_edges': n_possible_edges,
                'n_components': results['n_total_components'],
                'fe_mean_valid': results.get('fe_mean_valid', np.nan),
                'fe_std_valid': results.get('fe_std_valid', np.nan)
            })

            if METRIC_OUTPUT and results.get('largest_D') is not None:
                fname = os.path.join(script_dir, f"seed_{sample_idx}")
                np.save(fname, results['largest_D'])
                if verbose:
                    print(f"  [METRIC_OUTPUT] Matrix saved: {fname}.npy")

            if verbose:
                print(f"  [MAIN] Trajectory {sample_idx}: u_nt={results['u_nontrivial']:.2f}%, "
                      f"u_tr={results['u_trivial']:.2f}%, u_non={results['u_nonultrametric']:.2f}%")
        else:
            failure_reasons.append(reason)
            skipped_samples.append((sample_idx, reason))
            print(f"  [SKIP] Trajectory {sample_idx}: failed to compute result. Reason: {reason}")

    print(f"  MAIN_ANALYSIS: successful trajectories {len(u_nt_list)} out of {len(samples_data)}")
    if skipped_samples:
        print(f"  Skipped samples:")
        for sid, reason in skipped_samples:
            print(f"    seed={sid}: {reason}")

    if len(u_nt_list) == 0:
        print("\nERROR: All MAIN_ANALYSIS trajectories yielded no result.")
        if failure_reasons:
            unique_reasons = list(set(failure_reasons))
            print("Error reasons by trajectories:")
            for r in unique_reasons:
                count = failure_reasons.count(r)
                print(f"  - {r} (in {count} trajectories)")
        return None, None

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
        'n_valid': len(u_nt_list), 'elapsed': 0,
        'skipped_samples': skipped_samples
    }

    return stats_dict, sample_results

def print_main_analysis_results(results_main, verbose=True):
    if results_main is None:
        print("\nMAIN_ANALYSIS: no data to output.")
        return
    print("\n" + "=" * 70)
    print("SUMMARY TABLE OF MAIN ANALYSIS RESULTS (MAIN_ANALYSIS)")
    print(f"  (Details by trajectories)")
    print("=" * 70)
    header = f"{'Trajectory':<12} | {'u_nt (%)':<10} | {'u_tr (%)':<10} | {'u_non (%)':<10} | {'Basins':<6} | {'f_inter':<8}"
    print(header)
    print("-" * len(header))

    for i in range(len(results_main['u_nt_list'])):
        print(f"{i+1:<12} | {results_main['u_nt_list'][i]:>8.2f}   | "
              f"{results_main['u_tr_list'][i]:>8.2f}   | "
              f"{results_main['u_non_list'][i]:>8.2f}   | "
              f"{results_main['n_basins_list'][i]:>5} | "
              f"{results_main['f_inter_list'][i]:>7.4f}")
    print("-" * len(header))

    skipped = results_main.get('skipped_samples', [])
    if skipped:
        print(f"\nSkipped trajectories ({len(skipped)}):")
        for sid, reason in skipped:
            print(f"  seed={sid}: {reason}")

    mean_nt = results_main['mean_nt']
    std_nt = results_main['std_nt']
    cv = (std_nt / max(abs(mean_nt), 0.01)) * 100

    print(f"\nFINAL STATISTICS AND STABILITY REPORT:")
    print(f"  Number of successful trajectories:   {results_main['n_valid']}")
    print(f"  Mean u_nt:                           {mean_nt:.2f} %")
    if results_main['n_valid'] > 1:
        print(f"  Standard deviation (STD):            {std_nt:.2f} %")
        print(f"  Coefficient of variation (CV):       {cv:.1f} %")
        if cv < 5.0:
            print(f"  CONCLUSION: Result is HIGHLY STABLE (CV < 5%).")
        elif cv < 10.0:
            print(f"  CONCLUSION: Result is stable (5% <= CV < 10%).")
        elif cv < 25.0:
            print(f"  CONCLUSION: Result is moderately stable (10% <= CV < 25%).")
        else:
            print(f"  CONCLUSION: Result is UNSTABLE (CV >= 25%). Increase trajectory length.")
    else:
        print("  (STD and CV are not calculated, as analysis was performed for a single combined trajectory)")
    print("=" * 70)

def print_aggregate_tables(sample_results, samples_data, temperature_k):
    if not sample_results:
        return
    agg = {}
    for key in ['f_inter', 'n_valid_components', 'total_basins', 'u_nontrivial',
                'u_trivial', 'u_nonultrametric', 'total_triplets', 'n_edges',
                'n_components', 'fe_mean_valid', 'fe_std_valid']:
        values = [r[key] for r in sample_results if not np.isnan(r[key])]
        if len(values) > 0:
            agg[f'{key}_mean'] = np.mean(values)
            agg[f'{key}_std'] = np.std(values) if len(values) > 1 else 0.0
        else:
            agg[f'{key}_mean'] = np.nan
            agg[f'{key}_std'] = np.nan

    agg['n_states'] = sample_results[0]['n_states']
    agg['fe_min'] = np.mean([r['fe_min'] for r in sample_results])
    agg['fe_max'] = np.mean([r['fe_max'] for r in sample_results])
    agg['n_possible_edges'] = sample_results[0]['n_possible_edges']

    triplets_values = [r['total_triplets'] for r in sample_results if not np.isnan(r['total_triplets'])]
    if len(triplets_values) > 0:
        agg['total_triplets_median'] = np.median(triplets_values)
        agg['total_triplets_min'] = np.min(triplets_values)
        agg['total_triplets_max'] = np.max(triplets_values)
    else:
        agg['total_triplets_median'] = np.nan
        agg['total_triplets_min'] = np.nan
        agg['total_triplets_max'] = np.nan

    agg['n_valid_samples'] = len(sample_results)
    agg['n_total_samples'] = len(samples_data)

    def fmt(val, std, width):
        if np.isnan(val):
            return f"{'—':<{width}}"
        if len(samples_data) > 1 and std > 0:
            return f"{val:.2f}±{std:.2f}".center(width)
        return f"{val:.2f}".center(width)

    header1 = (f"{'f_inter':<8} | {'Comp.(sig.)':<12} | "
               f"{'Basins':<12} | {'u_nt (%)':<14} | {'u_tr (%)':<14} | {'u_non (%)':<14} | {'Triplets':<12} | {'Median triplets':<14} | {'[Min, Max]':<18} | {'Success':<12}")
    print("\n" + header1)
    print("-" * len(header1))
    r = agg
    f_inter_str = fmt(r['f_inter_mean'], r['f_inter_std'], 8)
    n_valid_str = fmt(r['n_valid_components_mean'], r['n_valid_components_std'], 12)
    basins_str = fmt(r['total_basins_mean'], r['total_basins_std'], 12)
    u_nt_str = fmt(r['u_nontrivial_mean'], r['u_nontrivial_std'], 14)
    u_tr_str = fmt(r['u_trivial_mean'], r['u_trivial_std'], 14)
    u_non_str = fmt(r['u_nonultrametric_mean'], r['u_nonultrametric_std'], 14)
    triplets_str = fmt(r['total_triplets_mean'], r['total_triplets_std'], 12)
    if not np.isnan(r.get('total_triplets_median', np.nan)):
        med_trip_str = f"{int(r['total_triplets_median'])}"
        min_max_str = f"[{int(r['total_triplets_min'])}, {int(r['total_triplets_max'])}]"
    else:
        med_trip_str = "—"
        min_max_str = "—"

    success_str = f"{r.get('n_valid_samples', 0)}/{r.get('n_total_samples', len(samples_data))}"

    print(f"{f_inter_str} | {n_valid_str} | {basins_str} | {u_nt_str} | {u_tr_str} | {u_non_str} | {triplets_str} | {med_trip_str:<14} | {min_max_str:<18} | {success_str:<12}")
    print("-" * len(header1))

    n_valid = r.get('n_valid_samples', len(samples_data))
    n_total = r.get('n_total_samples', len(samples_data))
    if n_valid < n_total:
        print(f"[WARNING] Statistics calculated over {n_valid} successful trajectories out of {n_total}.")

    print("\n" + "=" * 70)
    print("GRAPH AND FREE ENERGY STATISTICS")
    if len(samples_data) > 1:
        print("  (values: mean ± STD across trajectories)")
        print("  The table includes a 'Success' column: number of successful trajectories / total number of trajectories")
    print("=" * 70)

    header2 = (f"{'Micro.':<6} | {'F_min':<8} | {'F_max':<8} | "
               f"{'Edges':<14} | {'Poss. edges':<12} | {'Comp.(total)':<14} | {'F_mean±std (valid)':<20} | {'STD_F±std (valid)':<20} | {'STD_F/RT±std':<14} | {'Success':<12}")
    print(header2)
    print("-" * len(header2))

    fe_min_str = f"{r['fe_min']:.1f}" if not np.isnan(r['fe_min']) else "—"
    fe_max_str = f"{r['fe_max']:.1f}" if not np.isnan(r['fe_max']) else "—"

    if len(samples_data) > 1 and not np.isnan(r['n_edges_mean']):
        edges_str = f"{r['n_edges_mean']:.0f}±{r['n_edges_std']:.0f}"
    elif not np.isnan(r['n_edges_mean']):
        edges_str = f"{r['n_edges_mean']:.0f}"
    else:
        edges_str = "—"

    if len(samples_data) > 1 and not np.isnan(r['n_components_mean']):
        comp_str = f"{r['n_components_mean']:.1f}±{r['n_components_std']:.1f}"
    elif not np.isnan(r['n_components_mean']):
        comp_str = f"{r['n_components_mean']:.0f}"
    else:
        comp_str = "—"

    if not np.isnan(r['fe_mean_valid_mean']):
        if len(samples_data) > 1 and r['fe_mean_valid_std'] > 0:
            fe_valid_str = f"{r['fe_mean_valid_mean']:.1f}±{r['fe_mean_valid_std']:.1f}"
        else:
            fe_valid_str = f"{r['fe_mean_valid_mean']:.1f}"
    else:
        fe_valid_str = "—"

    if not np.isnan(r.get('fe_std_valid_mean', np.nan)):
        if len(samples_data) > 1 and r.get('fe_std_valid_std', 0) > 0:
            fe_std_str = f"{r['fe_std_valid_mean']:.1f}±{r['fe_std_valid_std']:.1f}"
        else:
            fe_std_str = f"{r['fe_std_valid_mean']:.1f}"
    else:
        fe_std_str = "—"

    RT_val = KB_KJ * temperature_k
    std_f_mean_rt = r['fe_std_valid_mean'] / RT_val if not np.isnan(r.get('fe_std_valid_mean', np.nan)) else np.nan
    std_f_std_rt = r.get('fe_std_valid_std', 0) / RT_val if not np.isnan(r.get('fe_std_valid_mean', np.nan)) else np.nan
    if len(samples_data) > 1 and std_f_std_rt > 0:
        std_f_rt_str = f"{std_f_mean_rt:.2f}±{std_f_std_rt:.2f}"
    elif not np.isnan(std_f_mean_rt):
        std_f_rt_str = f"{std_f_mean_rt:.2f}"
    else:
        std_f_rt_str = "—"

    n_states_print = r['n_states']
    if not np.isnan(n_states_print):
        n_states_print = f"{int(n_states_print)}"
    else:
        n_states_print = "—"

    poss_edges_print = r['n_possible_edges']
    if not np.isnan(poss_edges_print):
        poss_edges_print = f"{int(poss_edges_print)}"
    else:
        poss_edges_print = "—"

    print(f"{n_states_print:<6} | {fe_min_str:<8} | {fe_max_str:<8} | "
          f"{edges_str:<14} | {poss_edges_print:<12} | {comp_str:<14} | {fe_valid_str:<20} | {fe_std_str:<20} | {std_f_rt_str:<14} | {success_str:<12}")
    print("-" * len(header2))

# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    v = VERBOSE
    print("=" * 70)
    print("ULTRAMETRICITY ANALYSIS OF PROTEIN ENERGY LANDSCAPE")
    print("Kinetic metric based on spectral decomposition (v1.22)")
    print("=" * 70)
    print(f"  Protein:              {PDB_ID}")
    print(f"  Force field:          {'GB99dms' if USE_GB99DMS else 'Amber14/OBC2'}")
    print(f"  Temperature (unified): {TEMPERATURE_K} K")
    print(f"  MD steps:             {MD_STEPS}, save interval: {MD_REPORT_INTERVAL}")
    print(f"  MD step (initial):    {MD_TIME_STEP_FS} fs, rollback: {MD_ROLLBACK_STEPS} steps")
    print(f"  Neighborhood method:  {NEIGHBOR_METHOD}")
    if NEIGHBOR_METHOD == 'energy':
        print(f"  η mode:               {ETA_NEIGHBOR_MODE}")
        if ETA_NEIGHBOR_MODE == 'manual':
            RT_print = KB_KJ * TEMPERATURE_K
            df_max_print = -RT_print * np.log(ETA_NEIGHBOR)
            print(f"  eta (manual):         {ETA_NEIGHBOR:.0e} → ΔF_max = {df_max_print:.1f} kJ/mol")
    elif NEIGHBOR_METHOD == 'adaptive_topology':
        print(f"  Radius percentile:    {NEIGHBOR_RADIUS_PERCENTILE}%")
        print(f"  RMSD metric:          {'C-alpha' if GEOMETRY_RMSD_MODE == 'ca' else 'all atoms'}")
    print(f"  Microstates:          {N_CLUSTERS} (target), min size: {MIN_CLUSTER_SIZE}")
    print(f"  SCALE_F:              {SCALE_F} (scaling factor for free energies)")
    print(f"  Plateau threshold (EQ): {BASIN_EQ_THRESHOLD} kJ/mol, min basin size: {MIN_BASIN_SIZE}")
    print(f"  Min basins in comp.:  {MIN_BASINS_PER_COMPONENT}")
    print(f"  Modes:                {NUM_EIGENMODES} (gap threshold: {SPECTRAL_GAP_THRESHOLD:.0e})")
    print(f"  EIGS sigma:           {EIGS_SIGMA:.0e}, tol: {EIGS_TOL:.0e}, dense_threshold: {EIGS_DENSE_THRESHOLD}")
    print(f"  Ultrametricity:       ε={ULTRAMETRIC_EPSILON}, δ={ULTRAMETRIC_DELTA}")
    print(f"  CPU threads:          {num_cores} (limit: {MAX_CPU_THREADS})")
    print(f"  Matrix output (npy):  {'Yes' if METRIC_OUTPUT else 'No'}")
    print("-" * 70)
    print("  SELECTED OPERATING MODES:")
    print(f"  FROM_FILES_ANALYSIS:  {'Yes' if FROM_FILES_ANALYSIS else 'No'}")
    print(f"  FROM_FILES_POOL:      {'Yes' if FROM_FILES_POOL_ENABLE else 'No'}")
    print(f"  Sensitivity analysis: {'Yes (list: ' + str(SENSITIVITY_RANGE) + ')' if SENSITIVITY_ANALYSIS else 'No'}")
    print(f"  MAIN_ANALYSIS:        {'Yes (' + str(MAIN_NUM_SAMPLES) + ')' if MAIN_ANALYSIS else 'No'}")
    print(f"  GPU:                  {'Yes' if USE_GPU else 'No'}")
    print(f"  Verbose:              {'Yes' if v else 'No'}")
    print("=" * 70)

    sens_active = SENSITIVITY_ANALYSIS and isinstance(SENSITIVITY_RANGE, list) and len(SENSITIVITY_RANGE) > 0
    main_active = MAIN_ANALYSIS

    if not (sens_active or main_active):
        print("\n" + "="*70)
        print("ALL ANALYTICAL MODES ARE DISABLED.")
        print("Set at least one of the flags:")
        print("  SENSITIVITY_ANALYSIS = True (with non-empty SENSITIVITY_RANGE)")
        print("  MAIN_ANALYSIS = True")
        print("Terminating program.")
        print("="*70)
        return None

    total_start = time.time()

    base_name = os.path.splitext(os.path.basename(STARTING_STRUCTURE_FILE))[0] if STARTING_STRUCTURE_FILE and os.path.exists(STARTING_STRUCTURE_FILE) else PDB_ID
    samples_data = []

    if FROM_FILES_ANALYSIS:
        print("\n" + "="*70)
        print(">>> FROM_FILES_ANALYSIS MODE: LOADING DATA FROM DISK <<<")
        print("="*70)

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

        if FROM_FILES_POOL_ENABLE:
            print("\n" + "="*70)
            print(">>> FROM_FILES_POOL_ENABLE MODE: COMBINING ALL TRAJECTORIES <<<")
            print("="*70)
            import mdtraj as md
            trajs = []
            all_energies = []
            for sample_idx in samples:
                display_idx = sample_idx if sample_idx is not None else 1
                if v: print(f"\n--- Loading sample {display_idx} for pooling ---")

                dcd_file, pdb_file, energy_file = load_trajectory_and_energies_from_files(
                    PDB_ID, STARTING_STRUCTURE_FILE, USE_GB99DMS, USE_GPU,
                    TEMPERATURE_K, sample_idx=sample_idx, verbose=v
                )
                if dcd_file is None:
                    print(f"  [SKIP] Sample {display_idx}: failed to load.")
                    continue
                try:
                    t = md.load(dcd_file, top=pdb_file)
                    e = np.loadtxt(energy_file)
                    if len(e) != t.n_frames:
                        print(f"  [WARNING] Mismatch of frames and energies in sample {display_idx}. Skipping.")
                        continue
                    trajs.append(t)
                    all_energies.append(e)
                    if v: print(f"  Sample {display_idx} loaded: {t.n_frames} frames.")
                except Exception as e:
                    print(f"  [ERROR] Failed to read sample {display_idx}: {type(e).__name__}")
                    continue

            if not trajs:
                print("\n[ERROR] Failed to load any trajectory for pooling.")
                return None

            # FIXED IN 1.17: variable renamed from 'time' to 'time_arr'
            # to avoid shadowing the time module.
            xyz = np.concatenate([t.xyz for t in trajs])
            time_arr = np.concatenate([t.time for t in trajs]) if trajs[0].time is not None and len(trajs[0].time) > 0 else None
            unitcell_lengths = np.concatenate([t.unitcell_lengths for t in trajs]) if trajs[0].unitcell_lengths is not None else None
            unitcell_angles = np.concatenate([t.unitcell_angles for t in trajs]) if trajs[0].unitcell_angles is not None else None

            pooled_traj = md.Trajectory(
                xyz=xyz,
                topology=trajs[0].topology,
                time=time_arr,
                unitcell_lengths=unitcell_lengths,
                unitcell_angles=unitcell_angles
            )
            pooled_energies = np.concatenate(all_energies)

            print(f"\nTOTAL COMBINED: {len(trajs)} trajectories, {pooled_traj.n_frames} frames.")

            sd = process_trajectory(
                dcd_file=None, pdb_file=None, energy_file=None,
                n_clusters=N_CLUSTERS, min_cluster_size=MIN_CLUSTER_SIZE,
                temp_k=TEMPERATURE_K, rmsd_mode=GEOMETRY_RMSD_MODE,
                verbose=v, traj_obj=pooled_traj, energies_obj=pooled_energies
            )
            if sd is not None:
                sd['sample_idx'] = 'pooled'
                samples_data.append(sd)
            else:
                print("  [ERROR] Combined trajectory failed clustering.")

        else:
            for sample_idx in samples:
                display_idx = sample_idx if sample_idx is not None else 1
                if v and sample_idx is not None:
                    print(f"\n--- Loading and clustering sample {sample_idx} ---")

                dcd_file, pdb_file, energy_file = load_trajectory_and_energies_from_files(
                    PDB_ID, STARTING_STRUCTURE_FILE, USE_GB99DMS, USE_GPU,
                    TEMPERATURE_K, sample_idx=sample_idx, verbose=v
                )
                if dcd_file is None:
                    print(f"  [SKIP] Sample {display_idx}: failed to load trajectory or energies.")
                    continue

                sd = process_trajectory(dcd_file, pdb_file, energy_file, N_CLUSTERS, MIN_CLUSTER_SIZE, TEMPERATURE_K, GEOMETRY_RMSD_MODE, verbose=v)
                if sd is not None:
                    sd['sample_idx'] = sample_idx if sample_idx is not None else 1
                    samples_data.append(sd)
                else:
                    print(f"  [SKIP] Sample {display_idx} discarded at clustering stage (insufficient significant clusters).")

            if len(samples_data) < len(samples):
                print(f"[WARNING] Out of {len(samples)} detected samples, only {len(samples_data)} were prepared.")

    else:
        num_sims_to_run = MAIN_NUM_SAMPLES if main_active else 1
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
                print(f"  [SKIP] Simulation {i}: failed to complete MD.")
                continue

            sd = process_trajectory(dcd_file, pdb_file, energy_file, N_CLUSTERS, MIN_CLUSTER_SIZE, TEMPERATURE_K, GEOMETRY_RMSD_MODE, verbose=v)
            if sd is not None:
                sd['sample_idx'] = sample_idx if sample_idx is not None else 1
                samples_data.append(sd)
            else:
                print(f"  [SKIP] Simulation {i} discarded at clustering stage (insufficient significant clusters).")

        if len(samples_data) < num_sims_to_run:
            print(f"[WARNING] Out of {num_sims_to_run} planned simulations, only {len(samples_data)} were prepared.")

    if not samples_data:
        print("\n[ERROR] No successful trajectories for analysis.")
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
            sd0, effective_eta, NEIGHBOR_METHOD, NEIGHBOR_RADIUS_PERCENTILE, GEOMETRY_RMSD_MODE, verbose=v
        )
        run_sensitivity_analysis(
            K_sym=K_sym,
            adjacency=adjacency,
            free_energies=sd0['free_energies'],
            n_micro=sd0['n_states'],
            temp_k=TEMPERATURE_K,
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

    if main_active:
        if v:
            print("\n" + "="*70)
            print(">>> LAUNCHING MODE: MAIN ANALYSIS (MAIN_ANALYSIS) <<<")
            print("="*70)
        results_main, sample_results_main = run_main_analysis(
            samples_data=samples_data,
            temperature_k=TEMPERATURE_K,
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
            radius_percentile=NEIGHBOR_RADIUS_PERCENTILE,
            rmsd_mode=GEOMETRY_RMSD_MODE,
            verbose=v
        )
        if results_main is not None:
            print_main_analysis_results(results_main, verbose=v)
            print_aggregate_tables(sample_results_main, samples_data, TEMPERATURE_K)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"TOTAL EXECUTION TIME: {total_elapsed:.1f} sec ({total_elapsed/60:.1f} min)")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()
