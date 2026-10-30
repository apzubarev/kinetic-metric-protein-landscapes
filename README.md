ULTRAMETRIC ANALYSIS OF PROTEIN ENERGY LANDSCAPES
BASED ON KINETIC METRICS AND MARKOV STATE MODELS (MSM)
=====================================================

Author: A. P. Zubarev
Version: 1.22
Affiliation: Volga State Transport University; Samara University
Contact: apzubarev@mail.ru


OVERVIEW
--------

This program implements a computational method for testing the hypothesis
of ultrametric organization of protein energy landscapes. The method is
based on analyzing the kinetics of transitions between macrostates
(basins of attraction) using a kinetic metric constructed via spectral
decomposition of the symmetrized Kramers transition rate matrix and the
Mahalanobis distance in the space of relaxation modes.

The key idea is that the kinetic metric is NOT automatically ultrametric.
Therefore, a high degree of ultrametricity detected by this metric serves
as a meaningful property of the energy landscape, reflecting genuine
hierarchical organization of free energy barriers.

The program performs the following computational stages:

- Stage 1: Molecular dynamics (Langevin dynamics) using OpenMM
- Stage 2: Construction of Markov State Models (MSM) and neighborhood graphs
- Stage 3: Spectral analysis and ultrametricity testing

The method has been validated on the mini-protein Trp-cage (PDB: 1L2Y,
20 residues) at temperatures 300 K and 400 K with ten independent
trajectories for each temperature.


VERSION 1.22 NOTES
------------------

Version 1.22 is intended to accompany the paper on the kinetic metric
for protein energy landscapes. Compared with version 1.21:

- The ALPHA_ANALYSIS mode and the NEIGHBOR_ALPHA parameter are not used.
- The adaptive topological graph uses the effective radius overlap rule
  d(c_i, c_j) <= R_i + R_j.
- Kinetic metric matrices are saved as seed_N.npy, or seed_pooled.npy
  in pooled mode, when METRIC_OUTPUT is enabled.
- The BASIN_EQ_THRESHOLD value is internally multiplied by SCALE_F.
- In FROM_FILES_ANALYSIS mode, missing energy files are recalculated
  automatically with the selected force field.


THEORETICAL BACKGROUND
----------------------

The kinetic metric between basins of attraction is defined through the
spectral decomposition of the symmetrized transition rate matrix S and
the Mahalanobis distance in the space of physical relaxation modes.

For each basin B_a, a conditional equilibrium distribution vector p_a is
introduced. The physical projection of p_a onto the k-th relaxation mode
is computed as the scalar product with the left eigenvector. The squared
kinetic distance between two basins is defined as the weighted Euclidean
norm of the difference of projections, with weights 1/abs(lambda_k),
where lambda_k are the non-zero eigenvalues of S.

In compact form:

  D^2(B_a, B_b) = sum_k ((pi_ak - pi_bk)^2 / abs(lambda_k))

The sum is taken over physical relaxation modes remaining after spectral
gap filtering.

The symmetrized off-diagonal rate elements are

  S_ij = u_0 exp(-abs(F_i - F_j) / (2RT))

for neighboring microstates. The diagonal elements enforce conservation
of probability. The spectrum of S is real and non-positive.

Theorem 1 (proved in the companion paper): The squared kinetic metric
is proportional to the symmetrized combination of mean first passage
times between basins:

  D^2(B_a, B_b) = (tau_ab + tau_ba - tau_aa - tau_bb) / Z

where Z is the total partition function of the system.

Theorem 2 (proved for random Erdos-Renyi graphs): In the limit of
large energy spread (beta -> infinity), the logarithmic asymptotics of
the kinetic metric converges pointwise to the single-linkage ultrametric.

Ultrametricity is tested by enumerating all triplets of basins. For a
triplet, the three distances are ordered as

  d_min <= d_med <= d_max

A triplet is classified as trivially ultrametric if all three distances
are approximately equal. A triplet is classified as nontrivially
ultrametric if the two largest distances are approximately equal and the
smallest distance is significantly smaller. In the program, this means:

  (d_max - d_med) / d_med <= epsilon
  (d_med - d_min) / d_med > delta

The fraction of nontrivially ultrametric triplets is denoted u_nt and
reported as a percentage number. The program also reports u_tr, the
fraction of trivially ultrametric triplets, and u_non, the fraction of
non-ultrametric triplets. These three values sum to 100 percent.


INSTALLATION AND REQUIREMENTS
-----------------------------

Execution environment: Windows 10/11, Linux, macOS
Python version: 3.10 - 3.12

Required packages (install via pip):

  pip install openmm pdbfixer mdtraj numpy scipy threadpoolctl torch

Hardware requirements:

  - GPU (NVIDIA CUDA) is strongly recommended for molecular dynamics.
    The program automatically falls back to CPU if CUDA is unavailable.
  - MSM construction and spectral analysis stages are performed on CPU.
  - Minimum 8 GB RAM recommended for trajectories with 5000 microstates.
  - Disk space: approximately 200 MB per trajectory (DCD and energies).


USER PARAMETERS
---------------

All configurable parameters are grouped at the top of the script with
detailed docstrings. Below is a summary of the main parameter groups.

GROUP 1: EXECUTION MODES AND GENERAL SETTINGS

  MAX_CPU_THREADS
    Maximum number of CPU threads for computations (OpenMP, BLAS, LAPACK).
    Set to 0 for automatic detection of all available cores.
    Default: 20.
    Recommended: 20-22 for a 24-thread processor.

  FROM_FILES_ANALYSIS
    If True, the program skips molecular dynamics and loads existing
    trajectory files from disk. Files are searched by sample index:
    {PDB_ID}_trajectory_N.dcd, {PDB_ID}_N.pdb or {PDB_ID}_N.cif,
    {PDB_ID}_energies_N.txt. If no indexed files are found, a single
    file without index is also accepted. If the energy file is missing,
    energies are recalculated automatically.
    If False, the program runs the full cycle starting from MD.
    Default in the supplied script: True.
    Recommended: False for primary calculation, True for re-analysis.

  FROM_FILES_POOL_ENABLE
    If True (and FROM_FILES_ANALYSIS is True), all found trajectories
    are merged into one large pooled trajectory. Analysis is performed
    as for a single sample without computing standard deviations.
    If METRIC_OUTPUT is True, the matrix is saved as seed_pooled.npy.
    Default: False.
    Recommended: False for statistical analysis across independent runs.

  SENSITIVITY_ANALYSIS
    Enable sensitivity analysis to the energy equivalence threshold
    BASIN_EQ_THRESHOLD. The list of values is given by SENSITIVITY_RANGE.
    An empty list disables the mode even if the flag is True.
    This mode can be combined with MAIN_ANALYSIS. If both are active,
    sensitivity analysis is executed first on the first prepared sample.
    Default: False.

  SENSITIVITY_RANGE
    List of BASIN_EQ_THRESHOLD values (in kJ/mol) for sensitivity analysis.
    Default: [1.0, 3.0, 5.0, 7.0]

  MAIN_ANALYSIS
    Enable Main Analysis mode. The full computational cycle
    (MD -> clustering -> MSM -> analysis) is executed MAIN_NUM_SAMPLES
    times with different random seeds. Mean, standard deviation, and
    coefficient of variation are computed for stability assessment.
    In FROM_FILES_ANALYSIS mode, the number of samples is determined
    automatically from available files.
    Default: True.

  MAIN_NUM_SAMPLES
    Number of independent runs for MAIN_ANALYSIS.
    Default: 10.
    Ignored in FROM_FILES_ANALYSIS mode because the number of samples is
    determined from files on disk.
    Recommended: 5-10 (compromise between reliability and time).

  NULL_SEED_BASE
    Base seed for random number generation. Each realization uses
    seed = NULL_SEED_BASE + realization_number.
    Default: 12345.

  USE_GPU
    If True, the program attempts to use NVIDIA GPU for MD.
    Automatic fallback to CPU on CUDA initialization failure.
    If False, CPU is used at all stages.
    Default: True.

  VERBOSE
    If True, detailed output is printed for each trajectory including
    neighborhood graph diagnostics, basin formation tables, and
    individual result tables.
    If False, only summary tables and mandatory skip diagnostics are
    printed.
    Default: True.

  N_COMP_CON
    Number of largest connected components to display in console tables.
    Default: 10.

  METRIC_OUTPUT
    If True, kinetic metric matrices (Mahalanobis distance matrices)
    are saved as numpy .npy files for the analyzed connected component
    with the largest number of microstates.
    File naming: seed_N.npy for ordinary samples and seed_pooled.npy
    for pooled analysis.
    Default: False.


GROUP 2: MOLECULAR DYNAMICS PARAMETERS

  PDB_ID
    Protein identifier in the RCSB Protein Data Bank (4-character code).
    Example: "1L2Y" for Trp-cage.
    Used as the base name for file search in FROM_FILES_ANALYSIS mode
    if STARTING_STRUCTURE_FILE is empty.

  STARTING_STRUCTURE_FILE
    Path to a custom starting structure file (PDB or CIF format).
    If empty, the structure is downloaded from RCSB PDB by PDB_ID.
    In FROM_FILES_ANALYSIS mode, if this file exists, its base name is
    used for searching trajectory and energy files.

  ADD_HYDROGENS_TO_CUSTOM_START
    Whether to add hydrogen atoms to the custom starting structure.
    Set True if the custom file contains only heavy atoms.
    Default: False.

  USE_GB99DMS
    If True, use the GB99dms force field with implicit solvent.
    If False, use Amber14-all with OBC2 implicit solvent.
    GB99dms is optimized for both ordered and disordered proteins.
    The GB99dms.xml file is downloaded automatically from the official
    repository on first use.
    Default: True.

  TEMPERATURE_K
    Single system temperature in Kelvin. Used both as the Langevin
    thermostat temperature and as the Kramers temperature for
    transition rate calculation, ensuring thermodynamic consistency.
    Recommended: 300 K for native state, 400-500 K for extended sampling,
    800-1000 K for global landscape exploration.
    Default: 300.0.

  MD_STEPS
    Total number of Langevin dynamics integration steps.
    Default: 10000000.
    For quick tests, 500000 may be sufficient.
    For publication-grade sampling, use 10000000 or longer.

  MD_REPORT_INTERVAL
    Frame saving interval in integration steps.
    Default: 100. With 10000000 steps this gives 100000 frames.
    For shorter runs, 50 is a common choice.

  MINIMIZE_STARTING_STRUCTURE
    Perform local energy minimization before starting MD.
    Strongly recommended: True.
    Default: True.

  MD_SEED
    Seed for the pseudo-random number generator in Langevin dynamics.
    Ignored in MAIN_ANALYSIS mode (NULL_SEED_BASE is used instead).
    Default: 43.

  MD_TIME_STEP_FS
    Initial integration step in femtoseconds.
    Standard: 2.0 fs. Automatically reduced upon numerical instability.
    Default: 2.0.

  MD_ROLLBACK_STEPS
    Number of steps for rollback upon NaN detection.
    Default: 5000.

  MD_MAX_ROLLBACKS
    Maximum consecutive rollbacks before reducing the integration step.
    Default: 5.

  MD_TIME_STEP_REDUCTION_FACTOR
    Factor by which the step is reduced when rollback limit is exceeded.
    Default: 2.0.

  MD_MIN_TIME_STEP_FS
    Minimum allowed integration step. Below this, simulation is aborted.
    Default: 0.25.


GROUP 3: MSM AND NEIGHBORHOOD GRAPH PARAMETERS

  SCALE_F
    Scaling factor for free energies:
    F_i = -SCALE_F RT ln(N_i / N_total).
    Default 1.0 corresponds to standard Boltzmann distribution.
    Increasing SCALE_F is analogous to increasing beta in Theorem 2.
    The effective BASIN_EQ_THRESHOLD is multiplied by SCALE_F.

  NEIGHBOR_METHOD
    Method for determining microstate neighborhood.
    'energy' - Energetic criterion: neighbors if abs(F_i - F_j) <= dF_max.
    'adaptive_topology' - Adaptive topological graph based on effective radii.
    Recommended: 'adaptive_topology'.
    Default: 'adaptive_topology'.

  ETA_NEIGHBOR_MODE
    Mode for selecting eta in energetic neighborhood.
    Used only when NEIGHBOR_METHOD is 'energy'.
    'manual' - User-specified ETA_NEIGHBOR value.
    'auto' - Automatic selection based on graph topology analysis.
    In auto mode, eta is chosen from a logarithmic grid so that at least
    one significant connected component exists and the fragmentation
    index f_inter is not larger than 0.9.
    Default: 'manual'.

  ETA_NEIGHBOR
    Dimensionless intercluster transition rate threshold (manual mode).
    Used only when NEIGHBOR_METHOD is 'energy' and ETA_NEIGHBOR_MODE
    is 'manual'.
    dF_max = -RT ln(ETA_NEIGHBOR).
    Default: 1e-6.

  NEIGHBOR_RADIUS_PERCENTILE
    Percentile for computing effective cluster radius.
    Used only when NEIGHBOR_METHOD is 'adaptive_topology'.
    Default and recommended: 95.

  GEOMETRY_RMSD_MODE
    Unified RMSD calculation mode for all geometric operations.
    'ca' - RMSD over C-alpha atoms only (faster).
    'all' - Full-atom RMSD (stricter, more expensive).
    Default: 'ca'.
    The same mode is used for clustering and graph construction.

  N_CLUSTERS
    Target number of microstates for k-centers clustering.
    Default: 5000.
    Recommended: 1000 for approximately 10000 frames,
    4000-5000 for approximately 40000-100000 frames.

  MIN_CLUSTER_SIZE
    Minimum number of frames in a cluster to be considered significant.
    Default: 5.

  BASIN_EQ_THRESHOLD
    Energy equivalence threshold (in kJ/mol) for defining plateaus
    and local minima during gradient descent.
    Default: 1.0.
    The effective value used by the program is multiplied by SCALE_F.
    Recommended physical range: 1.0-3.0 kJ/mol before scaling.

  MIN_BASIN_SIZE
    Minimum number of microstates in an attraction basin.
    Default: 20.

  MIN_BASINS_PER_COMPONENT
    Minimum number of basins in a connected component for analysis.
    Default: 5.
    Recommended: 5-10.


GROUP 4: SPECTRAL ANALYSIS AND ULTRAMETRICITY PARAMETERS

  NUM_EIGENMODES
    Maximum number of eigenmodes for spectral decomposition.
    Default and recommended: 50.

  EIGS_SIGMA
    Shift sigma for the Lanczos algorithm (shift-invert mode).
    Default and recommended: 1e-4.

  EIGS_TOL
    Convergence tolerance for the Lanczos algorithm.
    Default and recommended: 1e-8.

  EIGS_DENSE_THRESHOLD
    Matrix size threshold for switching to dense diagonalization.
    Default and recommended: 6000.

  SPECTRAL_GAP_THRESHOLD
    Threshold for the ratio abs(lambda_k)/abs(lambda_k-1) to detect
    a spectral gap and separate physical modes from numerical noise.
    Default and recommended: 1e6.

  ULTRAMETRIC_EPSILON
    Relative precision for testing equality of the two largest sides
    of a triangle in triplet classification.
    Default and recommended: 0.05 (5 percent relative error).

  ULTRAMETRIC_DELTA
    Minimum relative difference between the middle and smaller sides
    for nontrivial ultrametric classification.
    Must be strictly greater than ULTRAMETRIC_EPSILON.
    Default and recommended: 0.1 (10 percent relative difference).


COMPUTATIONAL PIPELINE
----------------------

STAGE 1: MOLECULAR DYNAMICS

  The program generates an equilibrium conformational ensemble using
  Langevin dynamics with the OpenMM package. The starting structure is
  automatically cleaned: non-protein residues (ligands, ions, water) are
  removed, the longest polypeptide chain is selected, and missing
  atoms/hydrogens are added.

  Instead of writing directly to disk, the program uses a custom
  MemoryReporter that accumulates coordinates and energies in Python
  lists. This solves the problem of garbage frames when NaN values occur:
  lists in RAM can be atomically truncated, whereas binary DCD files
  cannot be rolled back. The DCD file is written to disk only once at
  the very end, after successful completion of the simulation.

  If the number of rollbacks exceeds MD_MAX_ROLLBACKS, the integration
  step is automatically reduced and the simulation is restarted.

STAGE 2: MSM CONSTRUCTION

  The continuous conformational space is discretized by clustering
  trajectory frames into microstates using the k-centers algorithm in
  the RMSD metric. For each microstate, the free energy is computed via
  cluster population:

    F_i = -RT ln(N_i / N_total)

  If SCALE_F is not equal to one, the formula becomes

    F_i = -SCALE_F RT ln(N_i / N_total)

  The neighborhood graph is constructed using one of two methods:

  - Energetic: neighbors if abs(F_i - F_j) <= dF_max
  - Adaptive topological: neighbors if d(c_i, c_j) <= R_i + R_j

  Macrostates (basins of attraction) are identified by gradient descent
  on the free energy landscape. Attractors are identified first, then
  equivalence analysis is performed among attractors only, and finally
  gradient descent is applied to unlabeled slope states.

STAGE 3: SPECTRAL ANALYSIS

  For each significant connected component, spectral decomposition of
  the symmetrized rate matrix is performed. Noise modes are filtered
  using the spectral gap criterion. Characteristic vectors of basins
  are projected onto physical eigenmodes. The Mahalanobis distance
  between basins is computed with weights 1/abs(lambda_k).

  Ultrametricity is tested by enumerating all triplets of basins and
  classifying them as trivially ultrametric, nontrivially ultrametric,
  or non-ultrametric. The final u_nt is computed as a weighted average
  across connected components. The weights are proportional to the
  number of triplets in each component.


EXECUTION MODES
---------------

The program operates in one or more of the following modes:

  FROM_FILES_ANALYSIS
    Skip MD, load existing files, cluster them, and run analysis.

  FROM_FILES_POOL_ENABLE
    Merge all loaded trajectories into one pooled trajectory.
    This mode is used only together with FROM_FILES_ANALYSIS.

  SENSITIVITY_ANALYSIS
    Run analysis for several BASIN_EQ_THRESHOLD values.

  MAIN_ANALYSIS
    Analysis with base parameters for all trajectories. Computes mean,
    standard deviation, and coefficient of variation.

IMPORTANT: At least one analytical mode must be active. A non-empty
SENSITIVITY_RANGE is required for SENSITIVITY_ANALYSIS. If both
SENSITIVITY_ANALYSIS and MAIN_ANALYSIS are active, the program first
runs sensitivity analysis on the first prepared sample and then runs
main analysis on all prepared samples.


OUTPUT DESCRIPTION
------------------

The program produces the following outputs:

  Console output:
  - Summary tables of ultrametricity metrics (u_nt, u_tr, u_non)
  - Connected component diagnostics and basin formation tables
  - Fragmentation index f_inter
  - Statistics of graph properties (edges, microstates, free energies)
  - Stability report (mean, standard deviation, coefficient of variation)
  - Mandatory diagnostics of skipped trajectories with reasons

  File output:
  - DCD trajectory files: {PDB_ID}_trajectory_N.dcd
  - Energy files: {PDB_ID}_energies_N.txt
  - Topology files: {PDB_ID}_N.pdb
  - Kinetic metric matrices (if METRIC_OUTPUT=True): seed_N.npy or
    seed_pooled.npy


ROBUST STRUCTURE PREPARATION
-----------------------------

Before starting MD, the structure is automatically processed:

  1. Non-protein residues (ligands, ions, water) are removed
  2. The longest polypeptide chain is selected
  3. Missing atoms and hydrogen atoms are added (pH 7.0)
  4. Local energy minimization is performed

This ensures compatibility with OpenMM force fields for any PDB entries,
including structures with multiple chains, ligands, or modified residues.


DIAGNOSTICS OF SKIPPED TRAJECTORIES
------------------------------------

The program always (regardless of the VERBOSE value) outputs information
about trajectories that were excluded from the analysis. For each value
of the analyzed parameter, the number of successful trajectories and a
list of skipped samples with the reason for skipping are printed.

A "Success" column is added to result tables, showing the ratio of
successful trajectories to the total number of loaded or generated
trajectories. This allows unambiguous interpretation of statistical
values as calculated only on successful trajectories.


KEY RESULTS FOR TRP-CAGE (PDB: 1L2Y)
-------------------------------------

At T = 300 K (native regime):
  Mean u_nt = 59.6 +/- 11.7 percent
  Mean number of basins = 26.1 +/- 11.8
  STD_F / RT = 0.74 +/- 0.06
  Fragmentation index f_inter = 0.06

At T = 400 K (unfolding regime):
  Mean u_nt = 32.0 +/- 7.4 percent
  Mean number of basins = 21.6 +/- 8.1
  STD_F / RT = 0.37 +/- 0.01
  Fragmentation index f_inter = 0.0

The trivial ultrametric fraction is negligible in both regimes, below
0.5 percent, indicating that the observed effect is not caused by
degenerate distances in a high-dimensional random point set.

The results show moderate nontrivial ultrametricity that decreases
with increasing temperature, consistent with the loss of hierarchical
organization upon unfolding.


CITATION
--------

If you use this program, please cite the companion paper and the
software record.

Companion paper:

  A. P. Zubarev, "Kinetic metric on the space of macrostates and
  analysis of ultrametricity of the energy landscape of a protein
  molecule," 2026.

Software:

  A. P. Zubarev, "Ultrametricity analysis of protein energy landscapes
  based on kinetic metrics and Markov state models," computer software,
  Zenodo, 2026.
  DOI: 10.5281/zenodo.23696385

Related method papers:

  A. P. Zubarev, "Kinetic metric for basins of attraction of RNA
  secondary structures and analysis of ultrametricity of the energy
  landscape," arXiv:2606.25824, 2026.
  DOI: 10.48550/arXiv.2606.25824

  A. P. Zubarev, "Ultrametric organization of energy landscapes on
  random Erdos-Renyi graphs: topological origin of barrier hierarchy,"
  p-Adic Numbers Ultrametric Anal. Appl. 18 (2026) 362-386.
  DOI: 10.1134/S2070046626040047


LICENSE
-------

This software is provided as-is for scientific research purposes.


ACKNOWLEDGMENTS
---------------

The method builds upon the theoretical framework of ultrametric spaces
developed in the context of p-adic mathematical physics and spin glass
theory. The computational implementation relies on the OpenMM molecular
dynamics package, the GB99dms force field, and standard scientific
Python libraries (NumPy, SciPy, mdtraj).


CONTACT
-------

A. P. Zubarev
Volga State Transport University
Pervyi Bezymyannyi pereulok 18, Samara, 443066, Russia

Samara University
Moskovskoe shosse 34, Samara, 443123, Russia

Email: apzubarev@mail.ru
