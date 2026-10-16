ULTRAMETRIC ANALYSIS OF PROTEIN ENERGY LANDSCAPES
BASED ON KINETIC METRICS AND MARKOV STATE MODELS (MSM)

Author: A. P. Zubarev
Version: 1.13
License: MIT


DESCRIPTION
===========

This program implements a method for testing the hypothesis of ultrametric
organization of protein energy landscapes. The method is based on the analysis
of transition kinetics between macrostates (basins of attraction) using a
kinetic metric constructed via spectral decomposition of the symmetrized
Kramers transition rate matrix and the Mahalanobis distance.

The program performs the following computational stages:

  1. MOLECULAR DYNAMICS (MD) STAGE
     Generation of an equilibrium conformational ensemble using Langevin
     dynamics with the OpenMM package. Implicit solvent force fields
     GB99dms and Amber14/OBC2 are supported.

  2. MARKOV STATE MODEL (MSM) CONSTRUCTION STAGE
     The continuous conformational space is discretized by clustering
     trajectory frames into microstates using the k-centers algorithm
     in the RMSD metric. For each microstate, the free energy is computed
     via the partition function. The neighborhood graph is built using
     an adaptive topological criterion. Macrostates (basins of attraction)
     are identified by gradient descent on the free energy landscape.

  3. SPECTRAL ANALYSIS AND ULTRAMETRICITY TESTING STAGE
     For each significant connected component, the spectral decomposition
     of the symmetrized rate matrix is performed. Noisy modes are filtered
     out using the spectral gap criterion. The kinetic metric between basins
     is the Mahalanobis distance in the space of physical relaxation modes.
     Ultrametricity is tested by enumerating all triplets of basins.


KEY FEATURES
============

  - Robust structure preparation: automatic cleaning of non-protein residues,
    chain selection, hydrogen addition via PDBFixer.

  - In-memory buffering with automatic NaN rollback: the program uses a
    custom MemoryReporter that accumulates coordinates in RAM. Binary DCD
    files are written only once at the very end, preventing corrupted output.

  - Automatic integration step reduction upon repeated numerical instability.

  - Two neighborhood graph methods: energetic (delta-F threshold) and
    adaptive topological (percentile-based effective radii with alpha scaling).

  - Spectral gap filtering to separate physical relaxation modes from
    numerical noise.

  - Weighted-average ultrametricity across connected components with
    fragmentation index reporting.

  - Multiple analysis modes: sensitivity to plateau threshold, sensitivity
    to neighbor alpha, main analysis with statistical reporting.

  - GPU acceleration via CUDA with automatic CPU fallback.


RELATED PUBLICATION
===================

A. P. Zubarev, "Kinetic metric on the space of macrostates of a protein
chain and analysis of ultrametricity of the energy landscape" (2026).

The method was previously applied to:
  - RNA secondary structure landscapes (arXiv:2606.25824)
  - Random Erdos-Renyi graphs (arXiv:2607.15902)
  - A toy model of a disordered heteropolymer (J. Phys. A, 2026)


REQUIREMENTS
============

  Operating System: Windows 10/11, Linux, macOS
  Python: 3.10, 3.11, or 3.12

  Required Python packages:

    openmm        - molecular dynamics engine
    pdbfixer      - structure preparation and repair
    mdtraj        - trajectory I/O and RMSD computation
    numpy         - numerical arrays
    scipy         - sparse linear algebra, connected components
    threadpoolctl - thread limit control
    torch         - CUDA JIT compilation support (optional but recommended)

  Hardware:
    GPU (NVIDIA CUDA) is strongly recommended for the MD stage.
    The program automatically falls back to CPU if no GPU is available.
    MSM construction and spectral analysis stages run on CPU.


INSTALLATION
============

  Step 1. Create a virtual environment (recommended):

    python -m venv protein_env
    source protein_env/bin/activate        (Linux/macOS)
    protein_env\Scripts\activate           (Windows)

  Step 2. Install dependencies:

    pip install openmm pdbfixer mdtraj numpy scipy threadpoolctl torch

  Step 3. Download the program file from this repository.

  Step 4. (Optional) For GPU support, ensure that NVIDIA drivers and
  the CUDA toolkit are installed. The program uses OpenMM built-in
  CUDA support.


USAGE
=====

  Run the program:

    python ultrametric_protein_analysis.py

  The program behavior is controlled entirely by user parameters defined
  at the top of the script. No command-line arguments are required.

  WORK MODES:

    FROM_FILES_ANALYSIS = True
      Skip MD generation. Load existing trajectory files from disk:
        {base_name}_trajectory_N.dcd
        {base_name}_N.pdb (or .cif)
        {base_name}_energies_N.txt
      The program scans the folder automatically.

    FROM_FILES_ANALYSIS = False
      Run the full cycle: MD generation, clustering, MSM, analysis.
      Generates MAIN_NUM_SAMPLES independent trajectories.

    SENSITIVITY_ANALYSIS = True
      Run analysis for multiple values of BASIN_EQ_THRESHOLD
      specified in SENSITIVITY_RANGE.

    ALPHA_ANALYSIS = True
      Run analysis for multiple values of NEIGHBOR_ALPHA
      specified in ALPHA_RANGE.

    MAIN_ANALYSIS = True
      Full statistical analysis over all trajectories with base parameters.
      Reports mean, standard deviation, and coefficient of variation.

  IMPORTANT: SENSITIVITY_ANALYSIS and ALPHA_ANALYSIS cannot both be True.


USER PARAMETERS (SUMMARY)
=========================

  GROUP 1: WORK MODE SELECTION AND GENERAL SETTINGS

    MAX_CPU_THREADS          - CPU thread limit (0 = auto, use all cores)
    FROM_FILES_ANALYSIS      - Load from files instead of running MD
    SENSITIVITY_ANALYSIS     - Enable EQ threshold sensitivity loop
    SENSITIVITY_RANGE        - List of EQ threshold values (kJ/mol)
    ALPHA_ANALYSIS           - Enable alpha sensitivity loop
    ALPHA_RANGE              - List of neighbor alpha values
    MAIN_ANALYSIS            - Enable main statistical analysis
    MAIN_NUM_SAMPLES         - Number of independent runs
    NULL_SEED_BASE           - Base random seed for reproducibility
    USE_GPU                  - Use CUDA GPU for MD (True/False)
    VERBOSE                  - Detailed console output (True/False)
    N_COMP_CON               - Number of components shown in diagnostics
    METRIC_OUTPUT            - Save distance matrices as .npy files

  GROUP 2: MOLECULAR DYNAMICS PARAMETERS

    PDB_ID                   - PDB identifier (e.g., "1L2Y")
    STARTING_STRUCTURE_FILE  - Custom starting structure path ("" = PDB)
    ADD_HYDROGENS_TO_CUSTOM_START - Add H to custom file
    USE_GB99DMS              - Force field: True = GB99dms, False = Amber14
    TEMPERATURE_K            - Langevin dynamics temperature (K)
    MD_STEPS                 - Total integration steps
    MD_REPORT_INTERVAL       - Frame saving interval (steps)
    MINIMIZE_STARTING_STRUCTURE - Energy minimization before MD
    MD_SEED                  - Random seed for single-run mode
    MD_TIME_STEP_FS          - Initial integration step (fs)
    MD_ROLLBACK_STEPS        - Rollback chunk size (steps)
    MD_MAX_ROLLBACKS         - Max consecutive rollbacks before step reduction
    MD_TIME_STEP_REDUCTION_FACTOR - Step division factor on instability
    MD_MIN_TIME_STEP_FS      - Minimum allowed step (fs)

  GROUP 3: MSM AND NEIGHBORHOOD GRAPH PARAMETERS

    KRAMERS_TEMPERATURE      - Temperature in Kramers formula (K)
    NEIGHBOR_METHOD          - 'energy' or 'adaptive_topology'
    ETA_NEIGHBOR_MODE        - 'manual' or 'auto' (for energy method)
    ETA_NEIGHBOR             - Transition rate threshold (for manual mode)
    NEIGHBOR_RADIUS_PERCENTILE - Percentile for effective radius
    NEIGHBOR_ALPHA           - Scaling factor for sphere overlap
    GEOMETRY_RMSD_MODE       - 'ca' (C-alpha only) or 'all' (all atoms)
    N_CLUSTERS               - Target number of microstates
    MIN_CLUSTER_SIZE         - Minimum frames per cluster
    BASIN_EQ_THRESHOLD       - Energy equivalence threshold (kJ/mol)
    MIN_BASIN_SIZE           - Minimum microstates per basin
    MIN_BASINS_PER_COMPONENT - Minimum basins per connected component

  GROUP 4: SPECTRAL ANALYSIS AND ULTRAMETRICITY PARAMETERS

    NUM_EIGENMODES           - Maximum eigenmodes to compute
    EIGS_SIGMA               - Lanczos shift-invert sigma
    EIGS_TOL                 - Lanczos convergence tolerance
    EIGS_DENSE_THRESHOLD     - Size threshold for dense diagonalization
    SPECTRAL_GAP_THRESHOLD   - Ratio threshold for noise filtering
    ULTRAMETRIC_EPSILON      - Relative tolerance for equal distances
    ULTRAMETRIC_DELTA        - Minimum relative difference for hierarchy


OUTPUT FILES
============

  During execution, the program creates the following files in the
  working directory:

    {base_name}_trajectory_N.dcd   - Binary trajectory (DCD format)
    {base_name}_N.pdb              - Cleaned topology (PDB format)
    {base_name}_energies_N.txt     - Potential energies (one per frame)
    seed_X_alpha_Y.npy             - Kinetic metric matrix (if METRIC_OUTPUT=True)

  Console output includes:
    - Stage progress and timing information
    - Basin formation diagnostics table
    - Significant connected components table
    - Ultrametricity results (u_nt, u_tr, u_non)
    - Sensitivity analysis tables
    - Statistical stability report (mean, SD, CV)


REPRODUCIBILITY
===============

  All random number generators are seeded deterministically:
    - In single-run mode: MD_SEED is used.
    - In multi-run mode (MAIN_ANALYSIS or ALPHA_ANALYSIS):
      seed = NULL_SEED_BASE + (run_number - 1)

  To reproduce results, use the same parameter values and seeds.


THEORETICAL BACKGROUND
======================

  The kinetic metric between basins of attraction is defined as:

    D^2(B_a, B_b) = sum_k (1/|lambda_k|) x (pi_ak - pi_bk)^2

  where lambda_k are eigenvalues of the symmetrized rate matrix,
  and pi_ak are projections of basin characteristic vectors onto
  the k-th relaxation mode.

  The metric satisfies the following identity (Theorem 1):

    D^2(B_a, B_b) = (tau_ab + tau_ba - tau_aa - tau_bb) / Z

  where tau_ab is the mean first-passage time between basins,
  and Z is the total partition function.

  In the limit of large free energy spread (Theorem 2):

    lim (1/beta) x ln D_beta(B_a, B_b) = (1/2) x rho_0(B_a, B_b)

  where rho_0 is the single-linkage ultrametric.

  Ultrametricity is assessed by classifying all triplets of basins
  into three categories:
    - Trivially ultrametric (all three distances approximately equal)
    - Nontrivially ultrametric (two large distances equal, third smaller)
    - Non-ultrametric (neither condition holds)

  The degree of nontrivial ultrametricity u_nt is the fraction of
  nontrivially ultrametric triplets (in percent).


RECOMMENDED PARAMETER VALUES
============================

  For a 20-residue mini-protein (e.g., Trp-cage, PDB 1L2Y):

    TEMPERATURE_K = 300 to 1100 (K)
    MD_STEPS = 10,000,000
    MD_REPORT_INTERVAL = 100
    N_CLUSTERS = 5000
    MIN_CLUSTER_SIZE = 10
    BASIN_EQ_THRESHOLD = 1.0 (kJ/mol)
    MIN_BASIN_SIZE = 10
    MIN_BASINS_PER_COMPONENT = 10
    NEIGHBOR_ALPHA = 0.5 to 1.5 (sensitivity scan)
    GEOMETRY_RMSD_MODE = 'all'
    NUM_EIGENMODES = 50
    SPECTRAL_GAP_THRESHOLD = 1e6
    ULTRAMETRIC_EPSILON = 0.05
    ULTRAMETRIC_DELTA = 0.1
    MAIN_NUM_SAMPLES = 5 to 10


LIMITATIONS
===========

  - The barrier approximation uses max(F_i, F_j) as the transition state
    energy, which may underestimate true barriers. However, for ultrametric
    analysis, the relative hierarchy of barriers is preserved.

  - At sampling temperatures above 700 K, the GB99dms implicit solvent
    model may lose physical accuracy. High-temperature results should be
    interpreted as a mathematical stress test of the method.

  - The number of microstates (N_CLUSTERS = 5000) and clustering parameters
    are chosen based on computational feasibility and may affect quantitative
    values of u_nt.

  - The limiting theorem was proven for Erdos-Renyi graphs. Its direct
    application to protein systems with topological constraints (chain
    connectivity, steric hindrance) is an extrapolation supported by
    numerical evidence but not rigorously proven.


CITATION
========

  If you use this software in your research, please cite:

  A. P. Zubarev, "Ultrametricity analysis of protein energy landscapes
  based on kinetic metrics and Markov state models," [Computer software],
  Zenodo, 2026.
  DOI: 10.5281/zenodo.23300800

  Related publications:

  A. P. Zubarev, "Kinetic metric for basins of attraction of RNA secondary
  structures and analysis of ultrametricity of the energy landscape,"
  arXiv:2606.25824 (2026).

  A. P. Zubarev, "Ultrametric organization of energy landscapes on random
  Erdos-Renyi graphs: topological origin of barrier hierarchy,"
  arXiv:2607.15902 (2026).


CONTACT
=======

  A. P. Zubarev
  Volga State Transport University, Samara, Russia
  Samara University, Samara, Russia
  E-mail: apzubarev@mail.ru


LICENSE
=======

  MIT License. See LICENSE file for details.
