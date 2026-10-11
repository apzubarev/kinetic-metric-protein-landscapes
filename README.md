PROTEIN ENERGY LANDSCAPE ULTRAMETRIC ANALYSIS
Kinetic Metric and Markov State Model Based Analysis
Version 1.12
Author: A. P. Zubarev

OVERVIEW
--------

This program analyzes protein energy landscapes and tests the hypothesis of ultrametric organization.
It combines Langevin molecular dynamics, clustering into microstates, Markov state model construction,
Kramers transition rates, spectral decomposition of a symmetrized rate matrix, and a kinetic metric
based on the Mahalanobis distance.

The program computes the fractions of ultrametric triplets of attraction basins and prints summary tables.
The final tables are printed even when detailed output is disabled.

MAIN FEATURES
-------------

- Automatic preparation of protein structures from PDB or CIF files.
- Removal of non-protein residues and selection of the longest polypeptide chain.
- Langevin dynamics with OpenMM and implicit solvent.
- Support for GB99dms and Amber14/OBC2 force fields.
- In-memory trajectory buffering with rollback protection against NaN coordinates.
- Automatic reduction of the integration step when repeated NaN errors occur.
- k-centers clustering in RMSD space using C-alpha atoms or all protein atoms.
- Free energy estimation for microstates from frame energies.
- Two neighborhood graph methods: energy threshold and adaptive topology.
- Basin identification by gradient descent on the free energy landscape.
- Spectral filtering of noise modes using a spectral gap criterion.
- Sensitivity analysis for plateau threshold and neighbor alpha.
- Repeated main analysis with mean, standard deviation, and coefficient of variation.

WORK MODES
----------

FROM_FILES_ANALYSIS
Use this mode to skip molecular dynamics and analyze existing trajectory files.
The program scans the working directory, loads available samples, clusters them,
and performs the selected analytical modes.

SENSITIVITY_ANALYSIS
Use this mode to test the stability of results to BASIN_EQ_THRESHOLD.
The values are taken from SENSITIVITY_RANGE.
This mode cannot be combined with ALPHA_ANALYSIS.

ALPHA_ANALYSIS
Use this mode to test the stability of results to NEIGHBOR_ALPHA.
The values are taken from ALPHA_RANGE.
This mode cannot be combined with SENSITIVITY_ANALYSIS.
It is meaningful only when NEIGHBOR_METHOD is adaptive_topology.

MAIN_ANALYSIS
Use this mode for the main repeated analysis.
The number of independent samples is MAIN_NUM_SAMPLES.
For file-based analysis, the number of samples is determined by the available files.

EXPECTED FILES FOR FROM_FILES_ANALYSIS
--------------------------------------

For sample 1:
PDBID_trajectory_1.dcd
PDBID_1.pdb or PDBID_1.cif
PDBID_energies_1.txt

For sample 2:
PDBID_trajectory_2.dcd
PDBID_2.pdb or PDBID_2.cif
PDBID_energies_2.txt

For a single sample without suffix:
PDBID_trajectory.dcd
PDBID.pdb or PDBID.cif
PDBID_energies.txt

Replace PDBID with the value of PDB_ID, for example 1L2Y.
If the energy file is absent, the program can recalculate energies using OpenMM.

REQUIREMENTS
------------

Operating systems:
Windows 10 or Windows 11, Linux, macOS.

Python:
3.10, 3.11, or 3.12.

Required packages:
openmm
pdbfixer
mdtraj
numpy
scipy
threadpoolctl
torch

Hardware:
An NVIDIA GPU is recommended for molecular dynamics.
If CUDA is unavailable, the program automatically uses CPU.
Markov state model construction and spectral analysis run on CPU.

INSTALLATION
------------

Create a virtual environment and install the required packages:

python -m venv env
source env/bin/activate
pip install openmm pdbfixer mdtraj numpy scipy threadpoolctl torch

On Windows, use:

python -m venv env
env\Scripts\activate
pip install openmm pdbfixer mdtraj numpy scipy threadpoolctl torch

USAGE
-----

1. Open the Python source file.
2. Set the desired work modes at the top of the file.
3. Set the protein identifier or provide a custom structure file.
4. Run the script.

Example command:

python ultrametric_landscape_analysis.py

TYPICAL SCENARIOS
-----------------

Scenario A: generate new trajectories and perform main analysis

FROM_FILES_ANALYSIS = False
MAIN_ANALYSIS = True
MAIN_NUM_SAMPLES = 5
SENSITIVITY_ANALYSIS = False
ALPHA_ANALYSIS = False

Scenario B: analyze existing trajectories

FROM_FILES_ANALYSIS = True
MAIN_ANALYSIS = True
SENSITIVITY_ANALYSIS = False
ALPHA_ANALYSIS = False

Scenario C: analyze sensitivity of the adaptive graph to neighbor alpha

FROM_FILES_ANALYSIS = True
NEIGHBOR_METHOD = adaptive_topology
ALPHA_ANALYSIS = True
ALPHA_RANGE = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SENSITIVITY_ANALYSIS = False
MAIN_ANALYSIS = False

Note: ALPHA_RANGE must be written as a Python list in the source file.

PARAMETER GROUPS
----------------

General parameters:
MAX_CPU_THREADS limits the number of CPU threads.
FROM_FILES_ANALYSIS enables analysis from existing files.
SENSITIVITY_ANALYSIS enables plateau threshold sensitivity analysis.
SENSITIVITY_RANGE gives the list of BASIN_EQ_THRESHOLD values.
ALPHA_ANALYSIS enables neighbor alpha sensitivity analysis.
ALPHA_RANGE gives the list of NEIGHBOR_ALPHA values.
MAIN_ANALYSIS enables repeated main analysis.
MAIN_NUM_SAMPLES sets the number of independent samples.
NULL_SEED_BASE sets the base seed for reproducible replicas.
USE_GPU enables GPU molecular dynamics when available.
VERBOSE controls detailed console output.
N_COMP_CON sets how many large connected components are shown in diagnostic tables.

Molecular dynamics parameters:
PDB_ID is the PDB identifier for downloading the starting structure.
STARTING_STRUCTURE_FILE is an optional local PDB or CIF file.
ADD_HYDROGENS_TO_CUSTOM_START controls adding hydrogens to a custom structure.
USE_GB99DMS selects GB99dms when True, otherwise Amber14/OBC2.
TEMPERATURE_K sets the simulation temperature.
MD_STEPS sets the total number of integration steps.
MD_REPORT_INTERVAL sets how often frames and energies are saved.
MINIMIZE_STARTING_STRUCTURE enables energy minimization before dynamics.
MD_SEED sets the random seed for a single simulation.
MD_TIME_STEP_FS sets the initial integration step in femtoseconds.
MD_ROLLBACK_STEPS sets the chunk size used for NaN rollback.
MD_MAX_ROLLBACKS sets the maximum consecutive rollbacks before reducing the step.
MD_TIME_STEP_REDUCTION_FACTOR sets the step reduction factor.
MD_MIN_TIME_STEP_FS sets the minimum allowed integration step.

Markov state model parameters:
KRAMERS_TEMPERATURE sets the temperature used in Kramers rates.
NEIGHBOR_METHOD selects energy or adaptive_topology neighborhood.
ETA_NEIGHBOR_MODE selects manual or automatic eta for energy neighborhood.
ETA_NEIGHBOR sets the manual eta value.
NEIGHBOR_RADIUS_PERCENTILE sets the percentile used for cluster effective radius.
NEIGHBOR_ALPHA sets the scaling factor for the adaptive topology neighborhood.
GEOMETRY_RMSD_MODE selects ca for C-alpha RMSD or all for all-atom RMSD.
N_CLUSTERS sets the target number of clusters.
MIN_CLUSTER_SIZE sets the minimum number of frames per cluster.
BASIN_EQ_THRESHOLD sets the energy equivalence threshold for plateaus.
MIN_BASIN_SIZE sets the minimum number of microstates per basin.
MIN_BASINS_PER_COMPONENT sets the minimum number of basins per significant component.

Spectral analysis parameters:
NUM_EIGENMODES sets the maximum number of requested eigenmodes.
EIGS_SIGMA sets the shift-invert sigma for sparse diagonalization.
EIGS_TOL sets the convergence tolerance.
EIGS_DENSE_THRESHOLD sets the matrix size threshold for dense diagonalization.
SPECTRAL_GAP_THRESHOLD sets the gap ratio used to filter noise modes.
ULTRAMETRIC_EPSILON sets the tolerance for equality of the two largest distances.
ULTRAMETRIC_DELTA sets the minimum separation of the smaller distance.

COMPUTATION PIPELINE
--------------------

1. Structure preparation.
The program downloads or loads a structure, removes non-protein residues,
selects the longest chain, adds missing atoms and hydrogens, and saves a cleaned topology.

2. Molecular dynamics.
Langevin dynamics is run with OpenMM. Coordinates and energies are stored in memory.
If NaN errors occur, the program rolls back to the last stable state.
If rollbacks are too frequent, the time step is reduced and the simulation restarts.

3. Trajectory clustering.
The trajectory is clustered into microstates by the k-centers algorithm.
The RMSD metric is the same for clustering and graph construction.

4. Free energy calculation.
For each cluster, the free energy is estimated from the energies of its frames.

5. Neighborhood graph.
The program connects microstates either by an energy difference criterion
or by an adaptive topological criterion based on effective cluster radii.

6. Rate matrix construction.
A symmetrized Kramers rate matrix is built from free energies and the neighborhood graph.

7. Basin identification.
Attractors are found first. Equivalent attractors are merged into plateaus.
Remaining microstates are assigned by descent to the nearest lower basin.

8. Kinetic metric and ultrametricity.
For each significant connected component, the program computes physical eigenmodes,
projects basin vectors onto these modes, computes distances, and classifies all basin triplets.

OUTPUT
------

The program writes trajectory, topology, and energy files.
It also prints tables for connected components, basin diagnostics, sensitivity analysis,
and main analysis results.

Important reported quantities:

f_inter is the fragmentation index of basins across connected components.
u_nontrivial is the fraction of nontrivial ultrametric triplets, reported in percent.
u_trivial is the fraction of trivial triplets, reported in percent.
u_nonultrametric is the fraction of non-ultrametric triplets, reported in percent.
CV is the coefficient of variation of u_nontrivial, reported in percent.

TROUBLESHOOTING
---------------

Problem: CUDA initialization fails.
Solution: the program should switch to CPU automatically. If the problem persists,
set USE_GPU = False.

Problem: pdbfixer is not found.
Solution: install pdbfixer with pip.

Problem: simulation stops with NaN errors.
Solution: reduce MD_TIME_STEP_FS, increase MD_MAX_ROLLBACKS, or lower TEMPERATURE_K.

Problem: too few significant clusters.
Solution: reduce N_CLUSTERS, reduce MIN_CLUSTER_SIZE, or run a longer trajectory.

Problem: no significant basins.
Solution: adjust BASIN_EQ_THRESHOLD, MIN_BASIN_SIZE, or MIN_BASINS_PER_COMPONENT.
For adaptive topology, try a larger NEIGHBOR_ALPHA.

Problem: tables are empty.
Solution: check that the neighborhood graph has connected components and that enough
frames and microstates are present. Increase MD_STEPS or reduce MIN_CLUSTER_SIZE.

LICENSE
-------

Add the license information required for your repository.

CITATION
--------

If you use this program in research, cite the original method and relevant software packages:
OpenMM, mdtraj, SciPy, and NumPy.
