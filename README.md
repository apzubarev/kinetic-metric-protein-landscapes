ULTRAMETRICITY ANALYSIS OF PROTEIN ENERGY LANDSCAPES
BASED ON KINETIC METRICS AND MARKOV STATE MODELS (MSM)
Version: 1.0.1
Author: A. P. Zubarev

OVERVIEW
This program implements a rigorous computational method for testing the
hypothesis of ultrametric organization in protein energy landscapes.
The approach is grounded in the analysis of transition kinetics between
macrostates (basins of attraction) using a kinetic metric. This metric
is constructed via spectral decomposition of the symmetrized Kramers
transition rate matrix and the Mahalanobis distance.

The software performs a complete pipeline from molecular dynamics
simulations to topological and spectral analysis of the conformational
space, providing quantitative measures of hierarchical organization.

METHOD DESCRIPTION AND COMPUTATIONAL STAGES

STAGE 1: MOLECULAR DYNAMICS (MD)
Generation of an equilibrium conformational ensemble via Langevin
dynamics using the OpenMM package. The program supports GB99dms and
Amber14/OBC2 implicit solvent force fields.

Robust Structure Preparation:
Before MD, the structure is automatically cleaned. Non-protein residues
such as ligands, ions, and water molecules are removed. The longest
polypeptide chain is selected, and missing atoms and hydrogens are
added. This ensures compatibility with OpenMM force fields for any
standard PDB entry.

In-Memory Buffering:
Instead of writing directly to disk, the program uses a custom
MemoryReporter that accumulates coordinates and energies in Python
lists. This solves the issue of corrupted frames during NaN events,
as RAM lists can be truncated atomically. A single DCD file is written
to disk via mdtraj only once at the very end, after successful
simulation completion. If rollbacks exceed the limit, the integration
step is automatically reduced and the simulation restarts.

STAGE 2: MARKOV STATE MODEL (MSM) CONSTRUCTION
Continuous conformational space is discretized by clustering trajectory
frames into microstates using the k-centers algorithm in RMSD metric.
The choice of atoms for RMSD calculation is determined by the geometry
mode: C-alpha atoms only or all-atom RMSD. The selected metric is used
uniformly for both clustering and neighbor graph construction.

For each microstate, free energy is computed via the partition function
over frame energies within that cluster. A neighbor graph is constructed
using either an energy-based threshold or an Adaptive Topological Graph.
In the adaptive method, an effective radius is computed for each cluster,
and two clusters are neighbors if the distance between centers does not
exceed a scaled sum of their radii.

Based on free energies and the neighbor graph, a symmetrized transition
rate matrix is built using the Kramers formula. Macrostates are
identified via gradient descent on the free energy landscape. Attractors
and plateaus are identified automatically, and each microstate is
assigned to a basin of attraction.

STAGE 3: SPECTRAL ANALYSIS AND ULTRAMETRICITY TESTING
For each significant connected component, spectral decomposition of the
rate matrix is performed. Noise modes are filtered via a spectral gap
criterion. Basin characteristic vectors are projected onto physical
eigenmodes. The distance between basins is defined as the Mahalanobis
distance, where slower relaxation modes contribute more to the distance.

Ultrametricity is tested by enumerating all basin triplets. A triplet
is non-trivially ultrametric if the two largest distances are
approximately equal and the third is significantly smaller. The fraction
of such triplets defines the degree of non-trivial ultrametricity.

EXECUTION ARCHITECTURE AND MODES
The program operates in one or more of the following modes:

- FROM_FILES_ANALYSIS:
Skip MD and load existing DCD/CIF/PDB files. The MSM is built once,
then enabled analytical modes run sequentially.

- SENSITIVITY_ANALYSIS:
Run analysis for multiple basin equivalence threshold values to
evaluate the stability of the landscape hierarchy.

- ALPHA_ANALYSIS:
Run analysis for multiple topological overlap parameters to study
the influence of graph connectivity on the results.

- NULL_H1_ENERGY_SHUFFLE:
Perform a baseline calculation, fix the topology and basins, then run
multiple permutations of free energies to estimate statistical
significance and separate topological effects from thermodynamic ones.

- MAIN_ANALYSIS:
Execute the full cycle multiple times with different random seeds.
Computes mean, standard deviation, and coefficient of variation to
assess the statistical stability and reproducibility of the results.

TECHNICAL AND HARDWARE REQUIREMENTS
Runtime: Windows 10/11, Linux, macOS. Python 3.10 to 3.12.
Required packages: openmm, pdbfixer, mdtraj, numpy, scipy,
threadpoolctl, and torch (required for CUDA JIT compilation support).

Hardware:
An NVIDIA GPU with CUDA support is highly recommended for the MD stage.
CPU fallback is automatic if GPU initialization fails. MSM construction
and spectral analysis stages run efficiently on multi-core CPUs.

INSTALLATION
Install the required dependencies using pip:
pip install openmm pdbfixer mdtraj numpy scipy threadpoolctl torch

USER PARAMETERS OVERVIEW
All configurable parameters are grouped at the beginning of the source
code and are extensively documented. They are divided into four groups:

Group 1: Execution Modes and General Settings
Controls the activation of specific analysis protocols, CPU thread
limits, random seed bases, and verbosity of the console output.

Group 2: Molecular Dynamics Parameters
Defines the PDB identifier, force field selection, Langevin temperature,
total integration steps, frame reporting intervals, and automatic
rollback mechanisms for numerical stability.

Group 3: MSM and Neighbor Graph Parameters
Configures the Kramers temperature, neighborhood method selection,
adaptive topology scaling factors, RMSD geometry modes, target number
of clusters, and basin equivalence thresholds.

Group 4: Spectral Analysis and Ultrametricity Parameters
Sets the number of eigenmodes to compute, Lanczos algorithm tolerances,
spectral gap thresholds for noise filtering, and the epsilon and delta
tolerances for triplet classification.

OUTPUT AND INTERPRETATION
The program outputs detailed diagnostic tables to the console, tracking
the formation of basins, connected components, and fragmentation
indices. The final results include the percentage of non-trivial,
trivial, and non-ultrametric triplets. A high percentage of
non-trivial ultrametricity, combined with a low fragmentation index,
indicates a strongly hierarchical, tree-like organization of the
protein energy landscape dictated by steric constraints.

STATISTICAL SIGNIFICANCE
The Energy Shuffle null model provides a p-value indicating whether
the observed ultrametricity is statistically distinguishable from
a randomized energy distribution on the same topological graph.
This allows researchers to conclude whether the hierarchy is driven
primarily by the topology of allowed transitions or by the specific
thermodynamic depths of the energy basins.

LICENSE AND CITATION
This software is provided for academic and research purposes.
When using this code or the underlying method in published research,
please cite the corresponding methodological paper on the kinetic
metric for protein energy landscapes and Markov State Models.
