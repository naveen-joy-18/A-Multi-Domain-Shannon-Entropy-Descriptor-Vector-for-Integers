# Universal Entropy Fingerprint Framework (UEFT)

## Reproducibility and Analysis Repository

This repository contains the computational implementation, input data, intermediate outputs, analysis results, and figure-generation scripts used for the manuscript on the **Universal Entropy Fingerprint Framework (UEFT)** for integers up to \(10^6\).

The framework represents each integer using a **19-dimensional Shannon-entropy fingerprint** constructed from five domains:

1. Algebraic entropy
2. Geometric entropy
3. Arithmetic entropy
4. Positional entropy across bases 2–16
5. Dynamical entropy

The computational study is exhaustive over the integer range

\[
n = 1,\ldots,1,000,000.
\]

No sampling of the integer domain is required for the principal fingerprint construction.

---

# 1. Repository contents

The repository contains the following major groups of files.

## 1.1 Core entropy implementation

| File | Purpose |
|---|---|
| `entropy_core.py` | Core Shannon-entropy calculations |
| `algebraic.py` | Algebraic entropy implementation |
| `geometric.py` | Geometric/divisor-lattice entropy implementation |
| `arithmetic.py` | Arithmetic entropy implementation |
| `positional.py` | Positional entropy implementation for bases 2–16 |
| `dynamical.py` | Dynamical/trajectory-based entropy implementation |
| `sieve.py` | Number-theoretic preprocessing/sieving |
| `data_loader.py` | Dataset loading and data handling |

## 1.2 Main pipeline

| File | Purpose |
|---|---|
| `main.py` | Main analysis entry point |
| `pipeline.py` | Computational pipeline and analysis orchestration |
| `analysis.py` | Downstream statistical/structural analyses |
| `Ueft_Update.py` | Additional/update analysis implementation |

## 1.3 PCA and domain-ablation results

| File | Purpose |
|---|---|
| `main_19D_standardized_explained_variance.csv` | Explained variance for the full 19-dimensional PCA |
| `main_19D_standardized_loadings.csv` | PCA component loadings |
| `main_19D_standardized_standardization.csv` | Standardization information |
| `main_19D_standardized_summary.json` | Summary of the full 19D PCA |
| `ablation_ALL_19_explained_variance.csv` | Full 19D ablation/PCA explained variance |
| `ablation_ALL_19_loadings.csv` | Full 19D loadings |
| `ablation_ALL_19_standardization.csv` | Full 19D standardization information |
| `ablation_ALL_19_summary.json` | Full 19D summary |
| `ablation_AGAr_explained_variance.csv` | AGAr ablation explained variance |
| `ablation_AGAr_loadings.csv` | AGAr loadings |
| `ablation_AGAr_standardization.csv` | AGAr standardization information |
| `ablation_AGAr_summary.json` | AGAr summary |
| `ablation_AGArD_explained_variance.csv` | AGArD ablation explained variance |
| `ablation_AGArD_standardization.csv` | AGArD standardization information |
| `ablation_AGArD_summary.json` | AGArD summary |
| `ablation_P_only_explained_variance.csv` | Positional-only ablation explained variance |
| `ablation_P_only_loadings.csv` | Positional-only PCA loadings |
| `ablation_P_only_standardization.csv` | Positional-only standardization information |
| `ablation_P_only_summary.json` | Positional-only summary |
| `domain_ablation_summary.csv` | Consolidated domain-ablation results |

## 1.4 Nearest-neighbour analysis

| File | Purpose |
|---|---|
| `knn_method.json` | Method/configuration used for nearest-neighbour analysis |
| `knn_proximity_k5.csv` | Proximity values for \(k=5\) |
| `knn_proximity_k10.csv` | Proximity values for \(k=10\) |
| `knn_proximity_k20.csv` | Proximity values for \(k=20\) |
| `knn_sensitivity_summary.csv` | Summary of \(k=5,10,20\) sensitivity analysis |
| `knn_pairwise_correlations.csv` | Pairwise correlations associated with the kNN analysis |

The primary manuscript analysis uses \(k=10\). Sensitivity was evaluated at \(k=5\), \(10\), and \(20\).

## 1.5 Arithmetic class validation

The repository contains class membership and validation outputs for several number-theoretic classes.

| File | Class/result |
|---|---|
| `class_counts.csv` | Class counts |
| `class_definitions.csv` | Definitions of evaluated classes |
| `class_centroids_standardized_19D.csv` | Standardized 19D class centroids |
| `class_feature_effects.csv` | Class-associated feature effects |
| `class_top5_effects.csv` | Top five effects by class |
| `prime_members.csv` | Prime members |
| `prime_power_members.csv` | Prime-power members |
| `squarefree_members.csv` | Squarefree members |
| `semiprime_members.csv` | Semiprime members |
| `abundant_members.csv` | Abundant members |
| `deficient_members.csv` | Deficient members |
| `carmichael_members.csv` | Carmichael members |
| `highly_composite_members.csv` | Highly composite members |
| `perfect_members.csv` | Perfect-number members |

## 1.6 Diagnostic and consistency outputs

| File | Purpose |
|---|---|
| `arithmetic_function_diagnostics.csv` | Arithmetic-function diagnostics |
| `dataset_consistency_checks.csv` | Dataset consistency checks |
| `collatz_cap_summary.json` | Sensitivity information for the dynamical iteration cap |
| `environment.json` | Computational environment information |
| `input_dataset.json` | Input dataset metadata |
| `run_arguments.json` | Run configuration/arguments |

## 1.7 Figure-generation scripts

| File | Figure |
|---|---|
| `fig_dimension_scree.py` | Cumulative explained-variance/scree figure |
| `fig_correlation.py` | Pearson correlation matrix |
| `fig_distributions.py` | Entropy-coordinate distributions |
| `fig_exemplar_fingerprints.py` | Exemplar fingerprint figure |
| `fig_density_vs_n.py` | Entropy-space proximity visualization |
| `fig_pca_projection.py` | PCA projection |

> `fig_density_vs_n.py` is retained as the current filename in the repository. The manuscript terminology should refer to the calculated quantity as **entropy-space proximity**, not as a formal statistical density estimator.

---

# 2. Software environment

The computational environment reported for the study is:

- **Python:** 3.10.5
- **NumPy:** 1.24.3
- **pandas:** 2.3.0
- **SciPy:** 1.15.3
- **scikit-learn:** 1.7.0
- **Matplotlib:** 3.10.3
- **Operating system:** Windows 11
- **CPU:** Intel Core i7-13650HX, 2.60 GHz
- **RAM:** 24 GB
- **Parallel workers:** 4 worker processes

The full one-million-integer dataset was reported to require approximately:

- **~24 minutes** for the full computational run
- **~18 GB peak memory**

Actual runtime and memory consumption may vary with hardware, operating system, Python environment, and process configuration.

The precise environment metadata is also stored in:

`environment.json`

---

# 3. Dataset

The principal analysis covers every integer in the range:

```text
1 through 1,000,000
```

The resulting fingerprint has 19 coordinates for each integer.

The five-domain structure is:

```text
1 Algebraic coordinate
1 Geometric coordinate
1 Arithmetic coordinate
15 Positional coordinates (bases 2–16)
1 Dynamical coordinate
--------------------------------
19 coordinates total
```

The positional domain deliberately depends on numeral representation and contains one entropy value for each base from 2 through 16.

Dataset metadata are provided in:

```text
input_dataset.json
```

Dataset consistency checks are provided in:

```text
dataset_consistency_checks.csv
```

---

# 4. Reproducing the principal analysis

The repository is organized so that the core fingerprint construction can be followed by downstream analyses.

A recommended reproduction sequence is:

```text
1. Environment setup
2. Input/dataset preparation
3. Fingerprint generation
4. PCA
5. Correlation analysis
6. Entropy-space proximity / kNN analysis
7. Arithmetic class validation
8. Domain ablation
9. Figure generation
```

The principal scripts are:

```text
main.py
pipeline.py
analysis.py
```

The domain-specific implementations are:

```text
algebraic.py
geometric.py
arithmetic.py
positional.py
dynamical.py
```

---

# 5. Running the computational pipeline

From the repository root, activate the Python environment and run the main pipeline.

Example:

```bash
python main.py
```

If the repository's `main.py` delegates execution through `pipeline.py`, the pipeline performs the corresponding dataset construction and analysis steps.

For individual components, the relevant scripts/modules can be run or imported according to the workflow implemented in the repository.

Before a full reproduction, inspect:

```text
run_arguments.json
environment.json
input_dataset.json
```

These files record run and environment metadata associated with the supplied results.

---

# 6. Entropy fingerprint construction

For each integer \(n\), the framework calculates entropy-based coordinates from five domains.

## 6.1 Algebraic domain

The algebraic coordinate is derived from the prime-exponent structure of the integer.

The implementation is contained in:

```text
algebraic.py
```

For \(n=1\), the algebraic entropy is defined as zero by convention.

---

## 6.2 Geometric domain

The geometric coordinate is based on the divisor lattice of the integer.

The divisor structure is represented through rank information associated with the exponent-vector representation of the prime factorization.

The implementation is contained in:

```text
geometric.py
```

For \(n=1\), the divisor structure reduces to a single outcome, giving zero Shannon entropy.

---

## 6.3 Arithmetic domain

The arithmetic coordinate uses the divisor-based probability distribution associated with the sum-of-divisors function.

The implementation is contained in:

```text
arithmetic.py
```

For \(n=1\), the distribution contains a single outcome with probability one, giving zero entropy.

---

## 6.4 Positional domain

Positional entropy is evaluated independently in numeral bases:

```text
2, 3, 4, ..., 16
```

This produces 15 positional entropy coordinates.

The implementation is contained in:

```text
positional.py
```

For \(n=1\), the representation in every considered base is a single digit, giving zero entropy.

---

## 6.5 Dynamical domain

The dynamical coordinate is calculated from the trajectory associated with the dynamical rule implemented in:

```text
dynamical.py
```

The treatment of \(n=1\) uses a zero-entropy convention because there is no nontrivial transition.

The repository also contains:

```text
collatz_cap_summary.json
```

for the iteration-cap sensitivity information.

---

# 7. Numerical precision

The entropy calculations use double-precision floating-point arithmetic.

The logarithmic calculations used in the Shannon entropy evaluations are therefore performed using:

```text
IEEE 754 binary64
```

corresponding to approximately 15–16 decimal digits of numerical precision.

---

# 8. PCA analysis

PCA is performed on the standardized 19-dimensional entropy fingerprint.

The standardization procedure uses the dataset-wide mean and standard deviation of each coordinate.

The PCA implementation uses:

```text
sklearn.preprocessing.StandardScaler
sklearn.decomposition.PCA
```

The full PCA results are stored in:

```text
main_19D_standardized_explained_variance.csv
main_19D_standardized_loadings.csv
main_19D_standardized_standardization.csv
main_19D_standardized_summary.json
```

The corresponding complete/ablation result files are also provided under the `ablation_*` filenames.

## Reported principal results

For the full 19-dimensional standardized fingerprint:

- PC1 explains **13.0641%**
- PC2 explains **10.4737%**
- PC1 + PC2 explain **23.5377% cumulatively**
- 15 PCs are required to exceed **90%** cumulative explained variance
- 17 PCs are required to exceed **95%**
- 18 PCs are required to exceed **99%**

The full per-component values are available in:

```text
main_19D_standardized_explained_variance.csv
```

---

# 9. PCA figure reproduction

The PCA projection can be generated using:

```bash
python fig_pca_projection.py
```

The PCA projection displays the first two principal components of the standardized 19-dimensional fingerprint.

The corresponding figure-generation script is:

```text
fig_pca_projection.py
```

The scree/cumulative explained-variance visualization is generated using:

```bash
python fig_dimension_scree.py
```

---

# 10. Correlation analysis

The repository includes the Pearson correlation analysis of the 19 entropy coordinates.

The corresponding script is:

```bash
python fig_correlation.py
```

The analysis covers all:

```text
19 × 19
```

coordinate pairs across the one-million-integer dataset.

The graphical correlation matrix is generated by the figure script.

The correlation analysis is intended to describe pairwise linear association. Low Pearson correlation should not be interpreted as proof of statistical independence.

---

# 11. Entropy-space proximity / nearest-neighbour analysis

For each integer, entropy-space proximity is calculated from the mean Euclidean distance to its nearest neighbours in the 19-dimensional fingerprint space.

The primary analysis uses:

```text
k = 10
```

The implementation uses the nearest-neighbour functionality from:

```text
sklearn.neighbors.NearestNeighbors
```

The primary output is:

```text
knn_proximity_k10.csv
```

Sensitivity outputs are:

```text
knn_proximity_k5.csv
knn_proximity_k10.csv
knn_proximity_k20.csv
knn_sensitivity_summary.csv
```

The corresponding method configuration is stored in:

```text
knn_method.json
```

The choice of \(k=10\) is assessed against \(k=5\) and \(k=20\).

The reported sensitivity analysis gives:

| k | Mean proximity | Median proximity |
|---:|---:|---:|
| 5 | 0.4288 | 0.4265 |
| 10 | 0.4084 | 0.4077 |
| 20 | 0.3887 | 0.3890 |

The analysis is described as **entropy-space proximity** rather than a formal statistical density estimator.

The visualization is generated using:

```bash
python fig_density_vs_n.py
```

---

# 12. Arithmetic class validation

The repository contains systematic validation outputs for several number-theoretic classes.

The class definitions are recorded in:

```text
class_definitions.csv
```

The class counts are recorded in:

```text
class_counts.csv
```

Additional outputs include standardized class centroids and class-associated feature effects:

```text
class_centroids_standardized_19D.csv
class_feature_effects.csv
class_top5_effects.csv
```

The member lists are provided separately for the evaluated classes.

This analysis is intended to examine whether known arithmetic categories exhibit distinguishable structure in the entropy fingerprint space.

---

# 13. Domain ablation

Domain ablation evaluates the contribution of different groups of entropy coordinates.

The repository includes results for:

```text
AGAr
AGArD
P-only
ALL-19
```

Corresponding outputs include:

```text
ablation_AGAr_*
ablation_AGArD_*
ablation_P_only_*
ablation_ALL_19_*
```

The consolidated results are available in:

```text
domain_ablation_summary.csv
```

The ablation analysis provides a comparison of the dimensional structure obtained from different subsets of the five-domain fingerprint.

---

# 14. Exemplar fingerprint analysis

The exemplar analysis is generated using:

```bash
python fig_exemplar_fingerprints.py
```

The analysis includes representative integers from different number-theoretic categories.

These examples are intended as **illustrative demonstrations of fingerprint structure**, not as a standalone validation of classification performance.

---

# 15. Entropy-coordinate distributions

The distribution of the 19 entropy coordinates can be generated using:

```bash
python fig_distributions.py
```

This visualization examines the empirical distributions of the individual coordinates across all one million integers.

---

# 16. Reproducibility outputs

The repository contains both scripts and generated outputs so that the principal results can be checked without reconstructing every intermediate calculation.

Important reproducibility files include:

```text
environment.json
input_dataset.json
run_arguments.json
dataset_consistency_checks.csv
main_19D_standardized_summary.json
main_19D_standardized_explained_variance.csv
knn_method.json
knn_sensitivity_summary.csv
domain_ablation_summary.csv
collatz_cap_summary.json
```

The CSV and JSON outputs provide numerical records corresponding to the principal analyses.

---

# 17. Expected principal results

A successful reproduction of the principal 19D PCA should recover approximately:

```text
PC1                  13.0641%
PC2                  10.4737%
PC1 + PC2            23.5377%
PCs for >90%         15
PCs for >95%         17
PCs for >99%         18
```

The nearest-neighbour sensitivity analysis should reproduce approximately:

```text
k = 5     mean 0.4288     median 0.4265
k = 10    mean 0.4084     median 0.4077
k = 20    mean 0.3887     median 0.3890
```

Small numerical differences may occur if the computational environment or numerical libraries differ from the reported environment.

---

# 18. Figure-generation summary

The principal figures can be regenerated using:

```bash
python fig_dimension_scree.py
python fig_correlation.py
python fig_distributions.py
python fig_exemplar_fingerprints.py
python fig_density_vs_n.py
python fig_pca_projection.py
```

These scripts correspond to the manuscript's principal visualization workflow.

---

# 19. Recommended reproduction workflow

For a clean reproduction, the following sequence is recommended:

### Step 1 — Create the Python environment

Use Python 3.10.5 with the package versions recorded in `environment.json`.

### Step 2 — Place the repository files in one working directory

The Python scripts and required data/configuration files should be accessible from the repository root.

### Step 3 — Inspect run metadata

Review:

```text
environment.json
input_dataset.json
run_arguments.json
```

### Step 4 — Generate/reconstruct the fingerprint dataset

Run the main computational pipeline:

```bash
python main.py
```

### Step 5 — Run downstream analyses

Use the analysis pipeline and corresponding scripts for:

- PCA
- correlation analysis
- entropy-space proximity
- arithmetic class validation
- domain ablation
- dynamical sensitivity analysis

### Step 6 — Generate figures

Run:

```bash
python fig_dimension_scree.py
python fig_correlation.py
python fig_distributions.py
python fig_exemplar_fingerprints.py
python fig_density_vs_n.py
python fig_pca_projection.py
```

### Step 7 — Compare outputs

Compare the regenerated outputs with the supplied CSV/JSON result files.

---

# 20. Computational limitations and interpretation

The framework provides deterministic entropy coordinates for the specified integer range and enables large-scale structural analysis.

The PCA, correlation, exemplar, and nearest-neighbour analyses are primarily **descriptive analyses of the resulting fingerprint space**.

They should not by themselves be interpreted as evidence that the fingerprint is an optimized predictive classifier or that any particular downstream application has already been demonstrated.

Potential applications to predictive modelling, classification, anomaly detection, or other task-specific problems require separate validation and are therefore considered future work.

---

# 21. Reproducibility statement

All principal computational components, input metadata, analysis scripts, generated numerical outputs, and figure-generation scripts required to reproduce the reported analyses are included in this repository.

The repository is intended to allow independent reconstruction and verification of the principal tables, numerical results, and figures reported in the manuscript.
