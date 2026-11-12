# Distilling a black-box classifier into interpretable students


A gradient-boosted tree ensemble (the teacher) is trained on two tabular classification datasets, frozen, and
distilled into two interpretable students: a sparse PCE-HDMR polynomial and an additive spline model (GAM).
The students are evaluated for accuracy, fidelity to the teacher, complexity, and the agreement of their
explanations with each other and with a post-hoc Shapley explanation of the teacher.

## Repository contents

| Path | Content |
|---|---|
| `interpretable_distillation.ipynb` | The complete experiment in one notebook, saved with its outputs |
| `data/raw/` | The two UCI archives used by the notebook (checked against SHA-256 digests when loaded) |
| `results/tables/` | Every table produced by the notebook (CSV) |
| `results/figures/` | Every figure produced by the notebook (PNG) |
| `requirements.txt` | Package versions used for the reported run |

## Reproducing the results

```bash
git clone <repository-url>
cd <repository-folder>
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace interpretable_distillation.ipynb
```

Alternatively, open the notebook in Jupyter and choose *Run All*. A full run takes about five minutes on a
laptop and rewrites `results/`.

* The reported run used Python 3.11 with the package versions in `requirements.txt`; these versions
  support Python 3.9 to 3.12.
* Every random step uses the fixed seed 42, so a rerun reproduces the saved tables.
* If the archives in `data/raw/` are missing, the notebook downloads them from the UCI repository.

## Datasets

| Dataset | Samples | Features | Classes | Source |
|---|---|---|---|---|
| HTRU2 (pulsar candidates) | 17,898 | 8 | 2 | Lyon, R. (2015). UCI Machine Learning Repository. https://doi.org/10.24432/C5DK6R |
| Gas Sensor Array Drift | 13,910 | 128 | 6 | Vergara, A. (2012). UCI Machine Learning Repository. https://doi.org/10.24432/C5RP6W |

Both datasets are distributed by the UCI Machine Learning Repository under the CC BY 4.0 licence.

## Method in brief

1. **Split.** One class-stratified 60/20/20 train/validation/test split per dataset (seed 42).
2. **Exploratory analysis** on the training partition: class balance, feature scales and shapes,
   class separability, dependence between features, batch drift, and comparability of the partitions.
3. **Teacher.** `HistGradientBoostingClassifier` (scikit-learn), chosen among three predefined
   configurations by validation macro-F1 and then frozen.
4. **Distillation target.** The teacher's log-odds scores against a fixed reference class.
5. **Students.**
   * Sparse PCE-HDMR: degree-2 polynomial in a data-driven orthonormal basis with screened pairwise
     terms; terms selected by orthogonal matching pursuit.
   * Additive spline model: one cubic B-spline curve per feature, fitted by ridge regression.

   The complexity of each student is selected by validation score error.
6. **Test evaluation.** Accuracy, fidelity (teacher agreement and normalised score RMSE) with bootstrap
   intervals, complexity, per-class results and failure analysis.
7. **Explanations.** The students' closed-form Shapley attributions and Sobol' indices are compared with
   Monte Carlo permutation Shapley values of the teacher, all under the same product-of-marginals
   reference. The estimator is validated on a function with known Shapley values and on the students,
   whose exact attributions are known.

## Main results (test partition)

| Dataset | Model | Accuracy | Teacher agreement | Score NRMSE | Features used | Coefficients |
|---|---|---|---|---|---|---|
| HTRU2 | Teacher | 0.976 | – | – | 8 | – |
| HTRU2 | PCE | 0.980 | 0.986 | 0.351 | 8 | 32 |
| HTRU2 | GAM | 0.977 | 0.986 | 0.340 | 8 | 48 |
| Gas Sensor Array Drift | Teacher | 0.996 | – | – | 128 | – |
| Gas Sensor Array Drift | PCE | 0.961 | 0.962 | 0.717 | 74 | 160 |
| Gas Sensor Array Drift | GAM | 0.994 | 0.994 | 0.189 | 128 | 3,840 |

Agreement between the students' explanations and the teacher's (rank correlation of global feature
importance) is 0.98 (PCE) and 0.88 (GAM) on HTRU2, and 0.29 (PCE) and 0.13 (GAM) on the gas data.
The notebook and the report discuss these results and their limitations.

