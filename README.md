# Fourier-SHAP for Feature Attribution in Clinical Classification

This repository accompanies the paper [*SHAP Values through General Fourier Representations: Theory and Applications*](https://arxiv.org/abs/2511.00185).
It provides a reproducible implementation of Fourier-SHAP, a spectral approach to SHAP feature attribution for 
predictors with discrete or multi-valued inputs. The included experiment trains a neural-network classifier on a 
coded stroke dataset, computes age-conditioned explanations on the logit scale, and compares Fourier-SHAP with Kernel SHAP.

## Motivation

SHAP values offer a principled, Shapley-value-based way to distribute a model prediction among its input features. 
Exact SHAP computation, however, requires evaluating many feature coalitions and becomes prohibitively expensive as 
the number of variables grows. Existing Fourier formulations primarily focus on binary inputs, whereas practical 
data often include categorical and ordinal features with several possible values.

The objective is to obtain a spectral description of SHAP values that accommodates general discrete, multi-valued 
features and non-uniform product distributions, while clarifying how model approximation affects feature attributions. 
This can make explanations more computationally efficient without losing the dominant attribution patterns.

## Approach implemented

The notebook trains a neural-network stroke classifier, builds a sparse Fourier surrogate of its logit outputs, 
and derives Fourier-SHAP values from the learned spectral coefficients. It compares these values with Kernel SHAP 
in eight age bins. Age is used to define the strata but is excluded from the explained variables, allowing the analysis 
to focus on the contributions of the remaining clinical and demographic features conditional on age. The workflow 
reports mean absolute SHAP scores, rankings, agreement between explainers, runtime, peak memory use, and diagnostic figures.

## Repository contents

| File | Purpose |
| --- | --- |
| `SHAPFourierdef.ipynb` | Complete experiment: data preparation, neural-network training, sparse Fourier surrogate construction, Fourier-SHAP and Kernel SHAP computation, age-bin comparisons, and output generation. |
| `stroke_coded_binned_gender012.csv` | Coded clinical stroke-classification dataset used by the notebook. |

The notebook creates a timestamped `OutExp2_<timestamp>/` directory containing generated tables and figures.

## Requirements

Use Python 3.10 or newer with Jupyter and the following packages:

```bash
pip install jupyter numpy pandas matplotlib scikit-learn shap
```

The notebook can additionally evaluate XGBoost and CatBoost baselines when these optional packages are installed:

```bash
pip install xgboost catboost
```

## Running the simulations

From this directory, launch Jupyter:

```bash
jupyter notebook
```

Open `SHAPFourierdef.ipynb` and run all cells from top to bottom. The notebook reads `stroke_coded_binned_gender012.csv`, 
trains the neural classifier and spectral surrogate, then computes Fourier-SHAP and Kernel SHAP explanations for every configured age bin.

The main experimental settings are collected in the `Config` class near the beginning of the notebook. In particular, `KERNEL_BIN_SUB_N` 
and `KERNEL_NSAMPLES` control the cost of Kernel SHAP; lower values provide a quicker exploratory run at the expense of sampling accuracy.

## Reference

R. Morales, *SHAP Values through General Fourier Representations: Theory and Applications*, 2025. 
The local manuscript is available on [arXiv:2511.00185](https://arxiv.org/abs/2511.00185).

## Funding

This project has received funding from the European Research Council (ERC) under the European Union's Horizon 2030 
research and innovation programme (grant agreement No. 101096251, CoDeFeL).
