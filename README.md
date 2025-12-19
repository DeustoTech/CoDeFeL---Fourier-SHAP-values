This repository contains a reproducible Python implementation comparing Fourier-based SHAP (Fourier-SHAP) and Kernel SHAP for explaining neural network predictions in a biomedical classification task.
The analysis is performed conditionally on age bins, while excluding age itself from the explanation variables, in order to isolate age-conditioned feature effects.

For more details about the experiments, please read Section 3 (Numerical experiments) in the paper https://arxiv.org/pdf/2511.00185.

The code is written in a python script .ipynb. Each code can be launched directly (for example) by using Anaconda. Please, make sure to create a kernel compatible with pytorch, sklearn, numpy, etc.
