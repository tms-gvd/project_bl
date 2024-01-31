Practical Bayesian optimization in the presence of outliers
=
Paper by Martinez-Cantin et al., 2017.

This repository: code I implemented to reproduce the results of the paper. The code is written in Python 3.10 and uses the following libraries:
```
numpy torch gpytorch matplotlib scipy tqdm lhsmdu
```

The code is organized as follows:
- `main.py`: main function to run the experiments.
- `gp_models.py`: implementation of the models.
- `data_generator.py`: implementation of the data generator.
- `utils.py`: utility functions.
- `plot.py`: functions to plot the results.
- `train.py`: functions to train a GP.

Some experiments are also present in two notebooks: `gp_student.ipynb` for the study of GP with Student-t likelihood, and `bo.ipynb` for the BO experiments.

Gif for the BO experiments:

### 1D BO

![hippo][https://github.com/tms-gvd/project_bl/blob/main/1d_bo.gif]

### 2D BO

![hippo][https://github.com/tms-gvd/project_bl/blob/main/2d_bo.gif]