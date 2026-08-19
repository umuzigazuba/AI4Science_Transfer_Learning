# Transfer Learning for Astronomical Tidal Disruption Event Classification

Classify tidal disruption events (TDEs) in a dataset also containing supernovae (SNe) and active galactic nuclei (AGN) using convolutional and gated recurrent neural networks, with transfer learning from Gaussian Process-augmented synthetic light curves to the simulated Mallorn light curves.

Data is from the [MALLORN Astronomical Classification Challenge](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/leaderboard) on Kaggle, combined with additional synthetic light curves generated for pretraining (see [Data](#data) below).

## Project Structure

```
AI4Science_Transfer_Learning/
├── data/                                       # Mallorn light curves and calculated features (needs to be downloaded first)
├── models/                                     # saved model checkpoints
│   ├── pretrain/                               # models pretrained on augmented light curves
│   ├── finetune/                               # models finetuned on the original Mallorn light curves
│   └── original/                               # models trained on the original light curves without transfer learning
├── notebooks/
│   ├── training_models.ipynb                     # pretrain/finetune/test runs and result plots
│   └── visualisation_generated_samples.ipynb   # visualisation of generated vs. original light curve properties
├── predictions/                                # test set predictions, one CSV per model
├── src/
│   ├── classification/               
│   │   ├── model.py                            # model
│   │   ├── datamodule.py                       # LightningDataModule (loading, padding, band encoding)
│   │   └── trainer.py                          # MultiClassClassifier (training/validation/prediction loop)
│   └── data_generation/                        # synthetic light curve generation for pretraining
│       ├── generation_functions.py             # GP fitting, extinction, time dilation, sample generation
│       └── event_window.py                     # detection of the transient event window within a light curve
├── __main__.py                                 # CLI entry point for generating synthetic samples
├── .gitignore
├── README.md
└── requirements.txt
```

The classes and functions used throughout the project are defined in the `src/` scripts; the notebooks only call into them to produce results and plots.

## Installation

Clone the repository:

```bash
git clone https://github.com/umuzigazuba/AI4Science_Transfer_Learning.git
cd AI4Science_Transfer_Learning
```

Create a conda environment and install dependencies:

```bash
conda create -n transfer_learning_env 
conda activate transfer_learning_env

pip install -r requirements.txt
```

## Data

Light curves, redshifts, extincintion parameters, and target labels come from the [MALLORN Astronomical Classification Challenge](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/leaderboard). Download the competition data and place it in the `data/` folder.


The `data/` folder also contains the synthetic lightcurves, parameters and labels generated using the .py`generation_functions.py` script. For each class (TDE, SN and AGN) 2000 samples were generated.

## Usage

Generate synthetic light curve samples for a given spectral type, used for pretraining:

```bash
python -m AI4Science_Transfer_Learning <spectral_type> <number> <output_file_name> --data_dir data --active_bands ugirz
```

Pretrain, finetune, and evaluate the CNN and GRU classifiers, and generate the result plots:

```bash
notebooks/lightning_cnn.ipynb
```

Visualise the generated lightcurves and compare their properties (redshift, EBV) against the original Mallorn light curves:

```bash
notebooks/visualisation_generated_samples.ipynb
```
