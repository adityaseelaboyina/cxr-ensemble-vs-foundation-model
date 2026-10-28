# Is a CNN–Transformer Ensemble Worth It When a Chest X-Ray Foundation Model Is Available?

Code for the paper by Aditya Seelaboyina and Irfan Khan (Texas A&M University).

We compare a five-backbone ImageNet ensemble (ResNet18, DenseNet121, CoAtNet-0, ViT-B/16, Swin-B) with the EVA-X-S chest X-ray foundation model on the official patient-disjoint split of NIH ChestX-ray14. All model selection, fusion, calibration, and thresholds use validation data only, and every difference is estimated with a patient-level paired bootstrap.

| System | Test macro-AUROC |
|---|---|
| EVA-X-S (own recipe) | 0.828 |
| Five ImageNet models, averaged (shared recipe) | 0.814 |
| Five ImageNet models, averaged (EVA-X recipe) | 0.817 |
| All six, averaged (EVA-X recipe) | 0.826 |

## Layout

```
src/
  cxr_pipeline.py    data download and preparation, official split, shared-recipe training, evaluation
  evax.py            EVA-X-S model and weight loading, layer-wise learning-rate decay
  train_v3.py        training for the shared and EVA-X recipes, seeds, latency benchmark
  analysis_v3.py     fusion rules, out-of-fold validation, subset search, paired bootstrap, Holm correction
  verify_run.py      read-only integrity checks (split, patient overlap, predictions, early stopping)
  build_tables.py    builds every table and number in the paper from the result folders
  plot_perfinding.py per-finding figure
  stacking_diag.py   diagnostic for the full-stacking results
notebooks/
  CXR_Ensemble_Experiments.ipynb   main run: data, five ImageNet backbones, evaluation
  CXR_v3_Phase2.ipynb              EVA-X-S runs, ViT rerun, foundation-model analyses
  CXR_v3_Phase3.ipynb              same-recipe controls and seed replicates
tests/
  synth.py           synthetic data for testing the analysis code (not real results)
```

## Setup

Python 3.10+.

```
pip install -r requirements.txt
```

## Reproducing

Training was done on Google Colab with one A100. The notebooks expect a Drive folder `MyDrive/Colab Notebooks/cxr_experiment` with `cxr_pipeline.py` in it and the other files from `src/` in `cxr_experiment/v3`.

1. Run `CXR_Ensemble_Experiments.ipynb`. It downloads ChestX-ray14 through `kagglehub`, builds the official split, and trains the five ImageNet backbones.
2. Run `CXR_v3_Phase2.ipynb`, then `CXR_v3_Phase3.ipynb`.
3. Build the tables:

```
python src/build_tables.py --r1 <results> --v3 <results_v3> --v3b <results_v3b> --v3c <results_v3b> --out tables
```

## Data and weights

- NIH ChestX-ray14 is available from the NIH Clinical Center and on Kaggle (`nih-chest-xrays/data`). It is not redistributed here.
- EVA-X-S weights are downloaded from Hugging Face (`MapleF/eva_x`, file `eva_x_small_patch16_merged520k_mim.pt`), released by the EVA-X authors under Apache-2.0.
- ImageNet weights come from `timm`.

## Citation

```
@misc{seelaboyina2026ensemble,
  title  = {Is a CNN--Transformer Ensemble Worth It When a Chest X-Ray Foundation Model Is Available?},
  author = {Seelaboyina, Aditya and Khan, Irfan},
  year   = {2026},
  note   = {arXiv preprint}
}
```
