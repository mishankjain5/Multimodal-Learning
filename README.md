# Multimodal Learning – CILP Assessment

This repository contains my solution for the **CILP Assessment – Multimodal Learning** assignment, extending the NVIDIA DLI multimodality workshop to RGB–LiDAR fusion, architectural ablations, and contrastive pretraining on the cubes vs spheres dataset.

## Repository structure

```text
Multimodal-Learning/
  notebooks/
    01_dataset_exploration.ipynb        # Task 2 – FiftyOne visualization
    02_fusion_comparison.ipynb          # Task 3 – Late vs intermediate fusion
    03_strided_conv_ablation.ipynb      # Task 4 – MaxPool2d vs strided conv
    04_final_assessment.ipynb           # Task 5 – CILP assessment performance

  src/
    __init__.py
    datasets.py                          # Dataset classes and transforms
    models.py                            # Fusion and CILP model architectures

  checkpoints/
    cilp_backbone_contrastive.pth        # Trained CILP backbone
    rgb_to_lidar_projector.pth           # RGB→LiDAR projector
    rgb_to_lidar_classifier.pth          # Final classifier

  results/
    Task_1_exp_track_with_wandb/…
    Task_2_dataset_vis_with_fo/…
    Task_3_fusion_architecture_comparison/…
    Task_4_MaxPool2D_to_Strided_Conv_Ablation/…
    Task_5_CILP_Assessment_Performance/…

  requirements.txt
  README.md
```
All training and analysis code lives in `src/` and is orchestrated from the notebooks to keep experiments modular and avoid code duplication, as requested in Task 6.

## Environment and setup

Target environment:

- Python 3.11+
- PyTorch 2.0+ with CUDA support
- Tested on Kaggle GPU (Tesla T4) and compatible with Google Colab.

Install dependencies locally:

```bash
git clone (https://github.com/mishankjain5/Multimodal-Learning.git)
cd Multimodal-Learning
pip install -r requirements.txt
```
### Dataset setup

The assessment dataset should follow the structure specified in the assignment.

```text
data/assessment/
  cubes/
    rgb/
      0000.png
      0001.png
      ...
    lidar/
      0000.npy
      0001.npy
      ...
  spheres/
    rgb/
      ...
    lidar/
      ...
```
To run notebooks:

1. Place the dataset under `data/assessment` (or update the `DATA_ROOT` path in each notebook).
2. On Colab or Kaggle, mount your dataset (Google Drive or Kaggle dataset) and point `DATA_ROOT` to the mounted location.

## Weights & Biases tracking

All experiments are logged to Weights & Biases for Task 1 and for tracking Tasks 3–5.

- **Project:** `cilp-extended-assessment`
- **Username:** `jain5`
- **Public project link:** [W&B project] (https://wandb.ai/jain5-university-of-potsdam/cilp-extended-assessment)

Each run logs:

- Training and validation loss/accuracy per epoch
- Learning rate, batch size, fusion strategy, model architecture name, and parameter count
- GPU memory usage and training time
- For CILP, the similarity matrix and sample predictions.

To use W&B yourself, create an account, run `wandb.login()` in the notebooks, and paste your API key when prompted.

## Summary of results

Key results required by the assignment are summarized below; exact tables are in the notebooks and exported CSV files in `results/`.

| Task | Description                                      | Key metric(s)                                 | Result (approx.)                   |
|------|--------------------------------------------------|-----------------------------------------------|------------------------------------|
| 1    | W&B experiment tracking                          | Metrics, configs, visualizations              | Completed                          |
| 2    | FiftyOne dataset visualization                   | Class distribution, modality pairing, stats   | Completed (plots + screenshots)    |
| 3    | Fusion architecture comparison                   | Best val acc / loss (Intermediate Hadamard)   | ≈99.9% val acc, very low val loss  |
| 4    | MaxPool2d → strided conv ablation               | Val loss, accuracy, params, time comparison   | Strided conv variant preferred     |
| 5    | CILP assessment performance                      | CILP val loss, projector MSE, classifier acc  | CILP loss < 3.5, acc > 95%         |

The precise numbers and comparison tables are printed at the end of each notebook and saved under `results/Task_*` as PNG and CSV exports.

## How to reproduce experiments

### Run in Colab or Kaggle

1. Start a **GPU** notebook.
2. Clone the repo and install dependencies:

   ```bash
   !git clone https://github.com/mishankjain5/Multimodal-Learning.git
   %cd Multimodal-Learning
   !pip install -r requirements.txt
   ```
Mount your data and set `DATA_ROOT` in the notebooks to point to `data/assessment`.
4. Run the notebooks in order:

   1. `01_dataset_exploration.ipynb` – FiftyOne visualization and dataset statistics.
   2. `02_fusion_comparison.ipynb` – late vs intermediate fusion experiments.
   3. `03_strided_conv_ablation.ipynb` – MaxPool2d vs strided convolution.
   4. `04_final_assessment.ipynb` – CILP contrastive pretraining, projector training, final classifier.

The notebooks set a global random seed and use shared dataset/model code from `src/`, so runs with the same hyperparameters are reproducible within normal stochastic variation.

### Reproduce final metrics from checkpoints

To reproduce the final Task‑5 metrics without full retraining:

1. Ensure the three `.pth` files in `checkpoints/` are present.
2. Open `04_final_assessment.ipynb` and run the evaluation section that loads:

   - `cilp_backbone_contrastive.pth`
   - `rgb_to_lidar_projector.pth`
   - `rgb_to_lidar_classifier.pth`

3. The notebook evaluates the CILP backbone, projector MSE, and RGB→LiDAR classifier on several validation batches and prints the final metrics (CILP loss, projector MSE, classifier accuracy).

## Notes and limitations

- For hyperparameter search, many experiments are run on a **subset** of the dataset to save compute; this is documented in the notebooks and W&B configs.
- Full‑dataset runs are used for the final CILP and classifier metrics reported above.
- The project follows the recommended modular structure (separate `src/datasets.py` and `src/models.py`), while training and visualization are controlled from the notebooks rather than separate `training.py` / `visualization.py` modules.

---
