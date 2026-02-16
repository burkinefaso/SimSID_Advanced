
# SimSID_Advanced - README

This project contains an advanced version of the SimSID model, originally developed for anomaly detection in medical images. In this project, the algorithm is adapted from its original application on chest radiographs to detect anomalies in brain MRI 2D slices, with several enhancements to improve its performance.

## Project Summary

- **SimSID** algorithm was originally developed for anomaly detection in chest radiographs. This project adapts the SimSID model to perform anomaly detection on **brain MRI** images.
- **Enhancements**: 
  - The model was successfully adapted from chest radiographs to brain MRI slices.
  - The model’s performance was enhanced by training it with different datasets.
  - The training and testing processes can be started via **command line**.

## Datasets Used

This project uses **unsupervised anomaly detection** to detect anomalies in brain MRI images. The datasets used are as follows:

- **FCP** (various brain MRI datasets)
- **ATLAS R2.0**
- **BraTS2017**
- **REMBRANDT**
- **UCSF-BMSR**
- **BraTS2023**

These datasets were used in both the training and testing phases of the model.

## Requirements

To run this project, the following software requirements are needed:

- **Python 3.x**
- **PyTorch** (recommended for GPU usage)
- **NumPy**
- **Matplotlib** (for visualizations)
- **Seaborn** (for visualizations)
- **scikit-learn** (sklearn)
- **lpips** (for LPIPS metrics)
- **TQDM** (for progress bars during training)

To install the required dependencies:

```bash
pip install -r requirements.txt
```

## Training Instructions

### 1. Clone the Repository

Clone the project to your local machine:

```bash
git clone https://github.com/burkinefaso/SimSID_Advanced.git
```

### 2. Directory Setup

Make sure the directories for training and testing scripts are correctly set up. The general directory structure should be as follows:

- **Datasets**: 
  - Keep training and testing datasets in the **data** folder.
  - Normal and anomalous images should be placed in separate directories. Example directory structure:

```
data/
  ├── normal/   # Normal MRI images
  ├── anomaly/  # Anomalous MRI images
  └── val/      # Validation set
```

- **Model Checkpoint**: Directory where the trained model is saved:
```
checkpoints/
  └── model_epoch800.pth  # Trained model weights
```

- **Results**: Directory where the test results will be saved:
```
results/
  └── experiment_1/    # Results of different test experiments
```

**Note:** Users must update **data directories** and **checkpoint paths** accordingly. You can define the paths as follows:

```python
NORMAL_ROOT = "data/normal"
ANOMALY_ROOT = "data/anomaly"
CHECKPOINT_PATH = "checkpoints/model_epoch800.pth"
RESULTS_BASE = "results"
```

### 3. Training Command

To train the model, use the following command. This will start the training process by running **`main.py`**.

```bash
cd /path/to/SimSID_Advanced
python main.py --config configs.karma4_dev --exp karma_13_experiment
```

#### Command Parameters:

- **`--config`**: Specifies the **config file** used for training. For example, `karma4_dev`.
- **`--exp`**: Sets the experiment name. This parameter allows you to keep different parameters for different experiments.

### 4. Testing Command

To test the model, you can use the following **test scripts**. The recommended script for testing is **`main_test_fully.py`**. This script covers the entire testing process and evaluations.

Test commands can be run as follows:

```bash
python main_test_fully.py --checkpoint /path/to/model_checkpoint.pth --data_dir /path/to/test_data
```

#### Parameters:
- **`--checkpoint`**: Path to the trained model (e.g., `model_epoch800.pth`).
- **`--data_dir`**: Path to the test data directory containing anomalous and normal MRI slices.

### 5. Results Visualization

You can visualize the test results using the following functions:
- **Histogram**
- **KDE Density Plot**
- **ROC Curve**
- **Precision-Recall Curve (PRC)**

These functions can be used to save and visualize the results in the **`results`** directory.

### 6. Directory Adjustments

- **Dataset Paths**: Parameters like `NORMAL_ROOT`, `ANOMALY_ROOT`, `CHECKPOINT_PATH` must be updated according to the test data directory location.
- **Checkpoint Path**: `CHECKPOINT_PATH` should be updated to the path where the trained model weights are saved.

### Contribution

Feel free to fork this repository, submit issues, or contribute improvements. When submitting a pull request, ensure that your code is properly tested.

### License

This project is licensed under the [MIT License](LICENSE).
