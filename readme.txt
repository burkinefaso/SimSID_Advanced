# SimSID_Advanced

This repository contains the advanced version of the SimSID model, originally developed for anomaly detection in medical images. In this project, the algorithm is adapted from its original application on chest radiographs to detect anomalies in brain MRI 2D slices.

## Background

The base model is derived from the work by Xiang, T., et al. (2024), titled _"Exploiting structural consistency of chest anatomy for unsupervised anomaly detection in radiography images"_, published in the IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 46(9), 6070–6081. DOI: [10.1109/TPAMI.2024.3382009](https://doi.org/10.1109/TPAMI.2024.3382009). The original method utilizes structural consistency in chest anatomy for anomaly detection in radiographs. This model is extended in this repository to work with **brain MRI 2D slices** for the task of anomaly detection in brain imaging.

## Purpose

The goal of this project is to enhance the SimSID algorithm to perform **unsupervised anomaly detection** on **brain MRI** images. This approach leverages the same structural consistency principles used in chest radiograph anomaly detection, but applies them to **brain MRI scans**. The model is trained and tested on MRI datasets to identify abnormalities, which could potentially include tumors, lesions, or other unusual structures in the brain.

## How It Works

SimSID_Advanced uses a **generative adversarial network (GAN)** to perform anomaly detection. The model first generates a reconstruction of a given MRI slice and compares it with the original input. Significant differences between the original and reconstructed images indicate anomalies. This methodology is entirely **unsupervised**, meaning it does not require labeled data.

## Datasets

The model is trained using publicly available brain MRI datasets, including **IXI** and **OAS1**, to detect anomalies in various brain regions.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/burkinefaso/SimSID_Advanced.git
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Follow the instructions in the provided Jupyter notebooks to train the model and run anomaly detection on MRI data.

## Usage

- The primary script for training the model is `train.py`. It allows the user to specify various training parameters, including dataset, model architecture, and hyperparameters.
- Anomaly detection is performed using the `detect_anomalies.py` script. This script takes MRI slices as input and identifies anomalies using the trained model.

## Contribution

Feel free to fork this repository, submit issues, or contribute improvements. Please ensure that any pull requests are properly tested.

## License

This project is licensed under the [MIT License](LICENSE).
