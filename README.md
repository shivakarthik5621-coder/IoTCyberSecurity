# A Fully Unsupervised Temporal Convolutional Autoencoder for IoT Intrusion Detection

## Overview

This repository contains the implementation of a **Temporal Convolutional Autoencoder (TCAE)** for unsupervised intrusion detection in Internet of Things (IoT) networks.

Unlike traditional supervised Intrusion Detection Systems (IDS), the proposed framework is trained **exclusively on benign network traffic**, enabling the detection of previously unseen attacks without requiring attack labels during training.

The model leverages **Residual Temporal Convolutional Networks (TCNs)** with causal dilated convolutions to capture long-range temporal dependencies efficiently while maintaining linear computational complexity.

---

## Paper Information

**Title:** A Fully Unsupervised Temporal Convolutional Autoencoder for IoT Intrusion Detection

**Authors:**

* Hrishikesh Harnoor
* K. Sai Nikhila
* K. Shiva Karthik Reddy
* Bidyapati Thiyam

**Conference:** IEEE

---

## Key Contributions

* Fully unsupervised intrusion detection framework.
* Training performed only on benign IoT traffic.
* Residual Temporal Convolutional Network architecture.
* Causal dilated convolutions for multi-scale temporal modeling.
* Percentile-based anomaly thresholding without attack labels.
* Lightweight architecture suitable for real-time IoT deployments.

---

## Dataset

This work uses the **CIC-IoT-2023** dataset.

Dataset characteristics:

* Realistic IoT network traffic
* Multiple attack categories
* Large-scale network flow features
* Benchmark dataset for IoT intrusion detection research

---

## Proposed Architecture

The framework consists of:

### 1. Data Preprocessing

* Remove invalid and missing samples
* Keep only numerical features
* Remove label-dependent attributes
* Min-Max normalization fitted on benign data only

### 2. Sequence Generation

Sliding window approach:

* Window Length: 20
* Stride: 3

### 3. Temporal Convolutional Autoencoder

#### Encoder

* Causal Conv1D layer
* Residual TCN Block (dilation = 1)
* Residual TCN Block (dilation = 2)
* Residual TCN Block (dilation = 4)

#### Latent Representation

* Conv1D bottleneck
* Latent dimension = 32

#### Decoder

* Symmetric residual TCN blocks
* TimeDistributed Dense reconstruction layer

### 4. Anomaly Detection

Anomaly score:

* Mean Absolute Reconstruction Error (MAE)

Threshold:

* 99.5th percentile of benign validation reconstruction errors

Decision rule:

```text
Error > Threshold  → Attack
Error ≤ Threshold → Benign
```

---

## Experimental Setup

| Parameter        | Value                    |
| ---------------- | ------------------------ |
| Window Length    | 20                       |
| Stride           | 3                        |
| Latent Dimension | 32                       |
| Batch Size       | 1024                     |
| Epochs           | 32                       |
| Optimizer        | Adam                     |
| Loss Function    | Mean Squared Error (MSE) |

---

## Results

### Performance Metrics

| Metric              | Value  |
| ------------------- | ------ |
| Accuracy            | 97.22% |
| Precision           | 99.98% |
| Recall              | 97.15% |
| F1 Score            | 98.55% |
| False Positive Rate | 0.50%  |

### Confusion Matrix

|               | Predicted Benign | Predicted Attack |
| ------------- | ---------------- | ---------------- |
| Actual Benign | 9,950            | 50               |
| Actual Attack | 9,558            | 325,946          |

---

## Model Complexity

| Property      | Value         |
| ------------- | ------------- |
| Parameters    | 162,503       |
| Model Size    | ~635 KB       |
| Training Time | 7–9 sec/epoch |
| Complexity    | O(L)          |

Advantages:

* Linear sequence scaling
* Fully parallel temporal computation
* Lower overhead compared to LSTM-based models
* Suitable for real-time deployment

---

## Repository Structure

```text
├── data/
│   ├── benign/
│   └── attack/
│
├── notebooks/
│
├── src/
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── results/
│   ├── figures/
│   ├── confusion_matrix.png
│   └── loss_curves.png
│
├── checkpoints/
│
├── requirements.txt
│
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/TCAE-IoT-IDS.git

cd TCAE-IoT-IDS

pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

---

## Evaluation

```bash
python evaluate.py
```

---

## Future Work

* Adaptive threshold selection
* Online streaming anomaly detection
* Lightweight edge deployment
* Hybrid graph-temporal architectures
* Enhanced zero-day attack resilience

---

## Citation

```bibtex
@inproceedings{harnoor2025tcae,
  title={A Fully Unsupervised Temporal Convolutional Autoencoder for IoT Intrusion Detection},
  author={Harnoor, Hrishikesh and Nikhila, K. Sai and Reddy, K. Shiva Karthik and Thiyam, Bidyapati},
  booktitle={IEEE Conference Proceedings},
  year={2025}
}
```

---

## License

This project is released under the MIT License.

---

## Acknowledgements

* CIC-IoT-2023 Dataset
* IEEE
* Amrita Vishwa Vidyapeetham
