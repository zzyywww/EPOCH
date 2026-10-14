# EPOCH: Sequence-Based Epitope and Paratope Prediction

EPOCH is a sequence-based framework for the simultaneous prediction of antibody paratopes and antigen epitopes. The model was developed by fine-tuning ProtBERT on paired antibody–antigen sequences and predicts the probability that each amino acid residue participates in the binding interface.

EPOCH requires only antibody and antigen sequences during inference and does not depend on experimentally determined or predicted three-dimensional structures.

## Key Features

- Simultaneous residue-level prediction of antibody paratopes and antigen epitopes
- Paired modeling of antibody and antigen sequences
- No requirement for three-dimensional structural information during inference
- Support for conventional Fab/Fv antibodies and single-domain antibodies
- Prediction across complete antibody sequences without requiring predefined CDR annotations
- Maximum input length of 40,000 tokens
- ROC-AUC of 0.9350 on the held-out test set
- Residue-level probabilities and binary interaction-site labels

> EPOCH predicts whether individual antibody and antigen residues participate in the binding interface. It does not predict direct physical contacts between specific antibody–antigen residue pairs.

Experimentally resolved antibody–antigen complex structures are used to generate residue-level labels during model development. During inference, EPOCH requires only paired antibody and antigen sequences.

## Development Environment

EPOCH was developed and tested using the following environment:

| Component | Version |
|---|---:|
| Python | 3.9.16 |
| CUDA environment | 12.2 |
| PyTorch | 1.13.1 |
| Transformers | 4.26.1 |
| Datasets | 2.9.0 |
| pandas | 1.5.3 |
| tqdm | 4.64.1 |

## System Requirements

### Software

- Python 3.9
- PyTorch 1.13.1
- Transformers 4.26.1
- Other Python dependencies listed in `requirements.txt`

### Hardware

- An NVIDIA GPU is recommended for efficient inference.
- CPU inference may be possible but can be substantially slower.
- GPU memory requirements depend on the combined length of the antibody and antigen sequences.
- Processing very long sequence pairs may require additional GPU memory despite the configured maximum input length of 40,000 tokens.

> CUDA 12.2 describes the environment in which EPOCH was tested. Users should install a PyTorch build compatible with their NVIDIA driver and local CUDA environment.

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/zzyywww/EPOCH.git
cd EPOCH
```

Install all required Python dependencies:

```bash
pip install -r requirements.txt --ignore-installed
```

Download and extract the pretrained EPOCH model:

```bash
wget https://i.uestc.edu.cn/EPOCH.zip
unzip EPOCH.zip
```

> **Important:** The extracted model directory named `EPOCH` must be located in the same directory as the `EPOCH.py` script.


Windows users can download the model directly by opening the following URL in a web browser:

```text
https://i.uestc.edu.cn/EPOCH.zip
```

The expected project structure is:

```text
EPOCH/
├── EPOCH.py
├── EPOCH/
│   └── pretrained model files
├── example/
│   └── example_seq.txt
├── figures/
│   └── EPOCH_workflow.png
├── requirements.txt
└── README.md
```

## Input Format

The input must be a tab-delimited text file containing the following columns:

| Column | Description |
|---|---|
| `id` | Unique identifier for the antibody–antigen sequence pair |
| `antibody_seq1` | Heavy-chain sequence or single-domain antibody sequence |
| `antibody_seq2` | Light-chain sequence; leave blank for single-chain or single-domain antibodies |
| `antigen_seq` | Antigen amino acid sequence |

Each row represents one antibody–antigen sequence pair. The column names must remain unchanged.

An example input file is provided at:

```text
./example/example_seq.txt
```

## Usage

Run EPOCH using:

```bash
python EPOCH.py <inputfile>
```

Example:

```bash
python EPOCH.py ./example/example_seq.txt
```

## Output

Prediction results are automatically saved in the `results` directory.

For each valid antibody and antigen residue, the output file reports:

- Residue-level binding probability
- Predicted binary label

Binary labels are assigned using the default classification threshold of `0.0775`.You can modify the source code in **EPOCH.py** to save your results more customistically. 

| Label | Interpretation |
|---:|---|
| `1` | Predicted interaction-site residue |
| `0` | Predicted non-interacting residue |

The output directory and other prediction settings can be customized in `EPOCH.py`.
