# COMET
🧬 COMET

Few-shot Single-cell Annotation via Class-Consistent Quantization

📄 Paper: To be released

🧾 Overview

COMET is a few-shot single-cell annotation framework based on class-consistent quantization.

It is designed to address:

limited labeled samples in single-cell datasets
noise introduced by sequencing technologies
intra-class heterogeneity
weak inter-class separability

COMET introduces class-consistency constraints into latent quantization, enabling:

structured latent codebooks for each cell type
purified representations via discretization
high-quality metacell prototypes

This leads to improved performance in few-shot learning scenarios, especially for rare cell types.

💻 Hardware Requirements

COMET requires a standard workstation with GPU support.

Recommended configuration:

CPU: ≥16 cores
RAM: ≥32 GB
GPU: NVIDIA GPU (≥8GB VRAM recommended)
🖥️ System Requirements
OS: Windows 11 (tested)
Python: 3.9
CUDA: compatible version with PyTorch
⚙️ Installation
conda create -n comet python=3.9
conda activate comet

git clone https://github.com/yourname/COMET.git
cd COMET

pip install -r requirements.txt

🧩 Method Pipeline

COMET consists of three stages:

1. Representation Learning

An encoder maps single-cell data into a latent space.

2. Class-consistent Quantization

Constructs structured codebooks for each class and discretizes latent representations.

3. Metacell Construction

Aggregates purified representations into prototypes for classification.

📊 Expected Results

COMET achieves:

higher Accuracy (ACC)
improved Macro-F1
better clustering consistency

Especially in:

few-shot settings
rare cell-type recognition

🔗 Code Availability

Code will be released upon acceptance.

📬 Contact

For questions:

Email: woshili18wei@163.com
