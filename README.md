# Human vs Machine Summarization of Clinical Notes
Evaluating Predictive Signal Retention for 30-Day Readmission After Joint Arthroplasty

This repository contains the complete data preprocessing pipeline, generative model architectures, and evaluation frameworks required to assess the efficacy of LongT5 and CVAE in compressing clinical notes and handling severe medical class imbalances.

---

## 1. Research and Selection of Methods

### Objectives & Literature Review
The primary objective of this pipeline is to compress extremely lengthy unstructured clinical notes into concise "Brief Hospital Course" (BHC) summaries while preserving the critical signals necessary for downstream 30-day readmission prediction. 

Standard transformer models (like BERT or T5-base) are constrained by a 512-token limit. Our data exploration on the MIMIC-IV joint arthroplasty cohort revealed that the clinical notes average **1,013 tokens**, with outliers exceeding 5,000 tokens. To avoid severe data truncation, we benchmarked extended-context models and selected **LongT5** due to its Transient Global (TGlobal) attention mechanism, which allows efficient processing of up to 4,096 tokens.

### Framework Selection
- **Hugging Face `transformers`:** Selected for its robust implementation of LongT5 and the highly optimized `Seq2SeqTrainer` API.
- **PyTorch:** Selected as the core tensor library for custom architecture development (CVAE).
- **Scikit-Learn:** Selected to rapidly establish a Bag-of-Words (BoW) + Logistic Regression global baseline to ground our predictive experiments.

### Preliminary Experiments & Adjustments
Before committing to this full-scale implementation, we conducted small-scale testing (e.g., testing LongT5 memory limits and discovering label leakage). Please see **[docs/preliminary_experiments.md](docs/preliminary_experiments.md)** for our complete findings and how we adjusted our architecture based on those experiments.

---

## 2. Model Implementation & Procedures

### A. Dataset Preparation (`src/preprocess.py`)
To ensure high data quality and avoid data contamination, our preprocessing script:
1. **Masks Label Leakage:** Uses regular expressions to replace explicit planned readmission phrases (e.g., "scheduled for second stage", found in ~1.2% of notes) with the `[MASK]` token.
2. **Stratified Splitting:** Splits the clean dataset into `train.csv` (80%), `val.csv` (10%), and `test.csv` (10%). Stratification ensures our severe 9.95% minority class ratio is strictly maintained across all splits.

### B. Global Baseline & Evaluation Models (`src/models/baseline_model.py` & `clinical_longformer_model.py`)
- **Global Baseline (Experiment 0):** We implemented a standard Bag-of-Words (TF-IDF) + Logistic Regression classifier on raw notes to establish floor predictive performance.
- **Classification Evaluator:** We utilize `yikuan8/Clinical-Longformer`. As per our experimental design, we freeze the base model weights, utilizing it solely as a fixed feature extractor to ensure a fair classification comparison between Raw Notes, Human BHC, and Machine BHC.

### C. Generative Summarization (`src/models/longt5_model.py` & `scripts/train_longt5.py`)
- **Architecture:** We utilize `google/long-t5-tglobal-base` initialized with pre-trained weights.
- **Training Configurations:** Defined within `Seq2SeqTrainingArguments`. We explicitly enforce a minimal batch size (`per_device_train_batch_size=1`) and gradient accumulation to prevent Out-Of-Memory (OOM) errors.

### D. Generative Augmentation (`src/models/cvae_model.py`)
- **Architecture:** A fully custom Deep Conditional Variational Autoencoder built in PyTorch.
- **Rationale:** Because 30-day readmission is a minority event (9.95%), the CVAE takes fixed-size dense text embeddings and generates synthetic representations of the minority class in the latent space to balance the training data for downstream classification.

### E. Evaluation & Metrics (`scripts/evaluate.py`)
Our evaluation framework measures two distinct outcomes:
1. **Summarization Quality:** We utilize the **ROUGE** metric suite (ROUGE-1, ROUGE-2, and ROUGE-L) to evaluate lexical overlap.
2. **Classification Performance:** We compute the **AUROC** (Area Under the Receiver Operating Characteristic Curve) to evaluate downstream predictive performance.

---

## 3. Repository Organization & Code Execution

```text
genai-clinical-readmission/
├── docs/
│   ├── merged_dataset_sample.txt       # Auto-generated preview of the merged dataframe
│   └── preliminary_experiments.md      # Findings & adjustments from small-scale testing
├── experiments/
│   ├── 01_test_longt5_feasibility.py   # Validates LongT5 with real clinical data
│   ├── 02_test_baseline_imbalance.py   # Validates class_weight adjustments
│   └── 03_data_leakage_eda.py          # Extracts leakage statistics 
├── data/                    
│   ├── README.md                       # DETAILED REPRODUCIBILITY GUIDE (SQL & Wget)
│   └── processed/                      # Contains train.csv, val.csv, test.csv
├── src/                     
│   ├── merge_datasets.py               # Joins structured cohort with unstructured BHC text
│   ├── preprocess.py                   # Pipeline for data cleaning, leakage masking, and splitting
│   └── models/
│       ├── baseline_model.py           # Bag-of-Words + Logistic Regression baseline
│       ├── clinical_longformer_model.py# Frozen feature extractor for classification
│       ├── longt5_model.py             # HuggingFace LongT5-base wrapper
│       └── cvae_model.py               # PyTorch Conditional Variational Autoencoder (CVAE)
├── scripts/                 
│   ├── train_longt5.py                 # Main training script using HuggingFace Seq2SeqTrainer
│   └── evaluate.py                     # Script for computing ROUGE and AUROC metrics
├── requirements.txt                    # Required Python packages
└── README.md                           # This document
```

### Setup Instructions
This project is designed to be executed within **Google Colab**. 

1. Upload the entire `genai-clinical-readmission` folder to your Google Drive and mount it in Colab.
2. Select a GPU runtime.
3. Install the specific project dependencies:
```bash
!pip install -r requirements.txt
```

### Running the Pipeline
Ensure your raw data files (`cohort_final_v2.csv` and `mimic-iv-bhc.csv`) are placed in the root of the repository before executing the pipeline. For instructions on how to acquire these two files from BigQuery and PhysioNet, please refer to our **[Data Acquisition Guide](data/README.md)**. 

Run the following commands inside your Colab notebook:

**1. Merge Raw Datasets:**
```bash
!python src/merge_datasets.py
```
*(This produces `merged_cohort_text_data.csv` which is required for downstream steps).*

**2. Preprocess Data:**
```bash
!python src/preprocess.py
```

**3. Evaluate Global Baseline (BoW + LR):**
```bash
!python src/models/baseline_model.py
```

**4. Train LongT5 Model:**
```bash
!python scripts/train_longt5.py
```
