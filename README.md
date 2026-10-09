# NewsSum-Bench: Comparative Benchmarking of Abstractive Text Summarization using Transformer Architectures

An end-to-end research framework and benchmark evaluating state-of-the-art Transformer models for abstractive text summarization on news datasets.

---

## 📌 Project Overview
Abstractive text summarization generates concise, coherent, and contextual summaries of lengthy documents. This repository benchmarks multiple pre-trained sequence-to-sequence and encoder-decoder models (**T5-base**, **BART-base**, **PEGASUS**, **mT5-base**, and **BERTsum**) on news data.

### Key Highlights:
* **Multi-Model Fine-Tuning Pipeline:** Standardized training workflow using HuggingFace `Seq2SeqTrainer`.
* **Data Processing & Cleaning:** Custom preprocessing pipeline for sentence tokenization and length verification.
* **Quantitative Evaluation:** ROUGE metric scoring (ROUGE-1, ROUGE-2, ROUGE-L) across test splits.
* **Interactive Prototype:** Web interface built with **Streamlit** for real-time article summarization.

---

## 📁 Repository Structure

```text
NewsSum-Bench/
├── notebooks/                  # Model fine-tuning & evaluation notebooks
│   ├── 01_T5_Base_Summarization.ipynb
│   ├── 02_BART_Base_Summarization.ipynb
│   ├── 03_PEGASUS_Summarization.ipynb
│   ├── 04_mT5_Base_Summarization.ipynb
│   └── 05_BERT_Summarization.ipynb
├── src/                        # Core Python scripts
│   ├── preprocessing.py
│   └── app.py                  # Streamlit application UI
├── data/                       # Preprocessed dataset
│   └── Preprocessed1_news_output.csv
├── docs/                       # Research Report
│   └── Abstractive_News_Summarization_Report.pdf
├── README.md                   # Documentation & Benchmarks
├── LICENSE                     # MIT License
└── requirements.txt            # Python Dependencies
