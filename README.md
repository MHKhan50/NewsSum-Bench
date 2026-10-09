# NewsSum-Bench: Comparative Benchmarking of Abstractive Text Summarization using Transformer Architectures

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23264727.svg)](https://doi.org/10.5281/zenodo.23264727)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end research framework and benchmark evaluating state-of-the-art Transformer models for abstractive text summarization on news datasets.

---

## Project Overview

Abstractive text summarization generates concise, coherent, and contextual summaries of lengthy documents. This repository benchmarks multiple pre-trained sequence-to-sequence and encoder-decoder models (**T5-base**, **BART-base**, **PEGASUS**, **mT5-base**, and **BERTsum**) on news data.

### Key Highlights & Architecture

* **Multi-Model Fine-Tuning Pipeline:** Standardized training workflow using Hugging Face `Seq2SeqTrainer` across diverse transformer backbones.
* **Data Processing & Cleaning:** Custom preprocessing pipeline for sentence tokenization, length verification, and cleaning to handle maximum context sizes.
* **Quantitative Evaluation:** Rigorous ROUGE metric scoring (`ROUGE-1`, `ROUGE-2`, `ROUGE-L`, and `ROUGE-Lsum`) across test splits.
* **Interactive Prototype:** Web interface built with Streamlit for real-time article summarization and qualitative human evaluation.

---

## Repository Structure

```text
NewsSum-Bench/
├── notebooks/                  # Model fine-tuning & evaluation notebooks
│   ├── 01_T5_Base_Summarization.ipynb
│   ├── 02_BART_Base_Summarization.ipynb
│   ├── 03_PEGASUS_Summarization.ipynb
│   ├── 04_mT5_Base_Summarization.ipynb
│   └── 05_BERT_Summarization.ipynb
├── src/                        # Core Python scripts & Streamlit UI
│   ├── preprocessing.py
│   └── app.py
├── data/                       # Preprocessed dataset files
│   └── Preprocessed1_news_output.csv
├── docs/                       # Comprehensive research report & documentation
│   └── Abstractive_News_Summarization_Report.pdf
├── README.md
├── LICENSE                     # MIT License
└── requirements.txt            # Python dependencies
