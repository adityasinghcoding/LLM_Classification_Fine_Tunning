# LLM Preference Prediction - Kaggle Competition

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/competitions/llm-classification-finetuning)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30%2B-yellow)](https://huggingface.co/docs/transformers)

A solution for predicting human preferences between LLM-generated responses in head-to-head chatbot battles.

## 📌 Overview

This repository contains code for the Kaggle competition !["LLM Classification Finetuning"](https://www.kaggle.com/competitions/llm-classification-finetuning), where the goal is to predict which of two AI-generated responses humans will prefer. The solution uses fine-tuned transformer models to analyze conversational data from Chatbot Arena.

## 🚀 Key Features

- **Transformer-based Architecture**: Utilizes DeBERTa for sequence classification
- **Bias Mitigation**: Handles position bias through data augmentation
- **Efficient Training**: Mixed-precision training with Hugging Face `Trainer`
- **Probability Calibration**: Softmax with label smoothing
- **Reproducible**: Standardized preprocessing and training pipeline

## 📋 Requirements

```bash
Python 3.8+
pip install -r requirements.txt
```


