# new-topic-classifiers
This project uses a fine-tuned BERT model to classify news headlines into four categories: World, Sports, Business, and Sci/Tech. It demonstrates NLP with Transformers and is deployed using Gradio and Streamlit for real-time predictions.


# BERT News Topic Classifier  
**Organization/Owner**: Personal ML Projects by [MARIA SAEED KHAN]  
**Repository**: `bert-news-classifier`

---

## 📌 Objective of the Task

The goal of this project is to build a machine learning model that classifies news text into predefined categories — **World**, **Sports**, **Business**, and **Sci/Tech** — using a fine-tuned BERT-based architecture. This serves as a practical demonstration of transfer learning on text classification and deploys the model in a user-friendly web interface.

---

## 🧪 Methodology / Approach

### 🔹 Dataset
A custom-labeled dataset was created consisting of news headlines mapped to four categories. Initial dataset included 10 examples and was later expanded to improve performance. The data was saved in CSV format and loaded using pandas.

### 🔹 Preprocessing
- Tokenization using `BertTokenizer` from Hugging Face
- Truncation and padding to uniform sequence length (`max_length=128`)
- Labels encoded to integers

### 🔹 Model
- Base Model: `bert-base-uncased`
- Fine-tuned using `BertForSequenceClassification`
- Optimizer: AdamW
- Loss: CrossEntropyLoss
- Epochs: 3
- Evaluation metrics: Accuracy and Weighted F1 Score

### 🔹 Deployment
Two UI variants were built:
- **Gradio**: Simple and fast API demo
- **Streamlit**: More customizable UI with animations and layout control

---

## 📊 Key Results / Observations

- **Accuracy Achieved**: 84.6%
- **F1 Score**: 0.846 (Weighted)
- Model learns well even with a small dataset (after expansion)
- Streamlit and Gradio both offer effective deployment solutions
- Gradio is quick to prototype, Streamlit is more flexible visually
- Disk space limitation may occur when saving the model (~1–2GB needed)

---

## 🖥️ How to Run

### Install requirements:
pip install -r requirements.txt
