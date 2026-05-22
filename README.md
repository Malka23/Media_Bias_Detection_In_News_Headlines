# 📰 Media Bias Detection in News Headlines

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn)
![HuggingFace](https://img.shields.io/badge/HuggingFace-SBERT-FFD21F?style=flat-square&logo=huggingface)
![NLP](https://img.shields.io/badge/NLP-Text_Classification-blueviolet?style=flat-square)

> Enter any news headline — get an instant **Left / Center / Right** political bias prediction using sentence embeddings and a trained classifier.

---

## 🧠 About the Project

Political bias in media is subtle, pervasive, and hard to detect manually. This project builds an NLP pipeline that classifies news headlines into three categories — **Left**, **Center**, or **Right** — using semantic sentence embeddings and a LinearSVC classifier.

The system encodes headlines into dense vector representations using **SBERT (all-MiniLM-L6-v2)** and classifies them using a **LinearSVC** trained on a balanced 3-class dataset of 300 labeled news headlines.

---

## ✨ Features

- 🔤 **Headline Input** — Type or paste any news headline for instant analysis
- 🧬 **SBERT Embeddings** — Semantic sentence-level representations via `all-MiniLM-L6-v2`
- 🎯 **3-Class Classification** — Left / Center / Right bias prediction
- 📊 **Confidence Score** — Model confidence displayed as High / Moderate / Low
- 🎨 **Styled UI** — Color-coded results with gradient title and clean layout
- 🧹 **Text Preprocessing** — URL removal, lowercasing, special character cleaning

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.8+ |
| Embeddings | SentenceTransformer (`all-MiniLM-L6-v2`) |
| Classifier | LinearSVC (scikit-learn) |
| Hyperparameter Tuning | GridSearchCV (3-fold CV, F1-weighted) |
| Frontend | Streamlit |
| Model Serialization | Joblib |

---

## 📁 Project Structure

```
Media_Bias_Detection/
│
├── app.py                  # Streamlit UI + prediction pipeline
├── final.ipynb             # Model training, evaluation & experimentation
├── models/
│   └── bias_model.pkl      # Trained LinearSVC model
└── data/
    └── media_bias_dataset.csv   # 300 labeled headlines (Left/Center/Right)
```

---

## ⚙️ How It Works

```
News Headline (raw text)
        ↓
Text Cleaning (lowercase, remove URLs & special chars)
        ↓
SBERT Encoding → 384-dim embedding vector
        ↓
LinearSVC Classifier (trained, C=1, class_weight=balanced)
        ↓
Prediction: Left / Center / Right
        ↓
Confidence score via decision_function
```

---

## 🔬 Model Training & Experiments

Three approaches were explored in `final.ipynb`:

| Model | Approach | Accuracy |
|-------|----------|----------|
| Logistic Regression | SBERT embeddings | 33% |
| Logistic Regression + GridSearchCV | SBERT embeddings | 28% |
| **LinearSVC** | **SBERT embeddings** | **35%** ✅ |
| LinearSVC + GridSearchCV | SBERT embeddings | 35% |

**Best model:** LinearSVC with `C=1`, `class_weight='balanced'`

### Dataset
- 300 headlines, perfectly balanced: **100 Left / 100 Center / 100 Right**
- 80/20 train-test split with stratification
- Features: headline text + article text combined

> 📌 **Note on accuracy:** The 35% accuracy (vs 33% random baseline) reflects the inherent difficulty of political bias detection — especially on a small 300-sample dataset. Bias is highly subjective and context-dependent. The project demonstrates the full NLP pipeline and experimentation process; accuracy would improve significantly with a larger, more diverse dataset.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Media_Bias_Detection.git
cd Media_Bias_Detection

# 2. Install dependencies
pip install streamlit scikit-learn sentence-transformers joblib numpy

# 3. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🖥️ Usage

1. Launch the app with `streamlit run app.py`
2. Type a news headline in the text area
   - Example: *"Government announces major tax reform plan"*
   - Example: *"New climate policy faces backlash from industry groups"*
3. Click **"Analyze Bias"**
4. See the predicted bias (Left / Center / Right) and confidence level

---

## 💡 Example Predictions

| Headline | Predicted Bias |
|----------|---------------|
| *"Officials review latest changes related to climate policy"* | Right |
| *"Debate intensifies over labor laws policy direction"* | Center |
| *"New developments emerge in voting laws debate"* | Left |

---

## 🔮 Future Improvements

- 📦 Train on a larger, real-world dataset (AllSides, MBFC labels)
- 🧠 Use semantic similarity instead of keyword-only matching
- 🔗 Fine-tune a BERT/RoBERTa model for higher accuracy
- 🌐 Add source-level bias detection (not just headline-level)
- 📊 Visualize which words drive the bias prediction

---

## 🙋‍♂️ Author

**Malka Naaz**
- GitHub: [@Malka23](https://github.com/Malka23)
- LinkedIn: [LinkedIn](https://www.linkedin.com/in/malka-naaz-870338145)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

⭐ **If you found this project helpful, please give it a star!**
