# 📩 Spam Message Classifier – Streamlit + Scikit-learn

An AI-powered app that detects spam messages using **Logistic Regression** or **Naive Bayes**, trained on SMS spam data. It features a sleek **Streamlit interface** where you can type a message and get instant feedback – 🚫 Spam or ✅ Not Spam!

---

## 🚀 Features

- ✅ Classifies messages as spam or not spam in real-time  
- 🧠 Trained on SMS spam dataset using TF-IDF vectorization  
- ⚖️ Handles class imbalance using **SMOTE** or `class_weight="balanced"`  
- 💬 Clean Streamlit UI with emoji-based predictions  
- 💾 Model + vectorizer saved via Joblib for easy deployment  

---

## 🧠 Model Highlights

- Accuracy: ~96–97%  
- Models supported: Logistic Regression / Naive Bayes  
- Class imbalance handled effectively  
- Text cleaning: lowercasing, URL/digit/punctuation removal  

---

## 🗂️ Project Structure

```
sms-spam-classifier/
├── app.py                         # Streamlit app
├── model.pkl                      # Trained model (Logistic Regression )
├── model_naive_bayes.pkl          # Trained model (Naive bayes)
├── vectorizer.pkl                 # Saved TF-IDF vectorizer
├── naive_bayes_classifier.py      # Naive Bayes training script
├── logistic_regression_classifier.py # Logistic Regression training script
├── requirements.txt               # All dependencies
├── README.md                      # You're reading it!
```

---

## ▶️ How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/spam-classifier.git
cd spam-classifier
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

Then visit: [http://localhost:8501](http://localhost:8501) in your browser 🚀

---

## 🌐 Live Demo

👉 [Click here to try the live app](https://your-streamlit-app-link.com)  

---

## 📸 Screenshot

> _(Add a screenshot of your app UI here for better visibility)_

---

## 🌟 Like This Project?

Star ⭐ the repo, share it on LinkedIn, and feel free to fork it!

---

## 🙌 Author

Made with ❤️ by [Parth Mendiratta](https://github.com/parthmendiratta)
