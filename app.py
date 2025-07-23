import streamlit as st
import joblib
import re
import string

model=joblib.load("model_naive_bayes.pkl")
vectorizer=joblib.load("vectorizer_naive_bayes.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # ✅ Remove URLs
    text = re.sub(r"\d+", "", text)                     # ✅ Remove digits (⚠️ Missing in yours)
    text = text.translate(str.maketrans("", "", string.punctuation))  # ✅ Remove punctuation
    text = text.strip()                                 # ✅ Trim leading/trailing whitespace
    return text

st.set_page_config(page_title="Spam Classifier",page_icon="📩" )


st.title("📩 Spam Message Classifier")

user_input=st.text_area("Enter a message to classify",height=150)

if st.button("Predict"):
    if user_input.strip()=="":
        st.warning("Please enter a message.")
    else:
        cleaned_input=clean_text(user_input)
        vectorizd_input=vectorizer.transform([cleaned_input])
        prediction=model.predict(vectorizd_input)

        label="🚫 Spam" if prediction[0]==1 else "✅ Not Spam"
        st.subheader("Prediction:")
        st.success(label)

st.markdown('Made with ❤️ using Streamlit & Scikit-learn</div>', unsafe_allow_html=True)