import pandas as pd
import re
import string
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay,classification_report
import matplotlib.pyplot as plt
import joblib

data=pd.read_csv("spam.csv",encoding="latin-1")[["v1","v2"]]
data.columns=["label","text"]

data['label']=data['label'].map({"ham":0,"spam":1})

def clean_text(text):
    text=text.lower()
    text=re.sub(r"http\S+|www\S+","",text)
    text=re.sub(r"\d+","",text)
    text=text.translate(str.maketrans("","",string.punctuation))
    text=text.strip()
    return text

data['cleaned_text']=data['text'].apply(clean_text)


# Vectorization
vectorizer=TfidfVectorizer()
x=vectorizer.fit_transform(data['cleaned_text'])
y=data['label']

# Train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

# SMOTE
smote=SMOTE(random_state=42)
x_train_smote,y_train_smote=smote.fit_resample(x_train,y_train)

# Model Training (Naive Bayes)
model=MultinomialNB()
model.fit(x_train_smote,y_train_smote)

y_predicted=model.predict(x_test)

# Accuracy, classification report and conusion matrix
print("Accurcy Score: ",accuracy_score(y_test,y_predicted))

print("Classiication report:\n",classification_report(y_true=y_test,y_pred=y_predicted))

print("Confusion Matrix:\n",confusion_matrix(y_test,y_predicted))

labels=["ham","spam"]
confu_matix=confusion_matrix(y_true=y_test,y_pred=y_predicted)

disp=ConfusionMatrixDisplay(confusion_matrix=confu_matix,display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

joblib.dump(model,"model_naive_bayes.pkl")
joblib.dump(vectorizer,"vectorizer_naive_bayes.pkl")