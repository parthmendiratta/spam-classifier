import string
import re
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Lead data
data=pd.read_csv("spam.csv",encoding="latin-1")[["v1","v2"]] # ["v1","v2"] this is a list of columns names that we require and [[]] double bracket means return a dataframe with those two column not a series 
data.columns=['labels','text'] # changes columns names
# print(data.head(7))

# Clean text
def clean_text(text):
    text=text.lower()
    text=re.sub(r"http\S+|www\S+","",text) # removes links
    text=re.sub(r"\d+","",text) # removes digits
    text=text.translate(str.maketrans("","",string.punctuation))
#               str.maketrans(characters to rplace,characters tobe replaces wih,characters to delete)
    text=text.strip()
    return text

data['cleaned text']=data['text'].apply(clean_text)

# Maping spam :0 and ham :1
data['labels']=data['labels'].map({"ham":0,"spam":1})

# Vectorization
vectorization=TfidfVectorizer()
x=vectorization.fit_transform(data['cleaned text'])
y=data['labels']

# Test-Train split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

# Model Training
# model=LogisticRegression() 96% accuracy ,recall_spam=71
model=LogisticRegression(class_weight="balanced") # 97% , recall_spam=87
model.fit(x_train,y_train)

# Prediction and accuracy
y_predicted=model.predict(x_test)
print("Accuracy: ",accuracy_score(y_test,y_predicted),"\n")

print("Classification report\n")
print(classification_report(y_test,y_predicted),"\n")

print("Confusion Matrix\n")
labels=["ham","spam"]

confusion_matrix=confusion_matrix(y_true=y_test,y_pred=y_predicted)
print(confusion_matrix)

disp=ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

joblib.dump(model,"model.pkl")
joblib.dump(vectorization,"vectorizer.pkl")

print("model and vectorizer saved")