import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

data = pd.read_csv("corpus/mental_health.csv")

text = data["text"]
label = data["label"]

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(text)

x_train, x_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=42)

model = XGBClassifier(n_estimators=100, use_label_encoder=False)

model.fit(x_train, y_train)

joblib.dump(model, 'pkl/mental_health_xgb_model.pkl')
joblib.dump(vectorizer, 'pkl/tfidf_vectorizer.pkl')

print("Model and vectorizer saved successfully.")

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")