import joblib

model = joblib.load('pkl/mental_health_xgb_model.pkl')
vectorizer = joblib.load('pkl/tfidf_vectorizer.pkl')

sentence = input("Give me a sentence:\n")

sentence_v = vectorizer.transform([sentence])
prediction = model.predict(sentence_v)

if prediction == 1:
    print("You are INSANE!!!!!")
else:
    print("You are normal")