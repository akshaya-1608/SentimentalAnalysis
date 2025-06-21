import pandas as pd
import random
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

nltk.download('movie_reviews')

documents = [(movie_reviews.raw(fileid), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)
texts, labels = zip(*documents)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42)

vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
print(colored(f"\n✅ Accuracy: {acc * 100:.2f}%\n", "green", attrs=["bold"]))

print(colored("📋 Classification Report:\n", "cyan", attrs=["bold"]))
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

cm = confusion_matrix(y_test, y_pred, labels=["neg", "pos"])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.title("📊 Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

def predict_sentiment(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    if pred == "pos":
        emoji = "😊"
        color = "green"
    else:
        emoji = "😞"
        color = "red"
    print(colored(f"\n💬 Input: {text}", "yellow"))
    print(colored(f"🔎 Predicted Sentiment: {'Positive' if pred == 'pos' else 'Negative'} {emoji}", color, attrs=["bold"]))

while True:
    user_input = input("\nEnter a review (or type 'exit' to stop): ")
    if user_input.lower() == "exit":
        break
    predict_sentiment(user_input)
