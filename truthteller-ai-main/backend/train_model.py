import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("AuthentiText_X_2026_AI_vs_Human_Detection_1K.csv")

# Keep only required columns
df = df[['content_text', 'author_type']]

# Drop missing values
df.dropna(inplace=True)

# Convert labels
df['author_type'] = df['author_type'].map({'Human': 0, 'AI': 1})

# Split data
X = df['content_text']
y = df['author_type']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF (improved)
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words='english'
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Models to compare
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC()
}

best_model = None
best_accuracy = 0
best_model_name = ""

print("\nModel Comparison Results:\n")

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)

    print(f"{name}: {acc:.4f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

print("\nBest Model:", best_model_name)
print("Best Accuracy:", best_accuracy)

# Save best model
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nBest model saved as model.pkl")