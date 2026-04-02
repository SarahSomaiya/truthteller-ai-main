import os
import joblib
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def fetch_and_prepare_data(max_samples_per_class=5000):                                            
    print("Downloading Hello-SimpleAI/HC3 dataset via HF API...")
    
    texts = []
    labels = []
    
    human_count = 0
    ai_count = 0
    offset = 0
    length = 100
    
    # We query the HuggingFace datasets-server API directly
    base_url = "https://datasets-server.huggingface.co/rows?dataset=Hello-SimpleAI%2FHC3&config=all&split=train&offset={}&length={}"
    
    try:
        while human_count < max_samples_per_class or ai_count < max_samples_per_class:
            url = base_url.format(offset, length)
            print(f"Fetching rows starting at offset {offset}...")
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            rows = data.get('rows', [])
            if not rows:
                break
            
            for row in rows:
                item = row['row']
                # human_answers is a list of strings
                for ans in item.get('human_answers', []):
                    if human_count < max_samples_per_class and len(ans.split()) > 10:
                        texts.append(ans)
                        labels.append(0) # 0 for Human
                        human_count += 1
                        
                # chatgpt_answers is a list of strings
                for ans in item.get('chatgpt_answers', []):
                    if ai_count < max_samples_per_class and len(ans.split()) > 10:
                        texts.append(ans)
                        labels.append(1) # 1 for AI
                        ai_count += 1
                        
                if human_count >= max_samples_per_class and ai_count >= max_samples_per_class:
                    break
                    
            offset += length
            
    except Exception as e:
        print(f"Error fetching from HF API: {e}")
            
    print(f"Collected {len([l for l in labels if l == 0])} Human texts and {len([l for l in labels if l == 1])} AI texts.")
    return texts, labels

def train_model():
    texts, labels = fetch_and_prepare_data(10000)
    
    if not texts:
        print("Failed to download data.")
        return
        
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    
    print("Training TF-IDF + Logistic Regression pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True)),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/real_model.pkl'
    joblib.dump(pipeline, model_path)
    print(f"Model saved successfully to {model_path}.")

if __name__ == "__main__":
    train_model()
