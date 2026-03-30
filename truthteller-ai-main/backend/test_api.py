import requests
import time

time.sleep(2)  # Wait for server to be fully ready
url = 'http://127.0.0.1:5000/predict'

human = "I went to the store today to buy some milk and eggs. The weather was really nice so I decided to walk instead of driving. I bumped into my neighbor and we had a good chat."
ai = "As an AI language model, I do not have personal experiences or feelings. However, I can provide a comprehensive overview based on the data I have been trained on. Multivariable Calculus is the extension of calculus in one variable to calculus with functions of several variables: the differentiation and integration of functions involving multiple variables, rather than just one."

try:
    print("Testing Human text...")
    res1 = requests.post(url, json={"text": human})
    print(res1.json())

    print("Testing AI text...")
    res2 = requests.post(url, json={"text": ai})
    print(res2.json())
except Exception as e:
    print("Error:", e)
