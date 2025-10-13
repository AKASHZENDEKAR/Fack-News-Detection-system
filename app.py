from flask import Flask, render_template, request
import pickle
from model import SimpleFakeNewsDetector

app = Flask(__name__)

# Load model
with open("fake_news_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['news']
    pred = model.predict(text)
    return render_template('index.html', prediction=pred)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
