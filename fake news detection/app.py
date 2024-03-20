from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the data
data_fake = pd.read_csv('fake.csv')
data_true = pd.read_csv('True.csv')

# Combine the data
data_fake['label'] = 0
data_true['label'] = 1
data = pd.concat([data_fake, data_true])

# Train the model
X = data['text']
y = data['label']
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X)
logistic_model = LogisticRegression()
logistic_model.fit(X_tfidf, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        text_tfidf = tfidf_vectorizer.transform([text])
        prediction = logistic_model.predict(text_tfidf)
        if prediction[0] == 0:
            result = "Fake News"
        else:
            result = "Real News"
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
