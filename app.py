from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

app = Flask(__name__)

# Load dataset
df = pd.read_csv('chatbot_dataset.csv')

# Preprocess the dataset: convert questions to lowercase
df['questions'] = df['questions'].str.lower()

# Initialize vectorizer and fit it on the questions column
vectorizer = TfidfVectorizer()
vectorizer.fit(df['questions'])

@app.route('/')
def home():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    response = get_best_answer(user_message)
    sentiment = analyze_sentiment(user_message)
    return jsonify({'response': response, 'sentiment': sentiment})

@app.route('/counselors')
def counselors():
    counselors_info = [
        {"name": "Dr. John Doe", "contact": "john.doe@example.com", "profile": "Expert in cognitive behavioral therapy."},
        {"name": "Dr. Jane Smith", "contact": "jane.smith@example.com", "profile": "Specializes in anxiety and depression."},
    ]
    return render_template('counselors.html', counselors=counselors_info)


def get_best_answer(user_input):
    user_input = user_input.lower()
    user_input_vec = vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_input_vec, vectorizer.transform(df['questions']))
    best_match_index = cosine_similarities.argmax()
    return df['answers'][best_match_index]

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return {'score': sentiment_score}

if __name__ == '__main__':
    app.run(debug=True)