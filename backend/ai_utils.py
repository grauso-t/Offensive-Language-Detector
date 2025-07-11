from transformers import pipeline
from googletrans import Translator
import langdetect
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import html

def clean_text(text):
    """
    Function to clean text by removing unnecessary characters,
    such as repeated quotes, mentions, hashtags, and URLs,
    while preserving potentially offensive words.
    """
    # Decode HTML entities (e.g., &amp; → &)
    text = html.unescape(text)
   
    # Remove mentioned users (e.g., @username or [USER])
    text = re.sub(r'(@\w+|\[USER\])', '', text)
   
    # Remove hashtags (e.g., #topic)
    text = re.sub(r'#\w+', '', text)
   
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
   
    # Remove non-ASCII characters (excluding common punctuation)
    text = re.sub(r'[^\w\s,.!?\'"]+', '', text)
   
    # Reduce repeated characters (e.g., cooool → cool)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
   
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
   
    # Remove leading and trailing whitespace
    text = text.strip()
    
    # Convert to lowercase
    text = text.lower()
   
    # Remove all non-alphanumeric characters (except spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
   
    # Normalize spaces again
    text = re.sub(r'\s+', ' ', text)
    
    # Final trim
    text = text.strip()
    
    return text

# Tranlator
translator = Translator()

# Model import
model = joblib.load('./models/logistic_regression/model.pkl')
vectorizer = joblib.load('./models/logistic_regression/vectorizer.pkl')

# Load a toxicity classification pipeline (RoBERTa-based model)
classifier = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)

def translate_to_english(text):
    """
    Translate the provided text to English if it is not already in English.
    """
    if not text.strip():
        return None
    
    try:
        detected_lang = langdetect.detect(text)
    except:
        return None
    
    if detected_lang == 'en':
        return text
    
    try:
        result = translator.translate(text, dest='en')
        return result.text
    except Exception as e:
        return None

def is_offensive_logistic_regression(text):
    """
    Classify a sentence as positive, negative, or neutral.
    If classified as negative, also assign a specific category.
    """
    translation = translate_to_english(text)
    if translation is None:
        return None
    
    cleaned = clean_text(translation)
    
    results = classifier(cleaned)[0]  # List of scores
    
    offensive_labels = {
        "toxicity", "severe_toxicity", "obscene", "identity_attack",
        "insult", "threat", "sexual_explicit"
    }
    
    for item in results:
        label = item['label'].lower()
        score = item['score']
        if label in offensive_labels and score > 0.3:
            final = vectorizer.transform([cleaned])
            prediction = model.predict(final)[0]
            class_map = {0: "homophobia", 1: "sexism", 2: "racism"}
            return class_map.get(prediction, "offensive content")
    
    return "no offensive content"