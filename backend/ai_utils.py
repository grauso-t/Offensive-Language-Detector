from transformers import pipeline
from googletrans import Translator
import langdetect
import joblib

# Tranlator
translator = Translator()

# Model import
model = joblib.load('dataset/logistic_regression_model.pkl')

# Toxicity Classification
classifier = pipeline("text-classification", model="unitary/toxic-bert")

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

def is_offensive(text):
    """
    Check if the provided text is offensive by translating it to English
    """
    translation = translate_to_english(text)

    if text is not None:
        result = classifier(translation)

        offensive_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        for item in result:
            label = item['label'].lower()
            score = item['score']
            if label in offensive_labels and score > 0.5:
                return True
        return False
    else:
        return None
    
def classify_text(text):
    """
    Classify the provided text using the toxicity classifier.
    """
    translation = translate_to_english(text)

    if translation is None:
        return None
    
    if model.predict([translation])[0] == 0:
        return "homophobia"
    elif model.predict([translation])[0] == 1:
        return "sexism"
    elif model.predict([translation])[0] == 2:
        return "recism"
    else:
        return "not classified"