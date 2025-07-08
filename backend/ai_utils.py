from transformers import pipeline
from googletrans import Translator
import langdetect

# Tranlator
translator = Translator()

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