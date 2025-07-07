from transformers import pipeline

# Translation Pipeline
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")

# Toxicity Classification
classifier = pipeline("text-classification", model="unitary/toxic-bert")

def is_offensive(text):
    """
    Check if the provided text is offensive by translating it to English
    """
    translated = translator(text, max_length=512)[0]['translation_text']
    print(f"Translated text: {translated}")
    result = classifier(translated)

    offensive_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    for item in result:
        label = item['label'].lower()
        score = item['score']
        if label in offensive_labels and score > 0.5:
            return True
    return False