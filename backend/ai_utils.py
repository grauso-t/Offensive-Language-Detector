from transformers import pipeline
from deep_translator import GoogleTranslator
import langdetect
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import html
from openai import OpenAI
import os
from dotenv import load_dotenv
from ctransformers import AutoModelForCausalLM
import torch

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

device = 0 if torch.cuda.is_available() else -1
gpu = 20 if device == 0 else 0

# Initialize OpenAI client
client = OpenAI(api_key=api_key, base_url="https://api.gpt4-all.xyz/v1")

# Initialize Google Translator
translator = GoogleTranslator(source='auto', target='en')

# Load HuggingFace toxicity classifier (based on RoBERTa)
classifier = pipeline("text-classification", model="unitary/toxic-bert", top_k=None, device=device)

"""
# Load pre-trained logistic regression and SVM models
logistic_regression = joblib.load('models/logistic_regression/model.pkl')
vectorizer_logistic_regression = joblib.load('models/logistic_regression/vectorizer.pkl')

linear_svm = joblib.load('models/linear_svm/model.pkl')
vectorizer_linear_svm = joblib.load('models/linear_svm/vectorizer.pkl')
"""

def clean_text(text):
    """
    Clean text by removing mentions, hashtags, URLs, special characters,
    repeated characters, and converting to lowercase.
    """
    text = html.unescape(text)  # Decode HTML entities
    text = re.sub(r'(@\w+|\[USER\])', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s,.!?\'"]+', '', text)  # Remove non-ASCII symbols
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # Normalize repeated characters
    text = re.sub(r'\s+', ' ', text).strip().lower()  # Normalize spaces, lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove remaining special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Final cleanup
    return text


def translate_to_english(text):
    """
    Translate text to English using Google Translate if it's not already in English.
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
        result = translator.translate(text)
        return result
    except Exception:
        return None

def is_offensive_logistic_regression(text):
    """
    Use toxic-BERT to check for offensive tone.
    If toxic, classify the type using a logistic regression model.
    Returns: 'homophobia', 'sexism', 'racism', or 'offensive content'
    """
    translation = translate_to_english(text)
    if translation is None:
        return None

    cleaned = clean_text(translation)
    results = classifier(cleaned)[0]

    offensive_labels = {
        "toxicity", "severe_toxicity", "obscene", "identity_attack",
        "insult", "threat", "sexual_explicit"
    }

    for item in results:
        label = item['label'].lower()
        score = item['score']
        if label in offensive_labels and score > 0.3:
            # Classify using logistic regression
            vectorized = vectorizer_logistic_regression.transform([cleaned])
            prediction = logistic_regression.predict(vectorized)[0]
            class_map = {0: "homophobia", 1: "sexism", 2: "racism"}
            return class_map.get(prediction, "offensive content")

    return "no offensive content"


def is_offensive_svm(text):
    """
    Same as the previous function, but uses SVM for category classification.
    """
    translation = translate_to_english(text)
    if translation is None:
        return None

    cleaned = clean_text(translation)
    results = classifier(cleaned)[0]

    offensive_labels = {
        "toxicity", "severe_toxicity", "obscene", "identity_attack",
        "insult", "threat", "sexual_explicit"
    }

    for item in results:
        label = item['label'].lower()
        score = item['score']
        if label in offensive_labels and score > 0.3:
            # Classify using linear SVM
            vectorized = vectorizer_linear_svm.transform([cleaned])
            prediction = linear_svm.predict(vectorized)[0]
            class_map = {0: "homophobia", 1: "sexism", 2: "racism"}
            return class_map.get(prediction, "offensive content")

    return "no offensive content"


def is_offensive_gpt(text):
    """
    Use OpenAI's GPT model to classify text into:
    'no offensive content', 'sexism', 'racism', or 'homophobia'.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an offensive sentence classifier. Respond only with one of these words: 'no offensive content', 'sexism', 'racism', 'homophobia'."
            },
            {"role": "user", "content": f"Sentence: {text}"}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def is_offensive_llm(text):
    """
    Classifica il testo in: 'no offensive content', 'sexism', 'racism', o 'homophobia'
    usando un modello LLaMA locale.
    """

    # Llama definition
    llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-v0.1-GGUF", model_file="mistral-7b-v0.1.Q4_K_M.gguf", model_type="mistral", gpu_layers=gpu)

    # Prompt"
    prompt = f"""[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <</SYS>>

    You are an offensive sentence classifier. 
    Classify the following sentence into one of the following categories: 
    'no offensive content', 'sexism', 'racism', or 'homophobia'. 

    Respond with just one of these categories.

    Sentence: {text}
    Category:
    [/INST]"""

    output = llm(prompt)
    return output

print(is_offensive_llm("All women are sluts"))