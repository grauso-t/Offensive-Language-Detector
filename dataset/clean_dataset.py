import pandas as pd
import html
import re

import pandas as pd

def undersample_to_minority(df, label_col='category'):
    """
    Function to undersample a DataFrame to balance the classes based on the minimum class size.
    """
    min_count = df[label_col].value_counts().min()
    df_undersampled = df.groupby(label_col).apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)
    return df_undersampled

def clean_text(text):
    """
    Funzione per pulire il testo rimuovendo caratteri inutili
    come virgolette multiple, ma senza eliminare parole offensive.
    """
    # HTML entities
    text = html.unescape(text)
   
    # Rimuovi utenti menzionati
    text = re.sub(r'(@\w+|\[USER\])', '', text)
   
    # Rimuovi hashtag
    text = re.sub(r'#\w+', '', text)
   
    # Rimuovi URL
    text = re.sub(r'http\S+|www\.\S+', '', text)
   
    # Rimuovi caratteri non ASCII
    text = re.sub(r'[^\w\s,.!?\'"]+', '', text)
   
    # Rimuovi ripetizioni di lettere (es. cooool -> cool)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
   
    # Rimuovi spazi multipli
    text = re.sub(r'\s+', ' ', text)
   
    # Rimuovi spazi iniziali/finali
    text = text.strip()
    
    # Converti in minuscolo
    text = text.lower()
   
    # Remove non-alphanumeric characters (except spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
   
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading and trailing spaces
    text = text.strip()
    
    return text

# Load datasets
dataset_homophobia = pd.read_csv('dataset/archive_homophobia/homophobiaDatasetAnonymous.csv')
dataset_sexism = pd.read_csv('dataset/archive_sexism/train (2).csv')
dataset_racism = pd.read_csv('dataset/archive_racism/twitter_racism_parsed_dataset.csv')

# Check the structure of the homophobia dataset
df_homophobia = dataset_homophobia[['text', 'category']]
df_homophobia = df_homophobia[df_homophobia['category'] == 1]
df_homophobia['category'] = 0
df_homophobia['text'] = df_homophobia['text'].apply(clean_text)

# Check the structure of the sexism dataset
df_sexism = dataset_sexism[['text', 'label_sexist']]
df_sexism = df_sexism[df_sexism['label_sexist'] == 'sexist']
df_sexism = df_sexism.rename(columns={'label_sexist': 'category'})
df_sexism['category'] = 1
df_sexism['text'] = df_sexism['text'].apply(clean_text)

# Check the structure of the racism dataset
df_racism = dataset_racism[['Text', 'Annotation']]
df_racism = df_racism[df_racism['Annotation'] == 'racism']
df_racism = df_racism.rename(columns={'Text': 'text', 'Annotation': 'category'})
df_racism['category'] = 2
df_racism['text'] = df_racism['text'].apply(clean_text)

print(df_homophobia.head())
print("Conteggio homophobia:", len(df_homophobia))
print(df_sexism.head())
print("Conteggio sexism:", len(df_sexism))
print(df_racism.head())
print("Conteggio racism:", len(df_racism))

# Concatenate all datasets
df = pd.concat([df_homophobia, df_sexism, df_racism], ignore_index=True)
df_balanced = undersample_to_minority(df, label_col='category')
print(df_balanced['category'].value_counts())
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
df_balanced.to_csv('dataset/cleaned_balanced_dataset.csv', index=False)