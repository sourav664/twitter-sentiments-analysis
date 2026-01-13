import contractions
import re
import emoji
from bs4 import BeautifulSoup
import spacy
import json

with open('./data/vocab.json', 'r') as f:
    vocab = json.load(f)


nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])


def preprocess_text(text):
    # Converts emoji to text
    text = emoji.demojize(text)
    text = BeautifulSoup(text, "html.parser").get_text() # Remove HTML tags
    
    text = contractions.fix(text)  # Expand contractions
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # Reduce repeated characters
    text=re.sub(r"[@#^<>|(){}[\]\~*+=]", " ", text) # Remove less important special characters
    text=re.sub(r"\s+", " ", text)
    text = text.strip()  # Remove leading and trailing spaces
    return text


# lemmatization
def lem_single(text):
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc)


# tokenization
def tokenize(text):
    tokens = []
    for token in nlp(text.lower()):
        tokens.append(token.text)
    return tokens


# replace words with numbers
def replace_word_with_numbers(text):
    tokens = []
    for token in text:
        if token in vocab:
            tokens.append(vocab[token])
        else:
            tokens.append(vocab['<unk>'])
    return tokens


# padding 
def padding(text, max_len=200):
    return text[:max_len] + [0] * (max_len - len(text))



def preprocesser(text):
    text = preprocess_text(text)
    text = lem_single(text)
    tokens = tokenize(text)
    tokens = replace_word_with_numbers(tokens)
    tokens = padding(tokens)
    return tokens


preprocesser("I be come to the border and I will kill you all ,")