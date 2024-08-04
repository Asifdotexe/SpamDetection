import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def nlp_cleaning(text):
    """
    This function performs various text cleaning operations on the input text.

    Parameters:
    text (str): The input text to be cleaned.

    Returns:
    str: The cleaned text after applying the following operations:
         - Convert the text to lowercase.
         - Remove URLs from the text.
         - Remove non-ASCII characters.
         - Remove punctuation marks.
         - Remove numbers.
         - Remove extra whitespaces.
    """
    text = text.lower()
    text = re.sub(r'http[s]?://\S+', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()    
    return text

def lemmatize_with_pos(tokens):
    """
    This function performs lemmatization on a list of tokens using the WordNetLemmatizer
    from the Natural Language Toolkit (nltk). It takes a list of tokens as input and
    returns a list of lemmatized tokens. The lemmatization process involves reducing
    each word to its base or root form, while preserving its grammatical category.

    Parameters:
    tokens (list): A list of tokens to be lemmatized. Each token should be a string.

    Returns:
    list: A list of lemmatized tokens. Each token is a string in its base or root form,
    according to its grammatical category.

    Example:
    >>> tokens = ['walking', 'quickly', 'beautiful', 'dogs', 'jumping']
    >>> lemmatize_with_pos(tokens)
    ['walk', 'quick', 'beauty', 'dog', 'jump']
    """
    pos_tags = pos_tag(tokens)
    lemmatized_tokens = []
    for token, tag in pos_tags:
        if tag.startswith('J'):  # Adjective
            lemmatized_tokens.append(lemmatizer.lemmatize(token, pos='a'))
        elif tag.startswith('V'):  # Verb
            lemmatized_tokens.append(lemmatizer.lemmatize(token, pos='v'))
        elif tag.startswith('N'):  # Noun
            lemmatized_tokens.append(lemmatizer.lemmatize(token, pos='n'))
        elif tag.startswith('R'):  # Adverb
            lemmatized_tokens.append(lemmatizer.lemmatize(token, pos='r'))
        else:
            lemmatized_tokens.append(lemmatizer.lemmatize(token))
    return lemmatized_tokens

def nlp_preprocessing(text, method='lemmatize'):
    """
    This function performs natural language processing (NLP) preprocessing on the input text.

    Parameters:
    text (str): The input text to be preprocessed.
    method (str, optional): The method to be applied for text preprocessing. Default is 'lemmatize'.
    The method can be either 'stem' or 'lemmatize'. If not specified, 'lemmatize' is used by default.

    Returns:
    str: The preprocessed text after applying the following operations:
         - Convert the text to lowercase.
         - Remove URLs from the text.
         - Remove non-ASCII characters.
         - Remove punctuation marks.
         - Remove numbers.
         - Remove extra whitespaces.
         - Tokenize the text into individual words.
         - Remove stop words from the tokenized words.
         - Apply the specified method (either 'stem' or 'lemmatize') to the remaining words.
         - Join the processed words back into a single string separated by spaces.

    Example:
    >>> text = "This is an example text with some URLs: https://www.example.com and numbers: 1234567890."
    >>> nlp_preprocessing(text)
    'this is example text with some urls and numbers'
    """
    text = nlp_cleaning(text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    if method == 'stem':
        tokens = [stemmer.stem(word) for word in tokens]
    elif method == 'lemmatize':
        tokens = lemmatize_with_pos(tokens)
    else:
        raise ValueError("Method must be either 'stem' or 'lemmatize'")

    processed_text = ' '.join(tokens)
    return processed_text