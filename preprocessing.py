import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from typing import List, Optional
from collections import Counter
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Use NLTK's stopwords
STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text: str,
                   use_stemming: bool = False,
                   use_lemmatization: bool = True,
                   min_length: int = 2,
                   remove_numbers: bool = True) -> List[str]:
    """
    Compact text preprocessing using NLTK.
    
    Args:
        text: Input text
        use_stemming: Whether to apply Porter stemming
        use_lemmatization: Whether to apply WordNet lemmatization
        min_length: Minimum token length
        remove_numbers: Whether to remove numbers
    
    Returns:
        List of preprocessed tokens
    """
    # Initialize stemmers/lemmatizers if needed
    stemmer = PorterStemmer() if use_stemming else None
    lemmatizer = WordNetLemmatizer() if use_lemmatization else None

    # Clean text
    text = text.lower()
    text = re.sub(r'http[s]?://\S+|www\.\S+|\S+@\S+|\[.*?\]|\(.*?\)', '', text)  # Remove URLs, emails, and references
    if remove_numbers:
        text = re.sub(r'\b\d+\b', '', text)

    # Tokenize and process
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if (
        len(token) >= min_length and  # Length check
        token.isalnum() and           # Alphanumeric only
        token not in STOPWORDS        # Not a stopword
    )]

    # Apply stemming/lemmatization
    if lemmatizer:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    elif stemmer:
        tokens = [stemmer.stem(token) for token in tokens]

    return tokens

def get_term_frequencies(tokens: List[str]) -> Counter:
    """Calculate term frequencies."""
    return Counter(tokens) 