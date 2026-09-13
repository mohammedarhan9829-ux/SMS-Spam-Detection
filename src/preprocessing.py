import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def init_nltk():
    required = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords')
    ]
    for path, package in required:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass

init_nltk()

stemmer = PorterStemmer()

try:
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    STOP_WORDS = set()

def clean_text(text: str) -> str:
    """
    Cleans and normalizes raw text for spam classification:
    - Lowercasing
    - Regex pattern substitutions for URLs, phone numbers, and currency symbols
    - Tokenization
    - Removal of punctuation and stop words
    - Porter Stemming
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Convert text to lowercase
    text_lower = text.lower()

    # Replace URLs
    text_lower = re.sub(r'https?://\S+|www\.\S+', ' urltoken ', text_lower)
    
    # Replace email addresses
    text_lower = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' emailtoken ', text_lower)

    # Tokenize words
    try:
        tokens = nltk.word_tokenize(text_lower)
    except Exception:
        tokens = re.findall(r'\b\w+\b', text_lower)

    cleaned_tokens = []
    for token in tokens:
        # Keep alphanumeric tokens
        if token.isalnum():
            if token not in STOP_WORDS and token not in string.punctuation:
                stemmed = stemmer.stem(token)
                cleaned_tokens.append(stemmed)

    return " ".join(cleaned_tokens)


def extract_metadata_features(text: str) -> dict:
    """
    Extracts structural text metrics useful for exploratory analysis.
    """
    if not isinstance(text, str):
        text = ""

    num_chars = len(text)
    words = text.split()
    num_words = len(words)
    num_uppercase = sum(1 for c in text if c.isupper())
    num_digits = sum(1 for c in text if c.isdigit())
    num_currency = sum(1 for c in text if c in {'$', '£', '€', '₹'})
    has_url = 1 if re.search(r'https?://|www\.', text, re.IGNORECASE) else 0

    return {
        'num_characters': num_chars,
        'num_words': num_words,
        'num_uppercase': num_uppercase,
        'num_digits': num_digits,
        'num_currency_symbols': num_currency,
        'has_url': has_url
    }
