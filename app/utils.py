import os
import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from .features.TFIDFSummarization import run_tfidf_summarization
from .features.WordFrequancySummarizatiom import run_wf_summarization
from .features.TextRankAlgorithm import TextRank4Sentences
import re
import PyPDF2
import docx

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize Spacy and NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')

nlp.max_length = 30000000

# Define the raw and processed data directories
RAW_DATA_DIR = 'datasets/raw/'
PROCESSED_DATA_DIR = 'datasets/processed/'


def preprocess_text(text):
    """ Preprocess text by tokenizing, removing stopwords, and lemmatizing """
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(
        word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)


def process_csv(file_name):
    """ Process a CSV file by reading, cleaning, and saving it """
    input_path = os.path.join(RAW_DATA_DIR, file_name)
    output_path = os.path.join(PROCESSED_DATA_DIR, f'processed_{file_name}')

    # Read the raw CSV file
    df = pd.read_csv(input_path)

    # Preprocess the text columns (assuming the text is in a column named 'text')
    df['processed_text'] = df['text'].apply(preprocess_text)

    # Save the processed data to a new CSV file
    df.to_csv(output_path, index=False)


def process_txt(file_name):
    """ Process a TXT file by reading, cleaning, and saving it """
    input_path = os.path.join(RAW_DATA_DIR, file_name)
    output_path = os.path.join(PROCESSED_DATA_DIR, f'processed_{file_name}')

    # Read the raw TXT file
    with open(input_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Preprocess the text
    processed_text = preprocess_text(text)

    # Save the processed text to a new TXT file
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(processed_text)


def process_html(file_name):
    """ Process an HTML file by reading, extracting text, cleaning, and saving it """
    input_path = os.path.join(RAW_DATA_DIR, file_name)
    output_path = os.path.join(PROCESSED_DATA_DIR, f'processed_{file_name}')

    # Read the raw HTML file
    with open(input_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Extract text from HTML
    text = extract_text_from_html(html_content)

    # Preprocess the text
    processed_text = preprocess_text(text)

    # Save the processed text to a new TXT file
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(processed_text)


def process_pdf(file_name):
    """ Process a PDF file by reading, extracting text, cleaning, and saving it """
    input_path = os.path.join(RAW_DATA_DIR, file_name)
    output_path = os.path.join(PROCESSED_DATA_DIR, f'processed_{file_name}')

    # Read the raw PDF file
    text = extract_text_from_pdf(input_path)

    # Preprocess the text
    processed_text = preprocess_text(text)

    # Save the processed text to a new TXT file
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(processed_text)


def extract_text_from_html(html_content):
    """ Extract text from HTML content """
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator=' ')
    return text


def extract_text_from_pdf(pdf_path):
    """ Extract text from a PDF file """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def extract_top_sentences(text, nbr):
    tr4sh = TextRank4Sentences()
    tr4sh.analyze(text)
    sentences = tr4sh.get_top_sentences(nbr)
    return sentences


def tfidf_summary(text):
    """ Provide an tfidf summary of the text """
    return run_tfidf_summarization(text)


def wf_summary(text):
    """ Provide an wf summary of the text """
    return run_wf_summarization(text)


def extract_formulas(text):
    """ Extract formulas from the text """
    formulas = re.findall(
        r'\b([A-Za-z0-9]+ *[=+*/-] *[A-Za-z0-9]+(\s*[=+*/-]\s*[A-Za-z0-9]+)*)\b', text)
    return [formula[0] for formula in formulas]


def search_documents(query):
    """ Perform a search for documents related to the query """
    input_dir = PROCESSED_DATA_DIR
    results = []

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.txt') or file_name.endswith('.csv'):
            file_path = os.path.join(input_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                if query.lower() in content.lower():
                    results.append(file_name)

    return results


def read_file(file_path: str) -> str:
    """
    Reads the content of a file and returns it as a string, handling text files, PDFs, and Word documents (DOC and DOCX).

    Args:
        file_path (str): The name of the file.

    Returns:
        str: The content of the file, or an empty string if the file cannot be read or is not supported.
    """
    if file_path.endswith(".txt"):
        try:
            with open(file_path, "r", encoding='UTF-8') as f:  # Try for text files first
                content = f.read()
                return content
        except Exception:
            pass  # Ignore file not found for text handling, check for other formats
    # Handle PDFs using PyPDF2
    elif file_path.endswith('.pdf'):
        try:
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                pages = len(pdf_reader.pages)
                content = ""
                for page_num in range(pages):
                    page = pdf_reader.pages[page_num]
                    content += page.extract_text()
                return content
        except Exception as e:
            print(e)

    # Handle Word documents (DOCX and DOCX) using python-docx
    elif file_path.endswith((".doc", ".docx")):
        try:
            doc = docx.Document(file_path)
            full_text = []
            for paragraph in doc.paragraphs:
                full_text.append(paragraph.text)
            content = "\n".join(full_text)
            return content

        except Exception:
            # Ignore docx errors and return empty string
            pass
    return False
