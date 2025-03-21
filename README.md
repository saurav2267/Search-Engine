# Information Retrieval System for Cranfield Collection

A Python-based Information Retrieval system implementing multiple ranking models (VSM, BM25, and Language Model) for the Cranfield collection. This system provides text preprocessing, indexing, and evaluation capabilities.

## Features

- **Multiple Ranking Models**:
  - Vector Space Model (VSM) with TF-IDF
  - BM25 (Best Match 25)
  - Language Model with Dirichlet smoothing

- **Advanced Text Preprocessing**:
  - NLTK-based tokenization
  - Stopword removal
  - Lemmatization/Stemming options
  - Special character and number handling
  - URL and email removal

- **Evaluation**:
  - TREC-style evaluation
  - Multiple metrics (MAP, NDCG, P@5)
  - NDCG at various cutoff points (5, 10, 20)
  - Windows WSL support for trec_eval

## Project Structure

```
.
├── main.py              # Main IR system implementation
├── preprocessing.py     # Text preprocessing module
├── parsers.py          # XML parsing for documents and queries
├── indexing.py         # Inverted index construction
├── ranking_models.py   # Implementation of ranking models
├── output.py           # TREC output generation
└── requirements.txt    # Python dependencies
```

## Prerequisites

- Python 3.8 or higher
- Windows with WSL (Windows Subsystem for Linux) for evaluation
- NLTK and required data
- trec_eval executable (provided)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data (automatically handled in code, but can be done manually):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

4. Make trec_eval executable in WSL:
```bash
wsl chmod +x trec_eval
```

## Usage

### Running the IR System

1. Run the main system to process documents and generate rankings:
```bash
python main.py
```

This will:
- Parse the Cranfield documents and queries
- Build the inverted index
- Generate rankings using all three models
- Save results in the Outputs directory

### Evaluating Results

open WSL, then 

The evaluation will show:
- MAP (Mean Average Precision)
- NDCG (Normalized Discounted Cumulative Gain)
- P@5 (Precision at 5)

### Customizing Preprocessing

You can customize text preprocessing in your code:

```python
from preprocessing import preprocess_text

# Basic usage
tokens = preprocess_text(text)

# Advanced usage with options
tokens = preprocess_text(
    text,
    use_stemming=False,
    use_lemmatization=True,
    min_length=3,
    remove_numbers=True
)
```

## Evaluation Metrics

- **MAP**: Mean Average Precision across all queries
- **NDCG**: Overall Normalized Discounted Cumulative Gain
- **P@10**: Precision at 5 documents retrieved

## File Formats

### Input Files
- Documents: XML format (Cranfield collection)
- Queries: XML format
- Relevance judgments: TREC format

### Output Files
- `vsm_run.txt`: VSM model results
- `bm25_run.txt`: BM25 model results
- `lm_run.txt`: Language Model results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Acknowledgments

- Cranfield collection providers
- NLTK team for text processing tools
- TREC for evaluation tools and metrics 
