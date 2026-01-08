# WordNet Synonym Explorer

This project demonstrates a WordNet-based phrase explorer using Weaviate, NLTK, and Streamlit. You can query synonyms and antonyms, filter by similarity and word length, and visualize the results interactively.

## Prerequisites

- Python 3.10+
- Docker & Docker Compose (for Weaviate)
- NVIDIA Docker (optional, if using GPU for text2vec-transformers)
- Virtual environment (venv) for Python dependencies

## Setup Instructions

### 1. Start Weaviate with Docker Compose

Make sure docker-compose.yml is in the current folder, then run:

```bash
docker compose up -d
```

Check that Weaviate is running:

```bash
docker ps
```

You should see Weaviate running on port 8080.

### 2. Set up Python virtual environment

- Linux/Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

- Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install pandas streamlit weaviate-client nltk
```

Download the WordNet corpus for NLTK:

```bash
python -m nltk.downloader wordnet
```

### 4. Index WordNet phrases into Weaviate

Run the indexing script to populate Weaviate:

```bash
python index.py
```

You should see phrases being created in the console. Wait until it finishes indexing ~10,000 phrases.

### 5. Run the Streamlit app

Start the interactive frontend:

```bash
streamlit run streamlit_client.py
```

Open your browser at http://localhost:8501. You will see:

- Query 1: Phrases similar to vocation and values, excluding negative words
- Query 2: Phrases similar to vices and debauchery, excluding positive words
- Interactive section: Synonyms and antonyms with sliders for word length

### 6. Using the app

- Adjust the attractive words and repelling words in the inputs.
- Use the slider to filter words by length.
- Synonyms appear in the left column, antonyms in the right column.
- Results show text and certainty scores from Weaviate.

### 7. Stop Weaviate

```bash
docker compose down
```

## Notes

- Make sure Weaviate is running before starting index.py or Streamlit.
- If using GPU, ensure NVIDIA Docker is installed and text2vec-transformers is selected in docker-compose.yml.
- Streamlit automatically refreshes queries when inputs change.

Enjoy exploring WordNet phrases interactively!
