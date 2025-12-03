# Resume Relevancy Analyzer

A web-based tool that analyzes how well a resume matches a job description using TF-IDF cosine similarity.

## Features

- Paste job descriptions in the left panel
- Upload resumes (PDF, DOCX, or TXT) in the right panel
- Get instant relevancy score with match category (Strong/Moderate/Weak)

## Requirements

- Python 3.x
- pipenv

## Installation

```bash
# Install pipenv if you don't have it
pip install pipenv

# Install dependencies
pipenv install
```

## Usage

```bash
# Run the web app
pipenv run python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Match Categories

| Score | Category |
|-------|----------|
| > 70% | Strong Match |
| 40-70% | Moderate Match |
| < 40% | Weak Match |
