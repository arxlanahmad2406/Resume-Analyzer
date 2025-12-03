import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# -----------------------------
# SIMPLE RELEVANCY EVALUATION TOOL
# Add Job Description text and Resume text in the variables below.
# Run the script to get a relevancy percentage.
# -----------------------------
# SAMPLE INPUTS (Replace these when using the tool)
job_description = """

"""
resume_text = """

"""
# -----------------------------
# CLEANING FUNCTION
# -----------------------------
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text
# -----------------------------
# RELEVANCY SCORE FUNCTION
# -----------------------------
def calculate_relevancy(jd, resume):
    jd_clean = clean_text(jd)
    resume_clean = clean_text(resume)
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([jd_clean, resume_clean])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    relevancy_percent = round(score * 100, 2)
    return relevancy_percent
# -----------------------------
# RUN TOOL
# -----------------------------
if __name__ == "__main__":
    relevancy = calculate_relevancy(job_description, resume_text)
    result = {
        "JD vs Resume Relevancy (%)": relevancy,
        "Match Category": "Strong Match" if relevancy > 70 else "Moderate Match" if relevancy > 40 else "Weak Match"
    }
    print(json.dumps(result, indent=4))