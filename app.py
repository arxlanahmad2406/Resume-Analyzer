import re
import json
import os
import io
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
from openai import OpenAI
from dotenv import load_dotenv
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize API clients
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Common word variations/synonyms (generic, not role-specific)
SYNONYMS = {
    # Verbs
    'develop': ['developing', 'developed', 'development', 'build', 'building', 'built', 'create', 'creating', 'created'],
    'manage': ['managing', 'managed', 'management', 'lead', 'leading', 'led', 'leadership', 'coordinate', 'coordinating', 'handle', 'handling', 'handled'],
    'design': ['designing', 'designed', 'architect', 'architecting', 'architected'],
    'implement': ['implementing', 'implemented', 'implementation', 'deploy', 'deploying', 'deployed'],
    'analyze': ['analyzing', 'analyzed', 'analysis', 'analyse', 'analysing', 'analysed', 'review', 'reviewing', 'reviewed'],
    'optimize': ['optimizing', 'optimized', 'optimization', 'improve', 'improving', 'improved', 'enhance', 'enhancing'],
    'test': ['testing', 'tested', 'qa', 'quality assurance'],
    'maintain': ['maintaining', 'maintained', 'maintenance', 'support', 'supporting', 'supported'],
    'process': ['processing', 'processed', 'handle', 'handling', 'handled'],
    'prepare': ['preparing', 'prepared', 'preparation'],
    'report': ['reporting', 'reported', 'reports'],
    'assist': ['assisting', 'assisted', 'assistance', 'help', 'helping', 'support', 'supporting'],
    
    # Tech terms
    'frontend': ['front-end', 'front end', 'ui', 'user interface', 'client-side'],
    'backend': ['back-end', 'back end', 'server-side', 'api'],
    'fullstack': ['full-stack', 'full stack'],
    'database': ['db', 'databases', 'data store', 'datastore'],
    'cloud': ['aws', 'azure', 'gcp', 'google cloud', 'amazon web services', 'cloud computing'],
    'devops': ['ci/cd', 'cicd', 'continuous integration', 'continuous deployment'],
    'ml': ['machine learning', 'ai', 'artificial intelligence', 'deep learning'],
    
    # Finance/Accounting terms
    'finance': ['financial', 'finances', 'fiscal'],
    'account': ['accounts', 'accounting', 'accountant', 'accountancy'],
    'invoice': ['invoices', 'invoicing', 'billing', 'bill', 'bills'],
    'payment': ['payments', 'pay', 'paying', 'payroll', 'payable', 'receivable'],
    'expense': ['expenses', 'expenditure', 'expenditures', 'costs', 'cost'],
    'budget': ['budgets', 'budgeting', 'budgeted'],
    'audit': ['audits', 'auditing', 'audited', 'auditor'],
    'tax': ['taxes', 'taxation', 'taxable'],
    'reconcile': ['reconciliation', 'reconciling', 'reconciled'],
    'ledger': ['ledgers', 'general ledger', 'gl'],
    'bookkeeping': ['bookkeeper', 'books'],
    
    # Experience
    'experience': ['experienced', 'expertise', 'proficient', 'proficiency', 'skilled', 'skills', 'background'],
    'senior': ['sr', 'lead', 'principal', 'staff'],
    'junior': ['jr', 'entry level', 'entry-level', 'associate'],
    
    # Education
    'bachelor': ['bachelors', "bachelor's", 'bs', 'ba', 'bsc', 'undergraduate', 'degree'],
    'master': ['masters', "master's", 'ms', 'ma', 'msc', 'graduate'],
    'phd': ['doctorate', 'doctoral', 'ph.d'],
    
    # Soft skills
    'communicate': ['communication', 'communicating', 'interpersonal', 'correspondence'],
    'collaborate': ['collaboration', 'collaborating', 'teamwork', 'team player', 'cooperative'],
    'problem-solving': ['problem solving', 'analytical', 'troubleshoot', 'troubleshooting'],
    'organize': ['organized', 'organisation', 'organization', 'organisational', 'organizational'],
    'detail': ['details', 'detailed', 'detail-oriented', 'meticulous', 'accuracy', 'accurate'],
}

# Stop words to filter out common non-meaningful words
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'shall', 'can', 'need', 'dare', 'ought', 'used', 'this', 'that', 'these', 'those',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'whom',
    'their', 'our', 'your', 'its', 'his', 'her', 'my', 'me', 'him', 'us', 'them',
    'also', 'just', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'such',
    'no', 'nor', 'not', 'other', 'some', 'any', 'each', 'every', 'all', 'both',
    'few', 'more', 'most', 'several', 'many', 'much', 'either', 'neither', 'one',
    'two', 'three', 'first', 'second', 'new', 'old', 'high', 'low', 'well', 'able',
    'about', 'above', 'after', 'again', 'against', 'along', 'already', 'although',
    'always', 'among', 'another', 'any', 'anyone', 'anything', 'anywhere', 'around',
    'because', 'before', 'being', 'below', 'between', 'beyond', 'during', 'etc',
    'however', 'including', 'into', 'over', 'under', 'until', 'upon', 'within', 'without',
}

def clean_text(text):
    """Clean and normalize text"""
    text = text.lower()
    # Keep alphanumeric, spaces, and some special chars used in tech (., +, #, /)
    text = re.sub(r"[^a-zA-Z0-9.+#/ ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def expand_with_synonyms(text):
    """Expand text with synonyms for better matching"""
    words = text.lower().split()
    expanded_words = set(words)
    
    for word in words:
        for base, synonyms in SYNONYMS.items():
            if word == base or word in synonyms:
                expanded_words.add(base)
                expanded_words.update(synonyms)
    
    return ' '.join(expanded_words)

def extract_important_terms(text):
    """Extract important/meaningful terms from text (skills, technologies, requirements)"""
    text_lower = text.lower()
    terms = set()
    
    # Extract individual words (filter stop words and short words)
    words = re.findall(r'\b[a-zA-Z0-9.+#]+\b', text_lower)
    for word in words:
        if word not in STOP_WORDS and len(word) > 2:
            terms.add(word)
    
    # Extract bigrams and trigrams for compound terms
    words_list = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    for i in range(len(words_list) - 1):
        bigram = f"{words_list[i]} {words_list[i+1]}"
        terms.add(bigram)
    
    for i in range(len(words_list) - 2):
        trigram = f"{words_list[i]} {words_list[i+1]} {words_list[i+2]}"
        terms.add(trigram)
    
    return terms

def extract_skills_and_requirements(jd_text):
    """Dynamically extract skills and requirements from JD"""
    jd_lower = jd_text.lower()
    
    extracted = {
        'skills': set(),
        'experience': set(),
        'education': set(),
        'tools': set(),
    }
    
    # Common patterns for skills sections
    skill_patterns = [
        r'(?:skills?|requirements?|qualifications?|must have|required|proficien\w+|looking for|you\'ll be)[:\s-]+([^.\n]+)',
        r'(?:experience (?:with|in)|knowledge of|expertise in|proficient in|familiar(?:ity)? with)[:\s]*([^.\n]+)',
        r'(?:responsible for|working on|you will)[:\s]*([^.\n]+)',
    ]
    
    for pattern in skill_patterns:
        matches = re.findall(pattern, jd_lower, re.IGNORECASE)
        for match in matches:
            words = re.findall(r'\b[a-zA-Z0-9.+#]+\b', match)
            for word in words:
                if word not in STOP_WORDS and len(word) > 2:
                    extracted['skills'].add(word)
    
    # Extract specific tools/software mentioned (capitalized words or known patterns)
    tool_patterns = [
        r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\b',  # Capitalized words (likely tools/products)
        r'\b(xero|quickbooks|sage|stripe|hubdoc|excel|word|powerpoint|sap|oracle|salesforce|slack|zoom|asana|jira|trello)\b',
    ]
    
    for pattern in tool_patterns:
        matches = re.findall(pattern, jd_text, re.IGNORECASE)
        for match in matches:
            if match.lower() not in STOP_WORDS and len(match) > 2:
                extracted['tools'].add(match.lower())
    
    # Extract years of experience
    exp_pattern = r'(\d+)\+?\s*(?:years?|yrs?)'
    exp_matches = re.findall(exp_pattern, jd_lower)
    extracted['experience'].update(exp_matches)
    
    # Extract education requirements
    edu_pattern = r"\b(bachelor'?s?|master'?s?|phd|doctorate|degree|bs|ms|ba|ma|mba|bsc|msc|cpa|acca|cima|aca)\b"
    edu_matches = re.findall(edu_pattern, jd_lower)
    extracted['education'].update(edu_matches)
    
    return extracted

def calculate_term_overlap(jd_terms, resume_terms):
    """Calculate weighted overlap between JD and resume terms"""
    if not jd_terms:
        return 0
    
    # Direct matches
    direct_matches = jd_terms.intersection(resume_terms)
    
    # Synonym-expanded matches
    expanded_matches = set()
    for jd_term in jd_terms:
        for resume_term in resume_terms:
            # Check if terms are synonyms
            for base, synonyms in SYNONYMS.items():
                all_forms = {base} | set(synonyms)
                if jd_term in all_forms and resume_term in all_forms:
                    expanded_matches.add(jd_term)
                    break
    
    total_matches = direct_matches | expanded_matches
    return len(total_matches) / len(jd_terms)

def extract_key_nouns(text):
    """Extract key nouns and noun phrases that are likely important terms"""
    text_lower = text.lower()
    key_terms = set()
    
    # Extract words that appear after key indicators
    indicator_patterns = [
        r'(?:experience in|with|using)\s+([a-zA-Z0-9]+)',
        r'(?:knowledge of|expertise in)\s+([a-zA-Z0-9]+)',
        r'(?:proficient in|skilled in)\s+([a-zA-Z0-9]+)',
        r'(?:familiar with)\s+([a-zA-Z0-9]+)',
    ]
    
    for pattern in indicator_patterns:
        matches = re.findall(pattern, text_lower)
        key_terms.update(matches)
    
    return key_terms

def get_embedding(text):
    """Get OpenAI embedding for text"""
    max_chars = 25000
    if len(text) > max_chars:
        text = text[:max_chars]
    
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    
    # Log token usage
    if hasattr(response, 'usage'):
        logger.info(f"[OpenAI Embedding] Input tokens: {response.usage.prompt_tokens}, Total tokens: {response.usage.total_tokens}")
    
    return response.data[0].embedding

def cosine_sim(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_analysis_prompt(jd, resume):
    """Generate the analysis prompt for LLM"""
    return f"""You are an expert HR recruiter and resume analyst. Analyze how well this resume matches the job description.

JOB DESCRIPTION:
{jd[:4000]}

RESUME:
{resume[:4000]}

Provide a JSON response with:
1. "score": A relevancy score from 0-100 (be realistic - 70+ means strong match, 50-70 moderate, below 50 weak)
2. "matched_skills": List of skills/requirements from JD found in resume
3. "missing_skills": List of important skills/requirements from JD NOT found in resume  
4. "experience_match": How well the experience level matches ("strong", "moderate", "weak")
5. "summary": One sentence summary of the match

Respond ONLY with valid JSON, no markdown or extra text."""

def analyze_with_openai(jd, resume):
    """Use OpenAI GPT to analyze resume-JD match"""
    prompt = get_analysis_prompt(jd, resume)
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    
    # Log token usage
    usage = response.usage
    logger.info(f"[OpenAI GPT-4o-mini] Input tokens: {usage.prompt_tokens}, Output tokens: {usage.completion_tokens}, Total: {usage.total_tokens}")
    
    return json.loads(response.choices[0].message.content)

def analyze_with_llm(jd, resume):
    """Analyze with OpenAI"""
    try:
        logger.info("Attempting analysis with OpenAI...")
        result = analyze_with_openai(jd, resume)
        logger.info("OpenAI analysis successful")
        return result, "openai"
    except Exception as e:
        logger.error(f"OpenAI failed: {e}")
        raise Exception(f"OpenAI API failed: {e}")

def calculate_relevancy(jd, resume):
    """Calculate relevancy score using embeddings + LLM analysis"""
    try:
        # Method 1: Semantic similarity using OpenAI embeddings (40% weight)
        try:
            jd_embedding = get_embedding(jd)
            resume_embedding = get_embedding(resume)
            embedding_similarity = cosine_sim(jd_embedding, resume_embedding)
            logger.info(f"Embedding similarity: {embedding_similarity:.4f}")
        except Exception as e:
            logger.warning(f"Embedding failed, using fallback: {e}")
            embedding_similarity = 0.5  # Neutral fallback
        
        # Method 2: LLM analysis for detailed matching (60% weight)
        llm_analysis, provider = analyze_with_llm(jd, resume)
        llm_score = llm_analysis.get('score', 50) / 100
        logger.info(f"LLM score from {provider}: {llm_score * 100:.1f}%")
        
        # Combined score
        final_score = (embedding_similarity * 0.40 + llm_score * 0.60) * 100
        
        # Ensure bounds
        final_score = max(0, min(95, final_score))
        
        logger.info(f"Final relevancy score: {final_score:.2f}%")
        
        # Return score along with analysis details
        return {
            'score': round(final_score, 2),
            'matched_skills': llm_analysis.get('matched_skills', []),
            'missing_skills': llm_analysis.get('missing_skills', []),
            'experience_match': llm_analysis.get('experience_match', 'unknown'),
            'summary': llm_analysis.get('summary', ''),
            'provider': provider
        }
        
    except Exception as e:
        logger.error(f"All LLM APIs failed: {e}")
        # Fallback to traditional method
        logger.info("Using traditional TF-IDF fallback method")
        fallback_score = calculate_relevancy_fallback(jd, resume)
        return {
            'score': fallback_score,
            'matched_skills': [],
            'missing_skills': [],
            'experience_match': 'unknown',
            'summary': 'Analysis performed using traditional matching (API unavailable)',
            'provider': 'fallback'
        }

def calculate_relevancy_fallback(jd, resume):
    """Fallback relevancy calculation without OpenAI"""
    jd_clean = clean_text(jd)
    resume_clean = clean_text(resume)
    
    jd_expanded = expand_with_synonyms(jd_clean)
    resume_expanded = expand_with_synonyms(resume_clean)
    
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        ngram_range=(1, 3),
        max_features=10000,
        min_df=1
    )
    vectors = vectorizer.fit_transform([jd_expanded, resume_expanded])
    tfidf_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    
    jd_terms = extract_important_terms(jd_expanded)
    resume_terms = extract_important_terms(resume_expanded)
    terms_overlap_score = calculate_term_overlap(jd_terms, resume_terms)
    
    jd_requirements = extract_skills_and_requirements(jd)
    resume_text_lower = resume.lower()
    
    all_jd_items = jd_requirements['skills'] | jd_requirements['tools']
    skills_found = 0
    total_skills = len(all_jd_items)
    
    if total_skills > 0:
        for skill in all_jd_items:
            if skill.lower() in resume_text_lower:
                skills_found += 1
        skills_score = skills_found / total_skills
    else:
        skills_score = terms_overlap_score
    
    raw_score = (tfidf_score * 0.4) + (terms_overlap_score * 0.3) + (skills_score * 0.3)
    final_score = (raw_score ** 0.5) * 100
    final_score = max(0, min(95, final_score))
    
    return round(final_score, 2)

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_file(file):
    filename = file.filename.lower()
    if filename.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif filename.endswith('.docx'):
        return extract_text_from_docx(file)
    elif filename.endswith('.txt'):
        return file.read().decode('utf-8')
    else:
        raise ValueError("Unsupported file format. Please upload PDF, DOCX, or TXT.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        job_description = request.form.get('job_description', '')
        
        if not job_description.strip():
            return jsonify({'error': 'Please provide a job description'}), 400
        
        if 'resume' not in request.files:
            return jsonify({'error': 'Please upload a resume file'}), 400
        
        resume_file = request.files['resume']
        if resume_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        resume_text = extract_text_from_file(resume_file)
        
        if not resume_text.strip():
            return jsonify({'error': 'Could not extract text from resume'}), 400
        
        result = calculate_relevancy(job_description, resume_text)
        relevancy = result['score']
        
        if relevancy > 70:
            category = "Strong Match"
            color = "#22c55e"
        elif relevancy > 40:
            category = "Moderate Match"
            color = "#f59e0b"
        else:
            category = "Weak Match"
            color = "#ef4444"
        
        return jsonify({
            'relevancy': relevancy,
            'category': category,
            'color': color,
            'matched_skills': result['matched_skills'],
            'missing_skills': result['missing_skills'],
            'experience_match': result['experience_match'],
            'summary': result['summary'],
            'provider': result['provider']
        })
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
