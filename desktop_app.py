import re
import sys
import os
import threading
import webview
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
from collections import Counter

# Get the correct path for templates when running as bundled app
if getattr(sys, 'frozen', False):
    # Running as compiled
    bundle_dir = sys._MEIPASS
else:
    # Running in development
    bundle_dir = os.path.dirname(os.path.abspath(__file__))

template_folder = os.path.join(bundle_dir, 'templates')
app = Flask(__name__, template_folder=template_folder)

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
    """Extract important/meaningful terms from text"""
    text_lower = text.lower()
    terms = set()
    
    words = re.findall(r'\b[a-zA-Z0-9.+#]+\b', text_lower)
    for word in words:
        if word not in STOP_WORDS and len(word) > 2:
            terms.add(word)
    
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
    
    extracted = {'skills': set(), 'experience': set(), 'education': set(), 'tools': set()}
    
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
    
    tool_patterns = [
        r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\b',
        r'\b(xero|quickbooks|sage|stripe|hubdoc|excel|word|powerpoint|sap|oracle|salesforce|slack|zoom|asana|jira|trello)\b',
    ]
    
    for pattern in tool_patterns:
        matches = re.findall(pattern, jd_text, re.IGNORECASE)
        for match in matches:
            if match.lower() not in STOP_WORDS and len(match) > 2:
                extracted['tools'].add(match.lower())
    
    exp_pattern = r'(\d+)\+?\s*(?:years?|yrs?)'
    extracted['experience'].update(re.findall(exp_pattern, jd_lower))
    
    edu_pattern = r"\b(bachelor'?s?|master'?s?|phd|doctorate|degree|bs|ms|ba|ma|mba|bsc|msc|cpa|acca|cima|aca)\b"
    extracted['education'].update(re.findall(edu_pattern, jd_lower))
    
    return extracted

def calculate_term_overlap(jd_terms, resume_terms):
    """Calculate weighted overlap between JD and resume terms"""
    if not jd_terms:
        return 0
    
    direct_matches = jd_terms.intersection(resume_terms)
    
    expanded_matches = set()
    for jd_term in jd_terms:
        for resume_term in resume_terms:
            for base, synonyms in SYNONYMS.items():
                all_forms = {base} | set(synonyms)
                if jd_term in all_forms and resume_term in all_forms:
                    expanded_matches.add(jd_term)
                    break
    
    total_matches = direct_matches | expanded_matches
    return len(total_matches) / len(jd_terms)

def calculate_relevancy(jd, resume):
    """Calculate relevancy score between JD and resume using multiple factors"""
    jd_clean = clean_text(jd)
    resume_clean = clean_text(resume)
    
    jd_expanded = expand_with_synonyms(jd_clean)
    resume_expanded = expand_with_synonyms(resume_clean)
    
    # 1. TF-IDF Cosine Similarity (30% weight)
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        ngram_range=(1, 3),
        max_features=10000,
        min_df=1
    )
    vectors = vectorizer.fit_transform([jd_expanded, resume_expanded])
    tfidf_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    
    # 2. Important terms overlap (25% weight)
    jd_terms = extract_important_terms(jd_expanded)
    resume_terms = extract_important_terms(resume_expanded)
    terms_overlap_score = calculate_term_overlap(jd_terms, resume_terms)
    
    # 3. Skills and requirements matching (25% weight)
    jd_requirements = extract_skills_and_requirements(jd)
    resume_text_lower = resume.lower()
    resume_expanded_lower = resume_expanded.lower()
    
    all_jd_items = jd_requirements['skills'] | jd_requirements['tools']
    skills_found = 0
    total_skills = len(all_jd_items)
    
    if total_skills > 0:
        for skill in all_jd_items:
            skill_lower = skill.lower()
            if skill_lower in resume_text_lower or skill_lower in resume_expanded_lower:
                skills_found += 1
            else:
                found = False
                for base, synonyms in SYNONYMS.items():
                    if skill_lower == base or skill_lower in [s.lower() for s in synonyms]:
                        all_forms = [base] + [s.lower() for s in synonyms]
                        if any(form in resume_text_lower or form in resume_expanded_lower for form in all_forms):
                            skills_found += 1
                            found = True
                            break
                if not found:
                    if len(skill_lower) > 3 and any(skill_lower in word for word in resume_text_lower.split()):
                        skills_found += 0.5
        skills_score = skills_found / total_skills
    else:
        skills_score = terms_overlap_score
    
    # 4. Direct keyword presence (20% weight)
    jd_words = set(w for w in jd_clean.split() if w not in STOP_WORDS and len(w) > 3)
    resume_words = set(w for w in resume_clean.split() if len(w) > 2)
    resume_expanded_words = set(resume_expanded_lower.split())
    
    if jd_words:
        direct_matches = sum(1 for w in jd_words if w in resume_words or w in resume_expanded_words)
        keyword_score = direct_matches / len(jd_words)
    else:
        keyword_score = 0
    
    # Combined weighted score
    raw_score = (
        (tfidf_score * 0.30) +
        (terms_overlap_score * 0.25) +
        (skills_score * 0.25) +
        (keyword_score * 0.20)
    )
    
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
        
        relevancy = calculate_relevancy(job_description, resume_text)
        
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
            'color': color
        })
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def start_flask():
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

if __name__ == '__main__':
    # Start Flask in a background thread
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    
    # Create native window
    webview.create_window(
        'Resume Relevancy Analyzer',
        'http://127.0.0.1:5000',
        width=1200,
        height=800,
        resizable=True,
        min_size=(800, 600)
    )
    webview.start()
