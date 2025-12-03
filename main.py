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
Summary
I already have the python script but it's not working like I'd like to.
Script needs to be rewrote/fixed (or change the logic)
It's a little bit complex (and long) so please apply only if you are a python expert.
The script will be shared with right candidate.
This task is urgent and needs to be complete today.
Thank you for understanding.
"""
resume_text = """
Summary
Full-Stack Developer with 2+ years of professional experience delivering scalable, high-quality web applications.
Proven expertise in React.js, Vue.js, Next.js, Golang, and Python with strong skills in backend APIs, mi-
croservices, and AI integrations (OpenAI, Anthropic, LangChain). Adept at collaborating with international
teams, building performant, user-focused solutions, and delivering results that align with business goals.
Skills
Frontend: HTML, CSS, JavaScript, React.js, Vue.js, Next.js, Nuxt.js, Tailwind CSS, SASS, Bootstrap, GSAP,
Particle.js, Material UI, Ant Design, Ark UI, Panda CSS
Backend: Golang (Chi, Echo), Node.js, FastAPI, Flask API, REST APIs, Microservices Architecture, Authenti-
cation & Authorization
AI & Data: LangChain, OpenAI API, Anthropic API, LLM Embeddings, AI Model Integration, Selenium, Play-
wright, Data Scraping, Pandas
Databases: PostgreSQL, MongoDB, SQL Queries
Tools: GitHub, Vercel, Netlify, Cypress Testing, FormKit, Pinia
DevOps: DigitalOcean Server, Vercel, Netlify, GitHub Pages
Professional Experience
Golden Gate Innovations (Remote, US-Based) Apr 2025 – Present
Frontend Developer (Part-time)
• Developing an e-commerce platform using React.js, Panda CSS, and Ark UI, with end-to-end Cypress
testing.
• Implementing responsive UI components, ensuring cross-browser compatibility, and optimizing for perfor-
mance.
• Collaborating remotely with US-based stakeholders, demonstrating adaptability across time zones and cul-
tural contexts.
Datum Brain (Full-time, with Freelance Full-Stack Work Merged) Sep 2023 – Present
Software Engineer & Freelance Full-Stack Developer
• Engineered and optimized scalable web applications using React.js, Vue.js, Next.js, Tailwind CSS.
• Reduced application load times from 8s to under 3s through performance tuning.
• Integrated AI models, built APIs, and developed backend microservices in Golang and Python.
• Led client meetings for requirements gathering and progress reporting.
Shayan Solutions Jun 2023 – Sep 2023
Full Stack Developer — Intern
• Built an AI-based content generator in Next.js.
• Integrated OpenAI APIs and payment gateways (Stripe, PayPal).
• Optimized rendering using SSR and CSR techniques.
Freelancing Feb 2023 – Present
Software Engineer & Freelance Full-Stack Developer
• Delivered freelance projects covering full-stack development, AI integration, and deployment for global clients.
Systems Limited Aug 2021 – Nov 2021
Salesforce Backend Developer — Intern
• Learned Salesforce architecture and SFRA workflows.
Projects
Key Projects
• Query GPT — Built AI models to analyze TikTok content, track trends, and monitor influencer engagement.
Designed a microservices architecture with ETL, GPT, and chat modules. (LangChain, Golang, OpenAI,
Python, Selenium, MongoDB, PostgreSQL, Chi, FastAPI )
• Bidding Extension for Upwork — Developed a browser extension to scrape job postings, rank matches,
and auto-generate proposals using LLMs. (React.js, Selenium, Anthropic)
• Inventory Management System — Implemented role-based authentication, middleware, and secure API
endpoints. (React.js, Golang, Echo, PostgreSQL)
1
• FinTech EMI Application — Web performance, authentication, and compliance checks for user/admin
management. (Vue.js, Tailwind, FormKit, Golang, Echo, PostgreSQL)
Other Projects
• Language Learner / Translator App — Spanish-to-English language learning and translation features
using Google Translate and OpenAI. (React.js, Tailwind CSS, OpenAI API )
• AI Python Models — Voice-based mood predictor and AI solver for Linear Differential Equations. (Next.js,
Python, Flask API )
• Mobile Apps — Messaging app, Blood Bank, Online Mart, News Feed. (Flutter, Firebase)
Education
Bachelor of Science in Computer Science 2019 – 2023
University of Engineering and Technology (UET), Lahore, Pakistan
FSc Pre-Engineering 2017 – 2019
Aspire College, Lahore, Pakistan
Matric in Computer Science 2015 – 2017
Society Public School, Lahore, Pakistan
Interests
Cricket and Planting
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