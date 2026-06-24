import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load English NLP model (Ensure 'python -m spacy download en_core_web_sm' is run)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Fallback/warning if not installed, though we should handle this via instructions
    print("Warning: spacy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

def extract_keywords(text: str) -> set:
    """
    Extracts keywords (nouns, proper nouns, and technical terms) from text using spaCy.
    """
    if not nlp:
        # Fallback to simple split if spaCy model isn't loaded
        words = re.findall(r'\b\w+\b', text.lower())
        return set(words)
        
    doc = nlp(text.lower())
    keywords = set()
    for token in doc:
        # Filter for nouns, proper nouns, adjectives, or terms that are NOT stop words
        if token.pos_ in ["NOUN", "PROPN", "ADJ", "VERB"] and not token.is_stop and not token.is_punct:
            keywords.add(token.lemma_)
    return keywords

def calculate_keyword_match(cv_text: str, jd_text: str) -> dict:
    """
    Matches keywords from the job description against the CV.
    Returns matched keywords, missing keywords, and a score (0-100).
    """
    cv_keywords = extract_keywords(cv_text)
    jd_keywords = extract_keywords(jd_text)
    
    if not jd_keywords:
        return {"score": 0, "matched": [], "missing": []}
        
    matched = jd_keywords.intersection(cv_keywords)
    missing = jd_keywords.difference(cv_keywords)
    
    score = (len(matched) / len(jd_keywords)) * 100
    return {
        "score": score,
        "matched": list(matched),
        "missing": list(missing)
    }

def detect_sections(cv_text: str) -> float:
    """
    Detects standard CV sections using regex.
    Returns a score from 0 to 100 based on found sections.
    """
    sections_to_find = [
        r"(?i)\b(experience|work history|employment)\b",
        r"(?i)\b(education|academic|qualifications)\b",
        r"(?i)\b(skills|technical skills|competencies)\b",
        r"(?i)\b(projects|personal projects)\b"
    ]
    
    found_count = 0
    for pattern in sections_to_find:
        if re.search(pattern, cv_text):
            found_count += 1
            
    score = (found_count / len(sections_to_find)) * 100
    return score

def calculate_similarity(cv_text: str, jd_text: str) -> float:
    """
    Calculates TF-IDF cosine similarity between the CV and Job Description.
    Returns a score from 0 to 100.
    """
    if not cv_text.strip() or not jd_text.strip():
        return 0.0
        
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([cv_text, jd_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity * 100)
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0

def calculate_ats_score(cv_text: str, jd_text: str) -> dict:
    """
    Combines different metrics to produce a final ATS score.
    Weights: Keyword Match (50%), Section Presence (25%), Similarity Alignment (25%).
    """
    keyword_results = calculate_keyword_match(cv_text, jd_text)
    section_score = detect_sections(cv_text)
    similarity_score = calculate_similarity(cv_text, jd_text)
    
    # Calculate weighted final score
    final_score = (keyword_results["score"] * 0.5) + (section_score * 0.25) + (similarity_score * 0.25)
    
    return {
        "overall_score": round(final_score, 1),
        "breakdown_scores": {
            "keyword_match": round(keyword_results["score"], 1),
            "section_presence": round(section_score, 1),
            "similarity_alignment": round(similarity_score, 1)
        },
        "matched_keywords": keyword_results["matched"],
        "missing_keywords": keyword_results["missing"]
    }
