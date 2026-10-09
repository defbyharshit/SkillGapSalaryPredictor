import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SkillGapMatcher:
    """
    NLP Engine for extracting skills, calculating resume-to-job TF-IDF similarity,
    identifying skill gaps, and offering personalized career growth tips.
    """
    
    COMMON_TECH_SKILLS = [
        'python', 'sql', 'pandas', 'numpy', 'scikit-learn', 'pytorch', 'tensorflow',
        'deep learning', 'machine learning', 'nlp', 'computer vision', 'r', 'excel',
        'tableau', 'power bi', 'plotly', 'seaborn', 'jupyter', 'git', 'docker',
        'kubernetes', 'aws', 'gcp', 'azure', 'spark', 'hadoop', 'fastapi', 'flask',
        'streamlit', 'react', 'javascript', 'html5', 'css3', 'postgresql', 'mysql',
        'mongodb', 'snowflake', 'bigquery', 'airflow', 'mlops', 'ci/cd', 'dbt',
        'statistics', 'a/b testing', 'exploratory data analysis', 'feature engineering',
        'data modeling', 'business analysis', 'problem solving', 'communication'
    ]

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

    def clean_text(self, text):
        """Clean and normalize raw text input."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s\+\#\/\.\-]', ' ', text)
        return " ".join(text.split())

    def extract_skills(self, text):
        """Extract recognized tech skills from input string/resume."""
        cleaned = self.clean_text(text)
        found_skills = set()
        
        for skill in self.COMMON_TECH_SKILLS:
            # Word boundary check for skill matching
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, cleaned):
                found_skills.add(skill.title())
                
        return sorted(list(found_skills))

    def calculate_match(self, candidate_resume_text, target_job_description, target_required_skills=None):
        """
        Calculate TF-IDF Cosine Similarity score, matching skills, missing skills,
        and overall match readiness percentage.
        """
        cleaned_resume = self.clean_text(candidate_resume_text)
        cleaned_job = self.clean_text(target_job_description)

        if not cleaned_resume or not cleaned_job:
            return {
                'similarity_score': 0.0,
                'match_percentage': 0.0,
                'candidate_skills': [],
                'matching_skills': [],
                'missing_skills': [],
                'recommendations': ["Please provide both resume details and target job description."]
            }

        # TF-IDF Cosine Similarity
        tfidf_matrix = self.vectorizer.fit_transform([cleaned_resume, cleaned_job])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Skill set extraction
        candidate_skills = set(self.extract_skills(candidate_resume_text))
        
        if target_required_skills:
            if isinstance(target_required_skills, str):
                job_skills = set([s.strip().title() for s in target_required_skills.split(',') if s.strip()])
            else:
                job_skills = set([s.title() for s in target_required_skills])
        else:
            job_skills = set(self.extract_skills(target_job_description))

        matching_skills = sorted(list(candidate_skills.intersection(job_skills)))
        missing_skills = sorted(list(job_skills.difference(candidate_skills)))

        # Hybrid Match Percentage (60% TF-IDF + 40% Direct Skill Overlap)
        skill_coverage = len(matching_skills) / max(len(job_skills), 1)
        hybrid_score = (similarity * 0.5) + (skill_coverage * 0.5)
        match_percentage = round(min(max(hybrid_score * 100, 0), 100), 1)

        # Generate Actionable Recommendations
        recommendations = []
        if missing_skills:
            top_missing = missing_skills[:4]
            recommendations.append(f"🎯 **High Priority Skills to Learn**: Focus on mastering {', '.join(top_missing)}.")
        
        if match_percentage >= 80:
            recommendations.append("🌟 **Excellent Profile Alignment**: You match over 80% of the job requirements!")
        elif match_percentage >= 50:
            recommendations.append("📈 **Moderate Fit**: Highlight any practical projects involving your missing skills to boost your resume.")
        else:
            recommendations.append("🚀 **Skill Building Opportunity**: Consider completing hands-on portfolio projects for missing core skills.")

        return {
            'similarity_score': round(float(similarity), 4),
            'match_percentage': match_percentage,
            'candidate_skills': sorted(list(candidate_skills)),
            'target_skills': sorted(list(job_skills)),
            'matching_skills': matching_skills,
            'missing_skills': missing_skills,
            'recommendations': recommendations
        }
