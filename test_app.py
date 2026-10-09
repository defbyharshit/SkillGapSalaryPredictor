from src.nlp_matcher import SkillGapMatcher
from src.salary_model import SalaryPredictorModel

def test_pipeline():
    print("Testing NLP Skill Matcher...")
    matcher = SkillGapMatcher()
    resume = "Python, SQL, Pandas, Scikit-Learn, Streamlit, Machine Learning, Plotly"
    job_desc = "Looking for a Data Scientist with Python, SQL, PyTorch, Docker, Machine Learning expertise."
    
    match_res = matcher.calculate_match(resume, job_desc, "Python, SQL, PyTorch, Docker, Machine Learning")
    print(f"Match Readiness: {match_res['match_percentage']}%")
    print(f"Matching Skills: {match_res['matching_skills']}")
    print(f"Missing Skills: {match_res['missing_skills']}")

    print("\nTesting ML Salary Predictor Model...")
    model = SalaryPredictorModel()
    salary_res = model.predict(
        job_title='Data Scientist',
        experience_level='Entry-Level (0-2 yrs)',
        years_exp=1,
        location='Remote',
        skills_str='Python, SQL, Pandas, Scikit-Learn, Machine Learning'
    )
    print(f"Predicted Salary: ${salary_res['predicted_salary']:,} USD")
    print(f"Salary Range: ${salary_res['min_salary']:,} - ${salary_res['max_salary']:,} USD")
    print("Pipeline Test Passed Successfully!")

if __name__ == '__main__':
    test_pipeline()
