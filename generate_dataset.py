import os
import random
import pandas as pd
import numpy as np

def generate_job_dataset(num_samples=1500, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    roles = {
        'Data Scientist': {
            'core_skills': ['Python', 'SQL', 'Pandas', 'Scikit-Learn', 'Exploratory Data Analysis', 'Machine Learning', 'Statistics', 'Plotly', 'Jupyter'],
            'advanced_skills': ['PyTorch', 'TensorFlow', 'Deep Learning', 'NLP', 'A/B Testing', 'BigQuery', 'Spark', 'Feature Engineering'],
            'base_salary': 95000,
            'skill_bonus': 4000
        },
        'Machine Learning Engineer': {
            'core_skills': ['Python', 'PyTorch', 'Scikit-Learn', 'Git', 'Docker', 'Machine Learning', 'SQL', 'FastAPI'],
            'advanced_skills': ['MLOps', 'TensorFlow', 'CUDA', 'AWS', 'Kubernetes', 'CI/CD', 'Model Deployment', 'ONNX'],
            'base_salary': 105000,
            'skill_bonus': 5000
        },
        'Data Analyst': {
            'core_skills': ['SQL', 'Python', 'Excel', 'Tableau', 'Power BI', 'Pandas', 'Data Visualization', 'Communication'],
            'advanced_skills': ['Statistics', 'A/B Testing', 'Snowflake', 'Looker', 'R', 'ETL', 'Dashboarding'],
            'base_salary': 72000,
            'skill_bonus': 3000
        },
        'Data Engineer': {
            'core_skills': ['Python', 'SQL', 'Spark', 'ETL', 'PostgreSQL', 'Git', 'Data Warehousing', 'Airflow'],
            'advanced_skills': ['Kafka', 'AWS', 'Snowflake', 'BigQuery', 'Scala', 'Docker', 'dbt', 'Kubernetes'],
            'base_salary': 100000,
            'skill_bonus': 4500
        },
        'AI Research Scientist': {
            'core_skills': ['Python', 'PyTorch', 'Deep Learning', 'Computer Vision', 'NLP', 'Mathematics', 'Statistics', 'Jupyter'],
            'advanced_skills': ['Transformers', 'LLMs', 'Reinforcement Learning', 'LaTeX', 'CUDA', 'Distributed Training'],
            'base_salary': 125000,
            'skill_bonus': 6000
        },
        'MLOps Engineer': {
            'core_skills': ['Python', 'Docker', 'Kubernetes', 'MLOps', 'CI/CD', 'AWS', 'Git', 'FastAPI'],
            'advanced_skills': ['Terraform', 'Mlflow', 'Prometheus', 'Airflow', 'PyTorch', 'Triton', 'GCP'],
            'base_salary': 110000,
            'skill_bonus': 5000
        },
        'Full-Stack Data Web Developer': {
            'core_skills': ['Python', 'JavaScript', 'HTML5', 'CSS3', 'React', 'FastAPI', 'SQL', 'Git'],
            'advanced_skills': ['Streamlit', 'Docker', 'PostgreSQL', 'Tailwind', 'REST APIs', 'Node.js', 'AWS'],
            'base_salary': 88000,
            'skill_bonus': 3500
        },
        'Business Intelligence Analyst': {
            'core_skills': ['SQL', 'Power BI', 'Tableau', 'Excel', 'Data Modeling', 'Business Analysis', 'Communication'],
            'advanced_skills': ['Python', 'DAX', 'Snowflake', 'ETL', 'Statistics', 'Financial Modeling'],
            'base_salary': 75000,
            'skill_bonus': 3000
        }
    }
    
    exp_levels = {
        'Entry-Level (0-2 yrs)': {'mult': 0.82, 'exp_num': 1},
        'Mid-Level (2-5 yrs)': {'mult': 1.15, 'exp_num': 3.5},
        'Senior-Level (5+ yrs)': {'mult': 1.55, 'exp_num': 7}
    }
    
    locations = {
        'Remote': 1.05,
        'San Francisco, CA': 1.35,
        'New York, NY': 1.30,
        'Seattle, WA': 1.25,
        'Austin, TX': 1.10,
        'Bengaluru, IN': 0.45,
        'London, UK': 0.90
    }
    
    data = []
    
    for i in range(1, num_samples + 1):
        job_id = f"JOB-{i:04d}"
        role_name = random.choice(list(roles.keys()))
        role_info = roles[role_name]
        
        # Pick skills
        num_core = random.randint(4, len(role_info['core_skills']))
        num_adv = random.randint(1, len(role_info['advanced_skills']))
        selected_core = random.sample(role_info['core_skills'], num_core)
        selected_adv = random.sample(role_info['advanced_skills'], num_adv)
        all_skills = selected_core + selected_adv
        
        # Soft skills addition
        soft_skills = random.sample(['Problem Solving', 'Teamwork', 'Agile', 'Critical Thinking', 'Git Workflow'], random.randint(1, 2))
        skills_str = ", ".join(all_skills + soft_skills)
        
        exp_name = random.choice(list(exp_levels.keys()))
        exp_info = exp_levels[exp_name]
        
        loc_name = random.choice(list(locations.keys()))
        loc_mult = locations[loc_name]
        
        # Calculate salary realistically
        base = role_info['base_salary']
        skill_boost = len(all_skills) * role_info['skill_bonus']
        noise = random.randint(-6000, 8000)
        
        calculated_salary = int((base + skill_boost + noise) * exp_info['mult'] * loc_mult)
        
        # Generate description text
        description = (
            f"We are seeking a talented {role_name} ({exp_name}) located in {loc_name}. "
            f"The ideal candidate will have strong expertise in {', '.join(selected_core[:3])}. "
            f"You will work with high-performing teams leveraging tools like {', '.join(selected_adv[:2])} to build scalable solutions. "
            f"Required key competencies include: {skills_str}. "
            f"Responsibilities include data modeling, algorithm optimization, automated pipelines, and cross-functional collaboration."
        )
        
        data.append({
            'job_id': job_id,
            'job_title': role_name,
            'experience_level': exp_name,
            'years_experience': exp_info['exp_num'],
            'location': loc_name,
            'skills_required': skills_str,
            'skill_count': len(all_skills),
            'job_description': description,
            'salary_in_usd': calculated_salary
        })
        
    df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/tech_jobs_dataset.csv', index=False)
    print(f"Generated {len(df)} job postings in data/tech_jobs_dataset.csv!")
    return df

if __name__ == '__main__':
    generate_job_dataset()
