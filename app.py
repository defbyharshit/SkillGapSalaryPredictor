import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.nlp_matcher import SkillGapMatcher
from src.salary_model import SalaryPredictorModel

# Page configuration
st.set_page_config(
    page_title="AI Job Skill Gap & Salary Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #00B4D8;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #00B4D8;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .skill-chip-match {
        background-color: #065F46;
        color: #D1FAE5;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.88rem;
        margin: 3px;
        display: inline-block;
    }
    .skill-chip-missing {
        background-color: #991B1B;
        color: #FEE2E2;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.88rem;
        margin: 3px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_components():
    matcher = SkillGapMatcher()
    model = SalaryPredictorModel()
    if not model.load():
        model.train()
    return matcher, model

@st.cache_data
def load_dataset():
    if os.path.exists('data/tech_jobs_dataset.csv'):
        return pd.read_csv('data/tech_jobs_dataset.csv')
    return pd.DataFrame()

def main():
    matcher, model = load_components()
    df_jobs = load_dataset()

    st.markdown('<div class="main-header">🎯 AI Job Skill Gap & Salary Analytics Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">NLP-driven Resume Matching & Machine Learning Salary Prediction Engine | Built for CS & Data Science Undergrads</div>', unsafe_allow_html=True)

    # Sidebar parameters
    st.sidebar.header("⚙️ Target Role & Profile Config")

    roles_list = [
        'Data Scientist', 'Machine Learning Engineer', 'Data Analyst', 
        'Data Engineer', 'AI Research Scientist', 'MLOps Engineer', 
        'Full-Stack Data Web Developer', 'Business Intelligence Analyst'
    ]
    target_role = st.sidebar.selectbox("Target Job Role", roles_list, index=0)

    exp_levels = ['Entry-Level (0-2 yrs)', 'Mid-Level (2-5 yrs)', 'Senior-Level (5+ yrs)']
    exp_level = st.sidebar.selectbox("Experience Level", exp_levels, index=0)

    years_exp = st.sidebar.slider("Years of Experience", 0, 12, 1)

    locations = ['Remote', 'San Francisco, CA', 'New York, NY', 'Seattle, WA', 'Austin, TX', 'London, UK', 'Bengaluru, IN']
    location = st.sidebar.selectbox("Target Location", locations, index=0)

    # Filter job template info
    role_jobs = df_jobs[df_jobs['job_title'] == target_role] if not df_jobs.empty else pd.DataFrame()
    sample_skills_str = role_jobs['skills_required'].iloc[0] if not role_jobs.empty else "Python, SQL, Pandas, Scikit-Learn, Machine Learning"
    sample_job_desc = role_jobs['job_description'].iloc[0] if not role_jobs.empty else "Seeking a Data Scientist proficient in Python, SQL, Pandas, Scikit-Learn, Plotly, and Machine Learning algorithms."

    st.sidebar.markdown("---")
    st.sidebar.markdown("📊 **Model Telemetry**")
    st.sidebar.write(f"• **ML Regressor**: RandomForest (120 Trees)")
    st.sidebar.write(f"• **Model R² Score**: `{model.r2_score:.4f}`")
    st.sidebar.write(f"• **Model MAE**: `${model.mae:,.2f}`")
    st.sidebar.write(f"• **Dataset Size**: `{len(df_jobs)} postings`")

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Skill Gap Matcher", 
        "💵 ML Salary Predictor", 
        "📈 Market EDA Insights", 
        "🤖 Architecture & Developer Bio"
    ])

    # Tab 1: Skill Gap Matcher
    with tab1:
        st.subheader("📄 Candidate Resume & Skill Analyzer")
        st.write("Paste your resume summary or list of technical skills to compare against the target role requirements using TF-IDF Natural Language Processing.")

        col_input1, col_input2 = st.columns(2)

        default_resume = (
            "Harshit - B.Tech CSE (Data Science) 1st Year Undergrad.\n"
            "Technical Skills: Python, SQL, Pandas, NumPy, Scikit-Learn, Streamlit, Plotly, "
            "Exploratory Data Analysis, Machine Learning, Git, VS Code, HTML5, CSS3, JavaScript."
        )

        with col_input1:
            resume_text = st.text_area("Your Resume / Skill Summary", value=default_resume, height=180)
        
        with col_input2:
            target_desc = st.text_area("Target Job Description", value=sample_job_desc, height=180)

        if st.button("🚀 Analyze Skill Match", type="primary", use_container_width=True):
            with st.spinner("Processing TF-IDF Cosine Similarity & extracting skill tokens..."):
                match_results = matcher.calculate_match(resume_text, target_desc, sample_skills_str)

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                st.metric("Overall Match Readiness", f"{match_results['match_percentage']}%")
            with col_m2:
                st.metric("TF-IDF Cosine Similarity", f"{match_results['similarity_score']}")
            with col_m3:
                st.metric("Matching Skills Found", f"{len(match_results['matching_skills'])}")
            with col_m4:
                st.metric("Skill Gaps Identified", f"{len(match_results['missing_skills'])}")

            st.markdown("---")

            col_skills1, col_skills2 = st.columns(2)

            with col_skills1:
                st.markdown("### ✅ Matching Competencies")
                if match_results['matching_skills']:
                    chips_html = "".join([f'<span class="skill-chip-match">✓ {s}</span>' for s in match_results['matching_skills']])
                    st.markdown(chips_html, unsafe_allow_html=True)
                else:
                    st.info("No explicit tech skill overlaps found.")

            with col_skills2:
                st.markdown("### ⚠️ Target Skill Gaps to Acquire")
                if match_results['missing_skills']:
                    chips_html = "".join([f'<span class="skill-chip-missing">✗ {s}</span>' for s in match_results['missing_skills']])
                    st.markdown(chips_html, unsafe_allow_html=True)
                else:
                    st.success("🎉 You meet all extracted skill requirements for this role!")

            st.markdown("---")
            st.markdown("### 💡 Tailored Career Development Recommendations")
            for rec in match_results['recommendations']:
                st.markdown(rec)

            # Plotly Skill Radar Chart
            st.markdown("### 🕸️ Skill Coverage Radar Visualizer")
            categories = ['Core Data Tools', 'Machine Learning', 'Big Data & Cloud', 'Web & APIs', 'Domain Knowledge']
            
            # Simple category scoring heuristic
            cand_score = min(len(match_results['matching_skills']) * 20, 100)
            target_score = 100

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[cand_score, cand_score*0.9, cand_score*0.7, cand_score*0.85, cand_score*0.95],
                theta=categories,
                fill='toself',
                name='Your Profile',
                line_color='#00B4D8'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[target_score]*5,
                theta=categories,
                fill='toself',
                name='Target Benchmark',
                line_color='#94A3B8',
                opacity=0.3
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                height=400,
                margin=dict(l=40, r=40, t=30, b=30)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # Tab 2: Salary Predictor
    with tab2:
        st.subheader("💵 Machine Learning Salary Estimation Engine")
        st.write("Predict estimated market compensation based on your selected role, experience level, location, and technical skill set.")

        custom_skills = st.text_input("Technical Skills for Salary Model", value=sample_skills_str)

        if st.button("💰 Estimate Salary Range", type="primary", use_container_width=True):
            salary_data = model.predict(target_role, exp_level, years_exp, location, custom_skills)

            st.markdown("### 📊 Predicted Compensation Benchmark")
            
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Estimated Minimum Salary", f"${salary_data['min_salary']:,} USD")
            with col_s2:
                st.metric("🎯 Predicted Mean Salary", f"${salary_data['predicted_salary']:,} USD", delta=f"{location}")
            with col_s3:
                st.metric("Estimated Maximum Salary", f"${salary_data['max_salary']:,} USD")

            # Salary Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=salary_data['predicted_salary'],
                number={'prefix': "$", 'valueformat': ","},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Salary Benchmark ({target_role} - {location})"},
                gauge={
                    'axis': {'range': [None, 250000]},
                    'bar': {'color': "#00B4D8"},
                    'steps': [
                        {'range': [0, 75000], 'color': "#1E293B"},
                        {'range': [75000, 140000], 'color': "#334155"},
                        {'range': [140000, 250000], 'color': "#475569"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': salary_data['predicted_salary']
                    }
                }
            ))
            fig_gauge.update_layout(height=350)
            st.plotly_chart(fig_gauge, use_container_width=True)

    # Tab 3: Market EDA Insights
    with tab3:
        st.subheader("📈 Tech Market Exploratory Data Analysis")
        st.write("Explore industry dataset trends across roles, locations, experience tiers, and compensation levels.")

        if not df_jobs.empty:
            col_eda1, col_eda2 = st.columns(2)

            with col_eda1:
                fig_hist = px.histogram(
                    df_jobs, x="salary_in_usd", color="job_title", 
                    nbins=30, title="Salary Distribution by Tech Role",
                    labels={'salary_in_usd': 'Salary (USD)'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_eda2:
                fig_box = px.box(
                    df_jobs, x="location", y="salary_in_usd", color="experience_level",
                    title="Salary Ranges across Global Tech Hubs",
                    labels={'salary_in_usd': 'Salary (USD)', 'location': 'Location'}
                )
                st.plotly_chart(fig_box, use_container_width=True)

            st.markdown("### 📋 Explore Job Dataset")
            st.dataframe(df_jobs.head(100), use_container_width=True)

    # Tab 4: Architecture & Bio
    with tab4:
        st.subheader("🤖 System Architecture & Developer Identity")

        st.markdown("""
        #### 🏗️ System Architecture Flow
        ```text
        +-----------------------------------------------------------------------+
        |                 User Inputs (Resume Text & Target Config)             |
        +-----------------------------------+-----------------------------------+
                                            |
                                            v
        +-----------------------------------+-----------------------------------+
        |                      NLP Skill Matching Engine                        |
        |  1. Text Preprocessing & Cleaning | 2. TF-IDF Token Vectorizer        |
        |  3. Cosine Similarity Calculation  | 4. Skill Gap Set Extraction       |
        +-----------------------------------+-----------------------------------+
                                            |
                                            v
        +-----------------------------------+-----------------------------------+
        |                Scikit-Learn ML Salary Regressor                       |
        |  1. OneHotEncoder (Cat Features)   | 2. StandardScaler (Numeric)       |
        |  3. CountVectorizer (Skills Text)  | 4. RandomForest (120 Trees)       |
        +-----------------------------------+-----------------------------------+
                                            |
                                            v
        +-----------------------------------------------------------------------+
        |                  Interactive Streamlit Dashboard                      |
        |    Match Score %   |   Salary Gauge   |  Radar Chart  |   Market EDA   |
        +-----------------------------------------------------------------------+
        ```

        #### 👤 Developer Identity
        * **Developer**: **Harshit** ([@defbyharshit](https://github.com/defbyharshit))
        * **Role**: 1st Year B.Tech Computer Science & Engineering (Specialization: **Data Science**)
        * **Core Stack**: Python, NumPy, Pandas, Scikit-Learn, Streamlit, SQL, Plotly, Git.
        """)

if __name__ == '__main__':
    main()
