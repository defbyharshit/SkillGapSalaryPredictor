# 🎯 AI Job Skill Gap & Salary Analytics Engine

[![Python](https://img.shields.io/badge/Python-3.14-00B4D8?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-Dashboards-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com)
[![R2 Score](https://img.shields.io/badge/Model%20R%C2%B2-0.9726-065F46?style=flat-square&logo=target)](https://github.com/defbyharshit)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

An end-to-end **Data Science & Natural Language Processing (NLP)** web application that analyzes candidate resumes against target job descriptions to extract skill gaps, evaluate match readiness, and predict target tech salary benchmarks using Machine Learning.

Developed by **[Harshit](https://github.com/defbyharshit)** (1st Year B.Tech Computer Science & Engineering - Data Science).

---

## ✨ Key Features

* **📄 NLP Resume & Job Matcher**: Uses **TF-IDF Vectorization** and **Cosine Similarity** to compute match readiness scores (0–100%) between candidate resumes and target job descriptions.
* **⚠️ Automatic Skill Gap Extraction**: Identifies matching competencies vs missing target technical skills (e.g. PyTorch, Docker, MLOps, SQL).
* **🕸️ Interactive Radar Visualization**: Displays skill set coverage across core data science domains using Plotly polar charts.
* **💵 ML Salary Regressor**: Powered by a **Scikit-Learn RandomForestRegressor** pipeline (`R² = 0.9726`, `MAE = $9,730`) trained on 1,500+ tech job postings.
* **📊 Market EDA Dashboard**: Explores global compensation distributions across job titles, experience tiers, and tech hubs (San Francisco, New York, Remote, London, Bengaluru).

---

## 🏗️ System Architecture

```text
       +-------------------------------------------------------------+
       |             Candidate Resume & Job Config Input             |
       |  [Resume Text] [Target Role] [Location] [Experience Tier]   |
       +------------------------------+------------------------------+
                                      |
                                      v
       +------------------------------+------------------------------+
       |             Natural Language Processing (NLP)               |
       |  1. Text Preprocessing & Cleaning | 2. TF-IDF Tokenizer      |
       |  3. Cosine Similarity Score       | 4. Skill Gap Extraction  |
       +------------------------------+------------------------------+
                                      |
                                      v
       +------------------------------+------------------------------+
       |           Scikit-Learn ML Salary Regressor Pipeline          |
       |  OneHotEncoder (Category) | StandardScaler | RandomForest   |
       +------------------------------+------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |               Interactive Streamlit Web Dashboard           |
       |  Match Readiness % | Salary Gauge | Skill Radar | EDA Charts|
       +-------------------------------------------------------------+
```

---

## 🛠️ Tech Stack & Dependencies

* **Programming**: Python 3.14
* **Data Science & ML**: NumPy, Pandas, Scikit-Learn, Joblib
* **NLP & Text Processing**: TF-IDF, Cosine Similarity, Regex Tokenizer, NLTK
* **Web & Data Viz**: Streamlit, Plotly Express & Graph Objects

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/defbyharshit/SkillGapSalaryPredictor.git
cd SkillGapSalaryPredictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the ML Model & Generate Dataset
```bash
python train_model.py
```

### 4. Launch the Streamlit Web Application
```bash
streamlit run app.py
```

---

## 📈 Model Performance & Telemetry

* **Machine Learning Model**: Scikit-Learn `RandomForestRegressor` (120 Estimators, Max Depth = 15)
* **Dataset Size**: 1,500 Tech Job Postings
* **R² Score**: `0.9726` (97.26% Variance Explained)
* **Mean Absolute Error (MAE)**: `$9,730.35 USD`

---

## 👤 Author & Identity

* **Developer**: **Harshit**
* **GitHub Profile**: [@defbyharshit](https://github.com/defbyharshit)
* **Academic Background**: 1st Year B.Tech CSE (Specialization: **Data Science**)
* **Engineering Philosophy**: *"Data is raw fuel; clean architecture and algorithms are the rocket engine."*

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
