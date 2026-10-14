import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

class SalaryPredictorModel:
    """
    Scikit-Learn Machine Learning Pipeline for predicting tech salaries
    based on Job Title, Experience Level, Location, and Technical Skills.
    """
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, 'salary_model_pipeline.joblib')
        self.pipeline = None
        self.r2_score = None
        self.mae = None

    def build_pipeline(self):
        """Construct Scikit-Learn ColumnTransformer and Pipeline."""
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['job_title', 'experience_level', 'location']),
                ('num', StandardScaler(), ['years_experience', 'skill_count']),
                ('skills_text', CountVectorizer(token_pattern=r'(?u)\b[\w\+\#\.-]+\b'), 'skills_required')
            ]
        )

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=120, max_depth=15, random_state=42, n_jobs=-1))
        ])
        
        return pipeline

    def train(self, dataset_path='data/tech_jobs_dataset.csv'):
        """Train ML pipeline on dataset and save model weights."""
        if not os.path.exists(dataset_path):
            from generate_dataset import generate_job_dataset
            generate_job_dataset(num_samples=1500)

        df = pd.read_csv(dataset_path)

        X = df[['job_title', 'experience_level', 'location', 'years_experience', 'skill_count', 'skills_required']]
        y = df['salary_in_usd']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X_train, y_train)

        # Evaluate model performance
        y_pred = self.pipeline.predict(X_test)
        self.r2_score = round(r2_score(y_test, y_pred), 4)
        self.mae = round(mean_absolute_error(y_test, y_pred), 2)

        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump({
            'pipeline': self.pipeline,
            'r2_score': self.r2_score,
            'mae': self.mae
        }, self.model_path)

        print(f"Model trained successfully! R² Score: {self.r2_score:.4f}, MAE: ${self.mae:,.2f}")
        return self.r2_score, self.mae

    def load(self):
        """Load pre-trained model pipeline from disk."""
        if not os.path.exists(self.model_path):
            return False
        
        data = joblib.load(self.model_path)
        self.pipeline = data['pipeline']
        self.r2_score = data['r2_score']
        self.mae = data['mae']
        return True

    def predict(self, job_title, experience_level, years_exp, location, skills_str):
        """
        Predict estimated salary range ($Min, $Predicted, $Max) for a candidate/job profile.
        """
        if self.pipeline is None:
            if not self.load():
                raise RuntimeError("Model pipeline is not trained or loaded. Train the model first.")

        skill_count = len([s for s in skills_str.split(',') if s.strip()])

        input_df = pd.DataFrame([{
            'job_title': job_title,
            'experience_level': experience_level,
            'location': location,
            'years_experience': years_exp,
            'skill_count': skill_count,
            'skills_required': skills_str
        }])

        pred_salary = float(self.pipeline.predict(input_df)[0])
        
        # Calculate realistic range bounds (+/- 8% margin of error)
        min_salary = int(pred_salary * 0.92)
        max_salary = int(pred_salary * 1.08)
        predicted_salary = int(pred_salary)

        return {
            'predicted_salary': predicted_salary,
            'min_salary': min_salary,
            'max_salary': max_salary,
            'r2_score': self.r2_score,
            'mae': self.mae
        }
