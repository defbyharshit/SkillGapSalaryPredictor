import os
from generate_dataset import generate_job_dataset
from src.salary_model import SalaryPredictorModel

def main():
    dataset_path = 'data/tech_jobs_dataset.csv'
    if not os.path.exists(dataset_path):
        print("Dataset not found. Generating dataset...")
        generate_job_dataset(num_samples=1500)
    else:
        print(f"Found existing dataset at {dataset_path}.")

    print("Training Machine Learning Salary Prediction Pipeline...")
    model = SalaryPredictorModel()
    r2, mae = model.train(dataset_path)
    print("--- Model Training Complete! ---")
    print(f"Evaluation Metrics -> R2 Score: {r2:.4f} | Mean Absolute Error (MAE): ${mae:,.2f}")

if __name__ == '__main__':
    main()
