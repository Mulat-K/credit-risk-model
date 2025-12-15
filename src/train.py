import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Configuration
EXPERIMENT_NAME = "Credit_Risk_Model_Comparison"
DATA_PATH = 'data/processed/train_data.csv'

def train_models():
    # 1. Load Data
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print("Data not found. Please run data_processing.py first.")
        return

    # Prepare X and y
    # Assuming 'is_high_risk' is the target and CustomerId is an ID
    X = df.drop(columns=['is_high_risk', 'CustomerId'], errors='ignore')
    y = df['is_high_risk']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Define Models to Compare
    # We include Logistic Regression (baseline, interpretable) and Random Forest (complex)
    models_to_train = [
        (
            "Logistic_Regression", 
            LogisticRegression(max_iter=1000, class_weight='balanced'),
            {"clf__C": 1.0}
        ),
        (
            "Random_Forest", 
            RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
            {"clf__max_depth": 10}
        )
    ]

    mlflow.set_experiment(EXPERIMENT_NAME)

    for model_name, model_instance, params in models_to_train:
        with mlflow.start_run(run_name=model_name):
            print(f"Training {model_name}...")
            
            # Create Pipeline (Imputation + Scaling is standard for LogReg)
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('clf', model_instance)
            ])
            
            # Fit
            pipeline.fit(X_train, y_train)
            
            # Predict
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            
            # 3. Compute ALL Metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_prob)
            }
            
            # Log Metrics & Params
            mlflow.log_metrics(metrics)
            mlflow.log_params(params) # Log specific params we set
            
            print(f"  --> ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  --> F1 Score: {metrics['f1']:.4f}")

            # 4. Generate & Log Artifacts (Confusion Matrix)
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'Confusion Matrix: {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            plot_filename = f"{model_name}_confusion_matrix.png"
            plt.savefig(plot_filename)
            plt.close()
            
            mlflow.log_artifact(plot_filename)
            
            # Log Classification Report as text artifact
            report = classification_report(y_test, y_pred)
            with open(f"{model_name}_report.txt", "w") as f:
                f.write(report)
            mlflow.log_artifact(f"{model_name}_report.txt")

            # Log Model
            mlflow.sklearn.log_model(pipeline, "model")

if __name__ == "__main__":
    train_models()