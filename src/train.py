import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from src.data_processing import get_training_pipeline

# Configuration
CAT_COLS = ['ProductCategory', 'ChannelId', 'PricingStrategy']
NUM_COLS = ['Amount', 'Tx_Hour', 'Tx_Day'] # Assuming these exist after basic aggregation
TARGET = 'is_high_risk'

def train_and_tune():
    # 1. Load Data
    df = pd.read_csv('data/processed/train_data.csv')
    
    # Split Data (Stratified because default risk is usually imbalanced)
    X = df.drop(columns=[TARGET, 'CustomerId'])
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Setup Pipeline
    # Note: We pass the preprocessor into a full pipeline including the model
    # This ensures WoE and Scaling happen inside Cross-Validation folds to prevent leakage
    preprocessor = get_training_pipeline(CAT_COLS, NUM_COLS)
    
    gbm = GradientBoostingClassifier(random_state=42)
    
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', gbm)
    ])

    # 3. Hyperparameter Grid
    param_dist = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 4, 5],
        'classifier__subsample': [0.8, 1.0]
    }

    # 4. MLflow Experiment
    mlflow.set_experiment("Credit_Risk_Optimization")
    
    with mlflow.start_run() as run:
        print("Starting Hyperparameter Tuning...")
        
        # Randomized Search with Stratified CV
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        search = RandomizedSearchCV(
            full_pipeline, 
            param_distributions=param_dist, 
            n_iter=5, # Keep low for demo, increase for prod
            scoring='roc_auc',
            cv=cv,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        
        search.fit(X_train, y_train)
        
        # 5. Log Best Params
        best_model = search.best_estimator_
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_auc", search.best_score_)
        
        # 6. Evaluate on Test Set
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        test_auc = roc_auc_score(y_test, y_prob)
        mlflow.log_metric("test_auc", test_auc)
        
        # 7. Generate and Log Artifacts (Plots)
        
        # ROC Curve
        fig, ax = plt.subplots()
        RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=ax)
        plt.title("ROC Curve")
        plt.savefig("roc_curve.png")
        mlflow.log_artifact("roc_curve.png")
        plt.close()
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # 8. Save Model
        # Using signature ensures input schema is enforced during deployment
        signature = mlflow.models.infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(
            best_model, 
            "model", 
            signature=signature,
            input_example=X_train.iloc[:5]
        )
        
        print(f"Run ID: {run.info.run_id}")
        print(f"Test AUC: {test_auc}")

if __name__ == "__main__":
    train_and_tune()