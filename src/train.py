import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import the pipeline builder from Task 3
from src.data_processing import get_preprocessing_pipeline, TemporalFeatureExtractor

# --- 1. Configuration ---
CONFIG = {
    "experiment_name": "Credit_Risk_Hyperparam_Tuning",
    "data_path": "data/processed/train_data.csv",
    "target": "is_high_risk",
    "test_size": 0.2,
    "random_state": 42,
    # Feature Groups
    "woe_features": ['ProductCategory', 'ProviderId'],
    "ohe_features": ['ChannelId', 'PricingStrategy'],
    "num_features": ['Amount', 'Value', 'Tx_Hour', 'Tx_Day', 'Tx_Month'], # Tx features added by extractor
    # Hyperparameter Grid
    "rf_param_grid": {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [5, 10, 20, None],
        'model__min_samples_split': [2, 5, 10],
        'model__class_weight': ['balanced', 'balanced_subsample', None]
    }
}

def eval_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

def run_training():
    mlflow.set_experiment(CONFIG["experiment_name"])
    
    # 1. Load Data
    df = pd.read_csv(CONFIG["data_path"])
    
    # 2. Pre-split Feature Extraction 
    # (We apply Temporal Extractor here or inside pipeline. Inside pipeline is safer for inference)
    # However, for column selection in ColumnTransformer, we need the columns to exist.
    # Let's use the TemporalExtractor inside the pipeline but we need to pass 'TransactionStartTime'
    
    X = df.drop(columns=[CONFIG["target"], 'CustomerId'])
    y = df[CONFIG["target"]]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["test_size"], stratify=y, random_state=CONFIG["random_state"]
    )

    # 3. Build Preprocessing Pipeline
    # Note: We must ensure X contains 'TransactionStartTime' for the temporal extractor
    # and the logic handles the generated columns. 
    # *Correction*: Standard pipelines apply transformers sequentially. 
    # To use Tx_Hour in ColumnTransformer, it must exist. 
    # Simpler approach for this snippet: Extract time BEFORE passing to get_pipeline
    
    time_ext = TemporalFeatureExtractor()
    X_train = time_ext.transform(X_train)
    X_test = time_ext.transform(X_test)
    
    preprocessor = get_preprocessing_pipeline(
        woe_cols=CONFIG["woe_features"], 
        ohe_cols=CONFIG["ohe_features"], 
        num_cols=CONFIG["num_features"]
    )

    # --- Run A: Baseline Model (Logistic Regression) ---
    with mlflow.start_run(run_name="Baseline_LogReg"):
        baseline_pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', LogisticRegression(max_iter=1000))
        ])
        
        baseline_pipe.fit(X_train, y_train)
        y_pred = baseline_pipe.predict(X_test)
        y_prob = baseline_pipe.predict_proba(X_test)[:, 1]
        
        metrics = eval_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(baseline_pipe, "baseline_model")
        print(f"Baseline AUC: {metrics['roc_auc']:.4f}")

    # --- Run B: Tuned Model (Random Forest with RandomSearch) ---
    with mlflow.start_run(run_name="Tuned_RandomForest"):
        rf_pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestClassifier(random_state=CONFIG["random_state"]))
        ])
        
        # Setup Cross-Validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=CONFIG["random_state"])
        
        search = RandomizedSearchCV(
            estimator=rf_pipe,
            param_distributions=CONFIG["rf_param_grid"],
            n_iter=10, # Number of parameter settings to sample
            scoring='roc_auc',
            cv=cv,
            verbose=1,
            n_jobs=-1,
            random_state=CONFIG["random_state"]
        )
        
        print("Starting Hyperparameter Tuning...")
        search.fit(X_train, y_train)
        
        # Log Best Params
        best_model = search.best_estimator_
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_auc", search.best_score_)
        
        # Evaluate Best Model on Test Set
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        metrics = eval_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)
        
        # Log Best Model
        mlflow.sklearn.log_model(best_model, "best_rf_model")
        print(f"Tuned RF AUC: {metrics['roc_auc']:.4f}")
        print(f"Best Params: {search.best_params_}")

if __name__ == "__main__":
    run_training()