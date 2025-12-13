import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_model():
    # Load processed data
    df = pd.read_csv('data/processed/train_data.csv')
    
    # Separate Features and Target
    X = df.drop(columns=['CustomerId', 'is_high_risk'])
    y = df['is_high_risk']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Set experiment
    mlflow.set_experiment("Credit_Risk_Model_Bati")
    
    # Define models to test
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_prob)
            }
            
            # Log params and metrics
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            
            # Log Model
            mlflow.sklearn.log_model(model, name)
            
            print(f"{name} trained. AUC: {metrics['roc_auc']}")

if __name__ == "__main__":
    train_model()