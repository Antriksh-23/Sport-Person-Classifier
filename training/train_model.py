import os
import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import json

def build_models():
    """
    Returns a dictionary of models to evaluate with their hyperparameter grids.
    """
    from sklearn.calibration import CalibratedClassifierCV
    models = {
        'svm': {
            'model': CalibratedClassifierCV(SVC(class_weight='balanced'), ensemble=False),
            'params': {
                'calibratedclassifiercv__estimator__C': [1, 10, 100, 1000],
                'calibratedclassifiercv__estimator__kernel': ['rbf', 'linear']
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(class_weight='balanced'),
            'params': {
                'randomforestclassifier__n_estimators': [10, 50, 100],
                'randomforestclassifier__max_depth': [None, 5, 10]
            }
        },
        'logistic_regression': {
            'model': LogisticRegression(class_weight='balanced', max_iter=1000),
            'params': {
                'logisticregression__C': [1, 5, 10]
            }
        }
    }
    return models

def train_and_evaluate(X, y, class_dict, models_dir="models", eval_dir="evaluation"):
    """
    Trains models, performs hyperparameter tuning, evaluates the best model, 
    and saves artifacts to disk.
    """
    if len(X) == 0:
        print("No training data provided.")
        return
        
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = build_models()
    scores = []
    best_estimators = {}
    
    # 6. Model training & 7. Hyperparameter tuning
    print("Starting Grid Search CV...")
    for algo_name, config in models.items():
        # Using a pipeline to include the Step 9: Normalization (StandardScaler)
        pipe = make_pipeline(StandardScaler(), config['model'])
        clf = GridSearchCV(pipe, config['params'], cv=5, return_train_score=False)
        clf.fit(X_train, y_train)
        
        scores.append({
            'model': algo_name,
            'best_score': clf.best_score_,
            'best_params': clf.best_params_
        })
        best_estimators[algo_name] = clf.best_estimator_
        
    df_scores = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
    print("\nGrid Search Results:")
    print(df_scores)
    
    # Select the best model automatically based on validation score
    best_model_name = df_scores.loc[df_scores['best_score'].idxmax()]['model']
    best_clf = best_estimators[best_model_name]
    
    print(f"\nBest model selected: {best_model_name}")
    
    # 8. Model Evaluation on test set
    y_pred = best_clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    
    print("\nEvaluation Metrics on Test Set:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Save evaluation artifacts
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {acc}\nPrecision: {prec}\nRecall: {rec}\nF1: {f1}\n\n")
        f.write("Classification Report:\n" + report)
        
    # Model Persistence
    os.makedirs(models_dir, exist_ok=True)
    # Save the best model
    joblib.dump(best_clf, os.path.join(models_dir, 'saved_model.pkl'))
    
    # Save class dictionary
    with open(os.path.join(models_dir, 'class_dictionary.json'), 'w') as f:
        json.dump(class_dict, f)
        
    print(f"Model and class dictionary saved to {models_dir}")
    
    return best_clf
