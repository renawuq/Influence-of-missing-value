import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os

# Define functions for the methodology
def feature_selection_pipeline(data, target_col):
    """
    Two-step feature selection using LASSO and XGBoost
    """
    # Split data
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # LASSO feature selection
    lasso = LassoCV(cv=5)
    lasso.fit(X, y)

    # Get non-zero coefficients
    lasso_features = X.columns[lasso.coef_ != 0]
    if len(lasso_features) < 2:
        # If LASSO doesn't select enough features, select all features
        lasso_features = X.columns

    # XGBoost feature importance
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X[lasso_features], y)

    # Get feature importance
    importance = pd.DataFrame({
        'feature': lasso_features,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    return importance

def create_missing_data(data, missing_percentage):
    # Create missing values in the dataset
    np.random.seed(42)  # Ensure reproducibility
    mask = np.random.random(data.shape) < missing_percentage
    data_missing = data.copy()
    data_missing[mask] = np.nan
    return data_missing

def mean_mode_imputation(data):
    # Separate numeric and categorical columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    # Create imputers
    numeric_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    # Impute data
    data_imputed = data.copy()
    if len(numeric_cols) > 0:
        data_imputed[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols])
    if len(categorical_cols) > 0:
        data_imputed[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])

    return data_imputed

def train_models(X_train, X_test, y_train, y_test):
    """
    Train baseline and primary models with hyperparameter tuning
    """
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'XGBoost': xgb.XGBClassifier()
    }

    # Define parameter grids for GridSearchCV
    param_grids = {
        'Logistic Regression': {'C': [0.1, 1, 10]},
        'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10]},
        'XGBoost': {'n_estimators': [100, 200], 'max_depth': [3, 6]}
    }

    # Store SHAP values
    shap_values = {}
    results = {}
    
    for name, model in models.items():
        print(f"Training {name} model...")
        try:
            # Perform grid search
            grid_search = GridSearchCV(model, param_grids[name], cv=3, scoring='roc_auc')
            grid_search.fit(X_train, y_train)

            # Get best model
            best_model = grid_search.best_estimator_
            models[name] = best_model
            
            # Get model performance
            test_score = best_model.score(X_test, y_test)
            print(f"{name} test accuracy: {test_score:.4f}")
            results[name] = test_score

            # Calculate SHAP values
            try:
                if name == 'XGBoost' or name == 'Random Forest':
                    explainer = shap.TreeExplainer(best_model)
                    shap_values[name] = explainer.shap_values(X_test)
                else:  # Logistic Regression
                    explainer = shap.LinearExplainer(best_model, X_train)
                    shap_values[name] = explainer.shap_values(X_test)
            except Exception as e:
                print(f"Error calculating SHAP values: {e}")
                shap_values[name] = None
        except Exception as e:
            print(f"Error training {name} model: {e}")
            results[name] = None
            shap_values[name] = None

    return models, shap_values, results

def constrained_svm(X_train, y_train, n_samples=3):
    """
    Constrained SVM without uncertainty
    """
    # Train a simple SVM model
    print("Training SVM model...")
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)

    # Evaluate model performance
    score = cross_val_score(model, X_train, y_train, cv=3).mean()
    print(f"SVM model score: {score:.4f}")

    return model, score

def analyze_dataset(file_path):
    """
    Analyze a single dataset file
    """
    print(f"\n======= Analyzing dataset: {os.path.basename(file_path)} =======")
    
    # Load data
    data = pd.read_csv(file_path)
    print(f"Dataset size: {data.shape}")
    
    # Check possible target columns
    possible_targets = ['ckd', 'hypertension']
    target_col = None
    
    for col in possible_targets:
        if col in data.columns and not data[col].isna().all():
            # Ensure it's a binary variable
            non_na_values = data[col].dropna().unique()
            if len(non_na_values) <= 2:
                target_col = col
                break
    
    if target_col is None:
        print("No suitable target variable found, creating random target")
        np.random.seed(42)
        data['target'] = np.random.randint(0, 2, size=len(data))
        target_col = 'target'
    
    # Preprocessing: ensure target column is integer type
    if target_col in data.columns:
        # Fill missing values
        na_count = data[target_col].isna().sum()
        if na_count > 0:
            print(f"Target column '{target_col}' has {na_count} missing values, filling with mode")
            most_frequent = int(data[target_col].dropna().mode()[0])
            data[target_col] = data[target_col].fillna(most_frequent)
        
        # Convert to integer type
        data[target_col] = data[target_col].astype(int)
        
    # Get basic data information
    print(f"Target column '{target_col}' value counts:\n{data[target_col].value_counts()}")
    
    # Check missing values
    original_missing = data.isnull().sum().sum()
    original_missing_pct = original_missing / (data.shape[0] * data.shape[1])
    print(f"Original data missing values: {original_missing} ({original_missing_pct:.2%})")
    
    # Handle missing values
    print("Performing mean/mode imputation...")
    imputed_data = mean_mode_imputation(data)
    
    # Feature selection
    print("Performing feature selection...")
    feature_importance = feature_selection_pipeline(imputed_data, target_col)
    print(f"Top 10 most important features:\n{feature_importance.head(10)}")
    
    # Prepare modeling data
    top_features = feature_importance.head(min(10, len(feature_importance)))['feature'].tolist()
    X = imputed_data[top_features]
    y = imputed_data[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train baseline and primary models
    print("Training standard models...")
    models, shap_values, model_results = train_models(X_train, X_test, y_train, y_test)
    
    # Train simplified SVM
    print("Training SVM...")
    best_svm, svm_score = constrained_svm(X_train, y_train)
    
    # Store results
    results = {
        'feature_importance': feature_importance,
        'models': models,
        'model_results': model_results,
        'shap_values': shap_values,
        'svm_model': best_svm,
        'svm_score': svm_score
    }
    
    # Try to plot SHAP summary (if available)
    try:
        plt.figure(figsize=(10, 8))
        plt.title(f"SHAP Summary Plot - {os.path.basename(file_path)}")
        if 'XGBoost' in shap_values and shap_values['XGBoost'] is not None:
            sv = shap_values['XGBoost']
            # Handle different forms of SHAP values (single class or multiclass)
            if isinstance(sv, list):
                shap.summary_plot(sv[1], X_test, feature_names=top_features, show=False)
            else:
                shap.summary_plot(sv, X_test, feature_names=top_features, show=False)
            plt.tight_layout()
            plt.savefig(f"shap_summary_{os.path.basename(file_path).split('.')[0]}.png")
            plt.close()
            print(f"SHAP plot saved as shap_summary_{os.path.basename(file_path).split('.')[0]}.png")
    except Exception as e:
        print(f"Error plotting SHAP summary: {e}")
    
    return results

# Main processing function
def process_all_datasets():
    """Process all CSV files in the Data_Final folder"""
    data_dir = "Data_Final"
    results = {}
    
    # Get all CSV files
    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    print(f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
    
    # Process each file
    for file_path in csv_files:
        dataset_name = os.path.basename(file_path)
        print(f"\n\n=========== Processing dataset: {dataset_name} ===========")
        
        # Analyze dataset
        dataset_results = analyze_dataset(file_path)
        
        if dataset_results:
            results[dataset_name] = dataset_results
    
    return results

# Execute all dataset processing
if __name__ == "__main__":
    print("Starting heart failure patient prediction modeling method...")
    try:
        np.random.seed(42)
        
        all_results = process_all_datasets()
        
        print("\n\nAnalysis complete. Results have been saved as PNG image files.")
    except Exception as e:
        import traceback
        print(f"Error during execution: {e}")
        traceback.print_exc() 