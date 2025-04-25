import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os
# For GPLVM imputation
import torch
from torch.nn import Parameter
import pyro
import pyro.distributions as dist
import pyro.contrib.gp as gp
import pyro.distributions.transforms as transforms
# Fix: pyro.stats doesn't exist, using torch instead
import torch.utils.data as torch_utils

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

def regression_imputation(data):
    """
    Impute missing values using Random Forest regression
    """
    # Use Random Forest for imputation
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(),
        max_iter=10,
        random_state=42
    )
    
    # Impute only numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    data_imputed = data.copy()
    
    # Handle numeric columns with Random Forest regression
    if len(numeric_cols) > 0:
        data_imputed[numeric_cols] = imputer.fit_transform(data[numeric_cols])
    
    # Handle categorical columns with most frequent value (mode)
    if len(categorical_cols) > 0:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        data_imputed[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])
    
    return data_imputed

def gplvm_imputation(data, latent_dim=2, num_inducing=32, num_steps=4000):
    """
    GPLVM-based imputation for missing values
    """
    # Make a copy of the data for imputation
    imputed_data = data.copy()
    
    # First, handle categorical columns with mode imputation
    categorical_cols = data.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        imputed_data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])
    
    # Now focus on numeric columns for GPLVM
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        numeric_data = data[numeric_cols]
        
        # Check if there are missing values in numeric columns
        if numeric_data.isnull().any().any():
            try:
                # Initialize pyro
                pyro.clear_param_store()
                
                # Temporarily fill NaNs with column means for initialization
                temp_data = numeric_data.fillna(numeric_data.mean())
                
                # Convert data to torch tensor
                data_tensor = torch.tensor(temp_data.values, dtype=torch.float32)
                y = data_tensor.t()
                
                # Create prior mean for latent variables
                X_prior_mean = torch.zeros(y.size(1), latent_dim)
                
                # Define RBF kernel
                kernel = gp.kernels.RBF(input_dim=latent_dim, lengthscale=torch.ones(latent_dim))
                
                # Initialize latent variables
                X = Parameter(X_prior_mean.clone())
                
                # Fix: Replace stats.resample with manual random sampling
                # Initialize inducing points (randomly sample from the data)
                n_samples = min(num_inducing, X_prior_mean.size(0))
                indices = torch.randperm(X_prior_mean.size(0))[:n_samples]
                Xu = X_prior_mean[indices].clone()
                
                # Create sparse GP model
                gplvm = gp.models.SparseGPRegression(
                    X, y, kernel, Xu,
                    noise=torch.tensor(0.01),
                    jitter=1e-5
                )
                
                # Set up prior for X
                gplvm.X = pyro.nn.PyroSample(
                    dist.Normal(X_prior_mean, 0.1).to_event()
                )
                
                # Set up guide
                gplvm.autoguide("X", dist.Normal)
                
                # Train the model
                losses = gp.util.train(gplvm, num_steps=num_steps)
                
                # Get imputed values and uncertainty
                gplvm.mode = "guide"
                X = gplvm.X_loc.detach().numpy()
                
                # Use the learned latent space to impute missing values
                for i, col in enumerate(numeric_cols):
                    if numeric_data[col].isnull().any():
                        # Get predictions for this column
                        pred_mean, _ = gplvm.forward(X)
                        # Extract the relevant column of predictions
                        pred_values = pred_mean[i].detach().numpy()
                        # Get indices of missing values
                        missing_idx = numeric_data[col].isnull()
                        # Fill in missing values in the original data
                        imputed_data.loc[missing_idx, col] = pred_values[missing_idx]
                
                print(f"GPLVM imputation completed with {num_steps} training steps")
            
            except Exception as e:
                print(f"Error in GPLVM imputation: {e}")
                print("Falling back to mean imputation for numeric columns")
                # Fallback to mean imputation if GPLVM fails
                numeric_imputer = SimpleImputer(strategy='mean')
                imputed_data[numeric_cols] = numeric_imputer.fit_transform(numeric_data)
    
    return imputed_data

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
    
    # Compare different imputation methods
    print("\n--- Comparing different imputation methods ---")
    
    # Method 1: Mean/Mode imputation
    print("\nPerforming mean/mode imputation...")
    mean_mode_imputed_data = mean_mode_imputation(data)
    
    # Method 2: Regression imputation
    print("\nPerforming regression imputation...")
    regression_imputed_data = regression_imputation(data)
    
    # Method 3: GPLVM imputation
    print("\nPerforming GPLVM imputation...")
    gplvm_imputed_data = gplvm_imputation(data, num_steps=1000)  # Reduced steps for faster execution
    
    # Store results for different imputation methods
    imputation_results = {}
    
    # Analyze with mean/mode imputation
    print("\n--- Results with Mean/Mode Imputation ---")
    # Feature selection
    print("Performing feature selection...")
    mean_mode_feature_importance = feature_selection_pipeline(mean_mode_imputed_data, target_col)
    print(f"Top 10 most important features:\n{mean_mode_feature_importance.head(10)}")
    
    # Prepare modeling data
    top_features = mean_mode_feature_importance.head(min(10, len(mean_mode_feature_importance)))['feature'].tolist()
    X = mean_mode_imputed_data[top_features]
    y = mean_mode_imputed_data[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train baseline and primary models
    print("Training standard models...")
    mean_mode_models, mean_mode_shap_values, mean_mode_model_results = train_models(X_train, X_test, y_train, y_test)
    
    # Train simplified SVM
    print("Training SVM...")
    mean_mode_svm, mean_mode_svm_score = constrained_svm(X_train, y_train)
    
    imputation_results['mean_mode'] = {
        'feature_importance': mean_mode_feature_importance,
        'models': mean_mode_models,
        'model_results': mean_mode_model_results,
        'shap_values': mean_mode_shap_values,
        'svm_model': mean_mode_svm,
        'svm_score': mean_mode_svm_score
    }
    
    # Analyze with regression imputation
    print("\n--- Results with Regression Imputation ---")
    # Feature selection
    print("Performing feature selection...")
    regression_feature_importance = feature_selection_pipeline(regression_imputed_data, target_col)
    print(f"Top 10 most important features:\n{regression_feature_importance.head(10)}")
    
    # Prepare modeling data
    top_features = regression_feature_importance.head(min(10, len(regression_feature_importance)))['feature'].tolist()
    X = regression_imputed_data[top_features]
    y = regression_imputed_data[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train baseline and primary models
    print("Training standard models...")
    regression_models, regression_shap_values, regression_model_results = train_models(X_train, X_test, y_train, y_test)
    
    # Train simplified SVM
    print("Training SVM...")
    regression_svm, regression_svm_score = constrained_svm(X_train, y_train)
    
    imputation_results['regression'] = {
        'feature_importance': regression_feature_importance,
        'models': regression_models,
        'model_results': regression_model_results,
        'shap_values': regression_shap_values,
        'svm_model': regression_svm,
        'svm_score': regression_svm_score
    }
    
    # Analyze with GPLVM imputation
    print("\n--- Results with GPLVM Imputation ---")
    # Feature selection
    print("Performing feature selection...")
    gplvm_feature_importance = feature_selection_pipeline(gplvm_imputed_data, target_col)
    print(f"Top 10 most important features:\n{gplvm_feature_importance.head(10)}")
    
    # Prepare modeling data
    top_features = gplvm_feature_importance.head(min(10, len(gplvm_feature_importance)))['feature'].tolist()
    X = gplvm_imputed_data[top_features]
    y = gplvm_imputed_data[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train baseline and primary models
    print("Training standard models...")
    gplvm_models, gplvm_shap_values, gplvm_model_results = train_models(X_train, X_test, y_train, y_test)
    
    # Train simplified SVM
    print("Training SVM...")
    gplvm_svm, gplvm_svm_score = constrained_svm(X_train, y_train)
    
    imputation_results['gplvm'] = {
        'feature_importance': gplvm_feature_importance,
        'models': gplvm_models,
        'model_results': gplvm_model_results,
        'shap_values': gplvm_shap_values,
        'svm_model': gplvm_svm,
        'svm_score': gplvm_svm_score
    }
    
    # Compare imputation methods
    print("\n--- Imputation Method Comparison ---")
    comparison = {
        'Mean/Mode': imputation_results['mean_mode']['model_results'],
        'Regression': imputation_results['regression']['model_results'],
        'GPLVM': imputation_results['gplvm']['model_results']
    }
    
    # Print comparison table
    for model_name in ['Logistic Regression', 'Random Forest', 'XGBoost']:
        print(f"{model_name} accuracy comparison:")
        for method, results in comparison.items():
            if results and model_name in results:
                print(f"  {method}: {results[model_name]:.4f}")
            else:
                print(f"  {method}: N/A")
    
    # SVM comparison
    print("SVM model score comparison:")
    print(f"  Mean/Mode: {imputation_results['mean_mode']['svm_score']:.4f}")
    print(f"  Regression: {imputation_results['regression']['svm_score']:.4f}")
    print(f"  GPLVM: {imputation_results['gplvm']['svm_score']:.4f}")
    
    # Try to plot SHAP summary for the best method (using XGBoost)
    try:
        # Find which method had the best XGBoost performance
        best_method = max(
            ['mean_mode', 'regression', 'gplvm'],
            key=lambda m: imputation_results[m]['model_results'].get('XGBoost', 0)
        )
        
        plt.figure(figsize=(10, 8))
        plt.title(f"SHAP Summary Plot - {os.path.basename(file_path)} (Best Method: {best_method})")
        
        if 'XGBoost' in imputation_results[best_method]['shap_values'] and imputation_results[best_method]['shap_values']['XGBoost'] is not None:
            sv = imputation_results[best_method]['shap_values']['XGBoost']
            
            # Get feature names for the best method
            top_features = imputation_results[best_method]['feature_importance'].head(10)['feature'].tolist()
            
            # Handle different forms of SHAP values (single class or multiclass)
            if isinstance(sv, list):
                shap.summary_plot(sv[1], X_test, feature_names=top_features, show=False)
            else:
                shap.summary_plot(sv, X_test, feature_names=top_features, show=False)
                
            plt.tight_layout()
            plt.savefig(f"shap_summary_{os.path.basename(file_path).split('.')[0]}_{best_method}.png")
            plt.close()
            print(f"SHAP plot saved as shap_summary_{os.path.basename(file_path).split('.')[0]}_{best_method}.png")
    except Exception as e:
        print(f"Error plotting SHAP summary: {e}")
    
    return imputation_results

# Main processing function
def process_all_datasets():
    """Process all CSV files in the Data_Final folder"""
    # Use local data directory
    data_dir = "Data_Final"
    print(f"Using data directory: {data_dir}")
    
    # Create results structure
    results = {}
    summary = {
        'mean_mode': {'accuracy': {}, 'svm_score': {}},
        'regression': {'accuracy': {}, 'svm_score': {}},
        'gplvm': {'accuracy': {}, 'svm_score': {}}
    }
    
    # Get all CSV files
    try:
        csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
        print(f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
    except Exception as e:
        print(f"Error accessing data directory: {e}")
        print("Checking for individual CSV files in the current directory...")
        csv_files = [f for f in os.listdir() if f.endswith('.csv')]
        if csv_files:
            print(f"Found {len(csv_files)} CSV files in current directory.")
        else:
            print("No CSV files found.")
            return results
    
    # Process each file
    for file_path in csv_files:
        if not os.path.exists(file_path):
            file_path = os.path.basename(file_path)  # Try just the filename
        
        if os.path.exists(file_path):
            dataset_name = os.path.basename(file_path)
            print(f"\n\n=========== Processing dataset: {dataset_name} ===========")
            
            try:
                # Analyze dataset
                dataset_results = analyze_dataset(file_path)
                
                if dataset_results:
                    results[dataset_name] = dataset_results
                    
                    # Collect accuracy metrics for summary
                    for method in ['mean_mode', 'regression', 'gplvm']:
                        if method in dataset_results:
                            # Model accuracies
                            for model, score in dataset_results[method]['model_results'].items():
                                if model not in summary[method]['accuracy']:
                                    summary[method]['accuracy'][model] = []
                                summary[method]['accuracy'][model].append(score)
                            
                            # SVM scores
                            if 'svm_score' in dataset_results[method]:
                                if dataset_name not in summary[method]['svm_score']:
                                    summary[method]['svm_score'][dataset_name] = []
                                summary[method]['svm_score'][dataset_name] = dataset_results[method]['svm_score']
            except Exception as e:
                print(f"Error processing dataset {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Warning: File not found: {file_path}")
    
    # Determine the best imputation method based on average model performance
    if results:
        print("\n\n=========== SUMMARY ===========")
        print("\nAverage model performance across all datasets:")
        
        best_method = None
        best_avg_score = -1
        
        for method in ['mean_mode', 'regression', 'gplvm']:
            method_avg = {}
            print(f"\n{method.upper()} Imputation Method:")
            
            # Calculate average model accuracies
            for model, scores in summary[method]['accuracy'].items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    method_avg[model] = avg_score
                    print(f"  {model} average accuracy: {avg_score:.4f} (across {len(scores)} datasets)")
            
            # Average SVM score
            svm_scores = list(summary[method]['svm_score'].values())
            if svm_scores:
                avg_svm = sum(svm_scores) / len(svm_scores)
                method_avg['SVM'] = avg_svm
                print(f"  SVM average score: {avg_svm:.4f} (across {len(svm_scores)} datasets)")
            
            # Calculate overall method average 
            if method_avg:
                method_overall_avg = sum(method_avg.values()) / len(method_avg)
                print(f"  OVERALL AVERAGE: {method_overall_avg:.4f}")
                
                # Track best method
                if method_overall_avg > best_avg_score:
                    best_avg_score = method_overall_avg
                    best_method = method
        
        if best_method:
            print(f"\nBEST IMPUTATION METHOD: {best_method.upper()} with average score {best_avg_score:.4f}")
    
    # Create results directory and save summary information
    try:
        # Create results directory
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save summary to file
        with open(os.path.join(results_dir, "imputation_summary.txt"), "w") as f:
            f.write("IMPUTATION METHODS COMPARISON SUMMARY\n")
            f.write("====================================\n\n")
            
            for method in ['mean_mode', 'regression', 'gplvm']:
                f.write(f"{method.upper()} Imputation Method:\n")
                
                # Write model accuracies
                for model, scores in summary[method]['accuracy'].items():
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        f.write(f"  {model} average accuracy: {avg_score:.4f} (across {len(scores)} datasets)\n")
                
                # Write SVM score
                svm_scores = list(summary[method]['svm_score'].values())
                if svm_scores:
                    avg_svm = sum(svm_scores) / len(svm_scores)
                    f.write(f"  SVM average score: {avg_svm:.4f} (across {len(svm_scores)} datasets)\n")
                
                f.write("\n")
            
            if best_method:
                f.write(f"BEST IMPUTATION METHOD: {best_method.upper()} with average score {best_avg_score:.4f}\n")
        
        print(f"\nSummary has been saved to {os.path.join(results_dir, 'imputation_summary.txt')}")
    except Exception as e:
        print(f"Error saving summary: {e}")
    
    return results

# Execute all dataset processing
if __name__ == "__main__":
    print("Starting heart failure patient prediction modeling method...")
    try:
        # Initialize PyTorch and Pyro
        pyro.set_rng_seed(42)
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Run analysis
        all_results = process_all_datasets()
        
        print("\n\nAnalysis complete. Results have been saved to the results directory.")
    except Exception as e:
        import traceback
        print(f"Error during execution: {e}")
        traceback.print_exc() 