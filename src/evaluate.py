import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from utils import load_model

def load_test_data():
    """Load the processed test data."""
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    return X_test, y_test

def evaluate_model(model, X_test, y_test, name):
    """Evaluate a model and return metrics."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    return {
        'Model': name,
        'RMSE': round(rmse, 2),
        'R2 Score': round(r2, 4)
    }

def main():
    # 1. Load data
    X_test, y_test = load_test_data()
    
    # 2. Load models
    lr_model = load_model('linear_regression.joblib')
    rf_model = load_model('random_forest.joblib')
    
    # 3. Evaluate models
    results = []
    results.append(evaluate_model(lr_model, X_test, y_test, 'Linear Regression'))
    results.append(evaluate_model(rf_model, X_test, y_test, 'Random Forest'))
    
    # 4. Create comparison table
    results_df = pd.DataFrame(results)
    print("\nModel Comparison Table:")
    print("=" * 40)
    print(results_df.to_string(index=False))
    print("=" * 40)
    
    # 5. Interpretation
    best_model = results_df.loc[results_df['R2 Score'].idxmax()]['Model']
    print(f"\nInterpretation:")
    print(f"- The {best_model} performed best on the test set.")
    print("- R2 Score indicates what % of variance in price is explained by features.")
    print("- RMSE shows the average error in prediction (in the same units as SalePrice).")

if __name__ == "__main__":
    main()
    print(f"- R2 Score measures how much variance is explained by the model (higher is better).")
    print(f"- RMSE measures the average error in prediction (lower is better).")

if __name__ == "__main__":
    main()
