import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder


def train_model_sklearn(data: pd.DataFrame, target: str):
    X = data.drop(columns=[target])
    y = data[target]

    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        le = None  # No encoding needed
    
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

    numeric_columns = X.select_dtypes(include=[np.number])
    if numeric_columns.empty:
        print("Warning: No numeric columns found in X. Please check your data preprocessing.")
        return None, None  # Retorna None para ambos os valores
    
    print(f"Shape of X after preprocessing: {X.shape}")

    if 'size' not in X.columns:
        print("Column 'size' not found. Renaming column 0 to 'size'.")
        X.columns = ['size']  # Renomear a coluna numerada para 'size'

    print("First 5 values in the 'size' column before conversion:")
    print(X['size'].head())

    X['size'] = pd.to_numeric(X['size'], errors='coerce')  # Tentar converter explicitamente a coluna 'size'
    
    print("First 5 values in the 'size' column after conversion:")
    print(X['size'].head())

    print("Number of NaN values after conversion:", X['size'].isna().sum())

    numeric_columns = X.select_dtypes(include=[np.number])
    print(f"Numeric Columns after conversion: {numeric_columns.columns.tolist()}")
    
    if numeric_columns.empty:
        print("Warning: No numeric columns found in X. Please check your data preprocessing.")
        return None, None  # Retorna None se não houver colunas numéricas

    print(f"Shape of X after conversion to numeric: {X.shape}")

    candidates = {
        'gradient_boosting': {
            'pipeline': Pipeline([('scaler', StandardScaler()), 
                                  ('regressor', GradientBoostingRegressor(random_state=123))]),
            'params': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__learning_rate': [0.01, 0.1, 0.2],
                'regressor__max_depth': [3, 5, 7]
            }
        },
        'ada_boost': {
            'pipeline': Pipeline([('regressor', AdaBoostRegressor(random_state=123))]),
            'params': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__learning_rate': [0.01, 0.1, 0.2]
            }
        },
        'xgboost': {
            'pipeline': Pipeline([('regressor', xgb.XGBRegressor(random_state=123, objective='reg:squarederror'))]),
            'params': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__learning_rate': [0.01, 0.1, 0.2],
                'regressor__max_depth': [3, 5, 7]
            }
        },
        'lightgbm': {
            'pipeline': Pipeline([('regressor', LGBMRegressor(random_state=123))]),
            'params': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__learning_rate': [0.01, 0.1, 0.2],
                'regressor__max_depth': [-1, 5, 10]
            }
        }
    }

    best_score = -np.inf
    best_model = None
    best_model_name = None

    for name, candidate in candidates.items():
        print(f"Training model: {name}")
        grid = GridSearchCV(candidate['pipeline'], candidate['params'], cv=5, scoring='r2', n_jobs=-1)
        
        print("Shape of X before fitting the model:", X.shape)
        print("Shape of y before fitting the model:", y.shape)
        
        try:
            grid.fit(X, y)
            print(f"Model: {name} | Best CV R² Score: {grid.best_score_:.4f}")
            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                best_model = grid.best_estimator_
                best_model_name = name
        except Exception as e:
            print(f"Error with model {name}: {e}")

    print(f"\nSelected Best Model: {best_model_name} with CV R² Score: {best_score:.4f}")
    
    return best_model, le  

