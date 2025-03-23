import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder


def preprocess_data(data: pd.DataFrame, target: str):
    X = data.drop(columns=[target])
    
    if 'Unnamed: 0' in X.columns:
        X = X.drop(columns=['Unnamed: 0'])
        
    data['price'] = data['price'].apply(lambda x: -1 if x == 'Price on request' else x)
    data['price'] = data['price'].apply(lambda x: int(x.replace('$', '').replace(',', '').replace("'", '')) if isinstance(x, str) else -1)
    
    if 'size' in X.columns:
        X['size'] = X['size'].apply(lambda x: str(x).replace(' mm', '') if isinstance(x, str) else x)
        X['size'] = pd.to_numeric(X['size'], errors='coerce')
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_pipeline = Pipeline(steps=[ 
        ('imputer', SimpleImputer(strategy='mean')), 
        ('scaler', StandardScaler()) 
    ])

    categorical_pipeline = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='most_frequent')),
      ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
      # ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[ 
        ('num', numeric_pipeline, numeric_features), 
        ('cat', categorical_pipeline, categorical_features) 
    ])
    
    X_preprocessed = preprocessor.fit_transform(X)

    # Gerar nomes das colunas transformadas
    all_columns = numeric_features + categorical_features  # Ajustar isso conforme a transformação feita
    
    # Converter para DataFrame para visualização
    X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=all_columns)

    y = data[target]

    return X_preprocessed_df, y, preprocessor
