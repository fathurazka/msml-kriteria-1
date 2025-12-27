from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.over_sampling import SMOTE

def preprocess_data(data, target_column, save_path, file_path):
    numeric_features = data.select_dtypes(include=['float64']).columns.tolist()
    boolean_features = data.select_dtypes(include=['int64']).columns.drop(target_column).tolist()
    
    column_names = data.columns.drop(target_column)
    
    df_header = pd.DataFrame(columns=column_names)
    
    df_header.to_csv(file_path, index=False)
    print(f"Nama kolom telah disimpan ke : {file_path}")
    
    # Making sure target column is not in features
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in boolean_features:
        boolean_features.remove(target_column)
    
    # Pipeline for numeric features   
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('bool', 'passthrough', boolean_features)
        ]
    )
    
    # Splitting features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Splitting into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fitting the preprocessor on training data
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    # Saving the preprocessor
    dump(preprocessor, save_path)
    print(f"Preprocessor telah disimpan ke : {save_path}")
    
    # Saving Dataset After Preprocessing as CSV
    processed_train_df = pd.DataFrame(X_train, columns=column_names)
    processed_train_df[target_column] = y_train.reset_index(drop=True)
    
    # Convert boolean features back to int
    for col in boolean_features:
        processed_train_df[col] = processed_train_df[col].round().astype('int64')
    
    processed_train_df.to_csv("preprocessing/preprocessed_train_data.csv", index=False)
    
    processed_test_df = pd.DataFrame(X_test, columns=column_names)
    processed_test_df[target_column] = y_test.reset_index(drop=True)
    
    # Convert boolean features back to int
    for col in boolean_features:
        processed_test_df[col] = processed_test_df[col].round().astype('int64')
    
    processed_test_df.to_csv("preprocessing/preprocessed_test_data.csv", index=False)
    
    print("Dataset setelah preprocessing telah disimpan sebagai CSV.")
    
    # SMOTE Implementation
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    print("Class distribution setelah SMOTE:")
    print(y_train.value_counts())
    
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    data = pd.read_csv("dataset.csv")
    X_train, X_test, y_train, y_test = preprocess_data(data, target_column='fraud', save_path='preprocessing/preprocessor.joblib', file_path='preprocessing/data_columns.csv')