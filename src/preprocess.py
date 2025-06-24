import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import ast

# Constants for consistent preprocessing
CATEGORIES = ["Electronics", "Fashion", "Grocery", "Books", "Beauty", "Home"]
OFFER_TYPES = ["Instant Discount", "Cashback", "No Cost EMI"]
PAYMENT_METHOD_COLS = [f'pm_{i}' for i in range(7)]

class OfferPreprocessor:
    def __init__(self):
        self.category_encoder = OneHotEncoder(categories=[CATEGORIES], sparse_output=False)
        self.offer_type_encoder = OneHotEncoder(categories=[OFFER_TYPES], sparse_output=False)
        self.card_encoder = LabelEncoder()
        self.feature_columns = None
        
    def fit(self, df):
        """Fit encoders on the training data"""
        # Fit encoders
        self.category_encoder.fit(df[['category']])
        self.offer_type_encoder.fit(df[['offer_type']])
        self.card_encoder.fit(df['card_type'])
        
        # Store feature columns for reference
        self._create_feature_columns()
        
    def transform(self, df):
        """Transform input data into model-ready format"""
        # Make a copy to avoid modifying original dataframe
        df = df.copy()
        
        # Convert payment methods to proper list format if needed
        df['payment_methods_available'] = df['payment_methods_available'].apply(
            self._convert_payment_methods
        )
        
        # Payment methods expansion
        try:
            payment_methods = pd.DataFrame(
                df['payment_methods_available'].tolist(),
                columns=PAYMENT_METHOD_COLS
            )
        except ValueError as e:
            raise ValueError(
                f"Payment methods format error. Expected list of 7 binaries. Error: {str(e)}"
            )

        # One-hot encoding
        category_encoded = self.category_encoder.transform(df[['category']])
        offer_type_encoded = self.offer_type_encoder.transform(df[['offer_type']])
        
        # Label encoding for card type
        card_encoded = self.card_encoder.transform(df['card_type'])
        
        # Numeric features
        numeric_features = df[[
            'cart_value', 'user_is_prime', 'min_cart_value', 
            'max_discount', 'discount_percent', 'requires_coupon',
            'is_first_use', 'valid_on_app_only', 'times_used_in_month'
        ]].copy()
        
        # Combine all features
        features = pd.concat([
            pd.DataFrame(category_encoded, columns=[f'cat_{c}' for c in CATEGORIES]),
            pd.DataFrame(offer_type_encoded, columns=[f'offer_{o}' for o in OFFER_TYPES]),
            pd.DataFrame({'card_type': card_encoded}),
            payment_methods,
            numeric_features
        ], axis=1)
        
        # Ensure all expected columns are present
        features = self._ensure_columns(features)
        
        return features
    
    def _convert_payment_methods(self, x):
        """Convert payment methods to list format"""
        if isinstance(x, str):
            try:
                # Handle both "[1,0,1,...]" and "1,0,1,..." formats
                if x.startswith('[') and x.endswith(']'):
                    return list(map(int, ast.literal_eval(x)))
                else:
                    return list(map(int, x.split(',')))
            except:
                return [0] * 7
        elif isinstance(x, list):
            return x
        else:
            return [0] * 7
    
    def _create_feature_columns(self):
        """Create the complete list of feature columns in correct order"""
        category_cols = [f'cat_{c}' for c in CATEGORIES]
        offer_cols = [f'offer_{o}' for o in OFFER_TYPES]
        card_col = ['card_type']
        payment_cols = PAYMENT_METHOD_COLS
        numeric_cols = [
            'cart_value', 'user_is_prime', 'min_cart_value', 
            'max_discount', 'discount_percent', 'requires_coupon',
            'is_first_use', 'valid_on_app_only', 'times_used_in_month'
        ]
        
        self.feature_columns = category_cols + offer_cols + card_col + payment_cols + numeric_cols
    
    def _ensure_columns(self, features):
        """Ensure all expected columns are present, filling missing with 0"""
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0
        return features[self.feature_columns]
    
    def save_encoders(self, path):
        """Save encoders and feature columns to disk"""
        joblib.dump({
            'category_encoder': self.category_encoder,
            'offer_type_encoder': self.offer_type_encoder,
            'card_encoder': self.card_encoder,
            'feature_columns': self.feature_columns
        }, path)
    
    @classmethod
    def load_encoders(cls, path):
        """Load encoders and feature columns from disk"""
        data = joblib.load(path)
        preprocessor = cls()
        preprocessor.category_encoder = data['category_encoder']
        preprocessor.offer_type_encoder = data['offer_type_encoder']
        preprocessor.card_encoder = data['card_encoder']
        preprocessor.feature_columns = data['feature_columns']
        return preprocessor

def load_and_preprocess(file_path):
    """Load and preprocess data for training"""
    df = pd.read_csv(file_path)
    
    # Convert payment methods to proper format
    df['payment_methods_available'] = df['payment_methods_available'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x
    )
    
    preprocessor = OfferPreprocessor()
    preprocessor.fit(df)
    
    X = preprocessor.transform(df)
    y = df['final_discount'].values
    
    return X, y, preprocessor

def preprocess_input(input_data, preprocessor=None):
    """Preprocess single input for prediction"""
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()
    
    if preprocessor is None:
        preprocessor = OfferPreprocessor()
        # In a real scenario, we should load a pre-trained preprocessor
        # For demo purposes, we'll create a new one
        preprocessor.fit(df)  # This is not ideal for production
    
    return preprocessor.transform(df)