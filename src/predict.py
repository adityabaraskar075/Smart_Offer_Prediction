import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from src.preprocess import preprocess_input, OfferPreprocessor

class OfferPredictor:
    def __init__(self, model_path='src/saved_models/offer_model.keras', 
                 preprocessor_path='src/saved_models/feature_columns.pkl'):
        """Initialize the predictor with saved model and preprocessor"""
        try:
            self.model = load_model(model_path)
            self.preprocessor = OfferPreprocessor.load_encoders(preprocessor_path)
        except Exception as e:
            raise ValueError(f"Failed to load model artifacts: {str(e)}")
    
    def predict(self, input_data):
        """Predict discount for input data"""
        try:
            # Preprocess input
            X = preprocess_input(input_data, self.preprocessor)
            
            # Predict
            prediction = self.model.predict(X)
            
            # Return as float (single prediction) or array (multiple predictions)
            if isinstance(input_data, dict) or len(prediction) == 1:
                return float(prediction[0][0])
            return prediction.flatten()
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")

def predict_from_input(input_data):
    """Convenience function for making predictions"""
    predictor = OfferPredictor()
    return predictor.predict(input_data)

if __name__ == '__main__':
    # Example usage
    sample_input = {
        "cart_value": 1899.0,
        "category": "Electronics",
        "user_is_prime": 1,
        "payment_methods_available": [1, 1, 1, 0, 1, 1, 0],
        "card_type": "HDFC Credit",
        "offer_type": "Instant Discount",
        "min_cart_value": 1000.0,
        "max_discount": 150.0,
        "discount_percent": 10.0,
        "requires_coupon": 0,
        "is_first_use": 1,
        "valid_on_app_only": 0,
        "times_used_in_month": 1
    }
    
    prediction = predict_from_input(sample_input)
    print(f"Predicted discount: {prediction:.2f}")