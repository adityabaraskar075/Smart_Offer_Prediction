import pandas as pd
import numpy as np
from preprocess import load_and_preprocess
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

def build_model(input_shape):
    """Build the neural network model"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  # Linear activation for regression
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train():
    """Train the offer prediction model"""
    # Load and preprocess data
    X, y, preprocessor = load_and_preprocess('data/offers_dataset.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Build model
    model = build_model(X_train.shape[1])
    
    # Train model
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MAE: {test_mae:.2f}")
    
    # Save artifacts
    os.makedirs('src/saved_models', exist_ok=True)
    model.save('src/saved_models/offer_model.keras')
    preprocessor.save_encoders('src/saved_models/feature_columns.pkl')
    
    print("Model training complete and artifacts saved.")

if __name__ == '__main__':
    train()