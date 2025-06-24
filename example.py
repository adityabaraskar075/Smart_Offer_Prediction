from src.predict import predict_from_input

def run_example():
    """Run example prediction"""
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
    
    print("Running example prediction...")
    print("\nInput features:")
    for key, value in sample_input.items():
        print(f"{key}: {value}")
    
    prediction = predict_from_input(sample_input)
    print(f"\nPredicted final discount: ₹{prediction:.2f}")

if __name__ == '__main__':
    run_example()