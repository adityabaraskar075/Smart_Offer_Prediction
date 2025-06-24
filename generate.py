import pandas as pd
import numpy as np
import random

# Generate synthetic dataset
np.random.seed(42)
n_samples = 10000

categories = ["Electronics", "Fashion", "Grocery", "Books", "Beauty", "Home"]
offer_types = ["Instant Discount", "Cashback", "No Cost EMI"]
card_types = ["HDFC Credit", "ICICI Debit", "SBI Credit", "Axis Credit", "Amazon Pay Card", "Other"]

data = {
    "cart_value": np.random.uniform(500, 10000, n_samples),
    "category": np.random.choice(categories, n_samples),
    "user_is_prime": np.random.randint(0, 2, n_samples),
    "payment_methods_available": [random.choices([0, 1], k=7) for _ in range(n_samples)],
    "card_type": np.random.choice(card_types, n_samples),
    "offer_type": np.random.choice(offer_types, n_samples),
    "min_cart_value": np.random.uniform(0, 5000, n_samples),
    "max_discount": np.random.uniform(50, 500, n_samples),
    "discount_percent": np.random.uniform(5, 30, n_samples),
    "requires_coupon": np.random.randint(0, 2, n_samples),
    "is_first_use": np.random.randint(0, 2, n_samples),
    "valid_on_app_only": np.random.randint(0, 2, n_samples),
    "times_used_in_month": np.random.randint(0, 5, n_samples),
}

# Calculate final discount (target)
df = pd.DataFrame(data)
df["final_discount"] = np.minimum(
    df["cart_value"] * df["discount_percent"] / 100,
    df["max_discount"]
)

# Add some randomness and prime member benefits
df["final_discount"] = df["final_discount"] * (0.9 + 0.2 * np.random.random(n_samples))
df["final_discount"] = df["final_discount"] * (1 + 0.1 * df["user_is_prime"])

# Save to CSV
df.to_csv("data/offers_dataset.csv", index=False)
print("Sample dataset generated at data/offers_dataset.csv")