# 🧠 Smart Offer Predictor

A Machine Learning-based system that predicts the **best available payment offer** for users during checkout. This tool is designed to optimize savings using contextual cart and user data—mimicking how Amazon recommends card offers like “Flat ₹150 off using HDFC Credit Card” or “10% off on UPI”.

---

## 📌 Table of Contents

- [🎯 Objective](#-objective)
- [📁 Project Structure](#-project-structure)
- [🛠️ How It Works](#️-how-it-works)
- [📊 Input Features](#-input-features)
- [🚀 Getting Started](#-getting-started)
- [✅ Running the Example](#-running-the-example)
- [📦 Model Artifacts](#-model-artifacts)
- [🔮 Sample Output](#-sample-output)
- [🔧 Future Enhancements](#-future-enhancements)
- [👨‍💻 Team](#-team)

---

## 🎯 Objective

This ML tool helps users make smarter checkout decisions by:
- Predicting the **most beneficial offer** dynamically.
- Saving users **maximum possible amount** using contextual data like cart value, category, available payment methods, coupons, etc.

---

## 🛠️ How It Works

1. You input your cart and user context (like Prime status, cart amount, available payment methods).
2. The model preprocesses these features.
3. It predicts the **best offer** out of all possibilities.
4. Output includes:
   - Best payment method + offer combo
   - Maximum discount achievable
   - Whether a coupon is required

---

## 📊 Input Features

| Feature Name             | Description                                                             |
|--------------------------|-------------------------------------------------------------------------|
| `cart_value`             | Total value of the shopping cart                                        |
| `category`               | Product category (e.g., Electronics, Fashion, etc.)                     |
| `user_is_prime`          | 1 if user is a Prime member, else 0                                     |
| `payment_methods_available` | Multi-hot list: `[Card, UPI, NetBanking, ...]`                        |
| `card_type`              | Payment card used (e.g., "HDFC Credit", "SBI Debit")                    |
| `offer_type`             | Type of offer: "Instant Discount", "Cashback", "Coupon-based"           |
| `min_cart_value`         | Minimum cart value for offer applicability                              |
| `max_discount`           | Max discount cap (e.g., ₹150)                                           |
| `discount_percent`       | Percentage discount offered                                             |
| `requires_coupon`        | 1 if user must apply a coupon                                           |
| `is_first_use`           | 1 if user is using offer/payment method for the first time              |
| `valid_on_app_only`      | 1 if offer is valid only on app                                         |
| `times_used_in_month`    | How many times the user has used this offer/payment method this month   |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/smart-offer-predictor.git
cd smart-offer-predictor

python generate.py

python example.py
```
