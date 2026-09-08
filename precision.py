#!/usr/bin/env python3
import csv
import sys


def load_data(filepath='data.csv'):
    """Load data.csv"""

    kms, prices = [], []
    try:
        with open(filepath, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                kms.append(float(row['km']))
                prices.append(float(row['price']))
        return kms, prices
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        sys.exit(1)


def load_thetas(filepath='thetas.csv'):
    """Load thetas.csv"""

    try:
        with open(filepath, newline='') as f:
            reader = csv.DictReader(f)
            row = next(reader)
            return float(row['theta0']), float(row['theta1'])
    except (FileNotFoundError, StopIteration, KeyError):
        print(
            "Error: thetas.csv not found or invalid. "
            "Please run train.py first."
        )
        sys.exit(1)


def evaluate_precision(kms, prices, theta0, theta1):
    """
    Evaluate the precision of the trained model using R², MAE, MSE, and RMSE.
    """

    m = len(prices)
    mean_price = sum(prices) / m

    predictions = [theta0 + theta1 * km for km in kms]
    residuals = [prices[i] - predictions[i] for i in range(m)]

    # 1. R² Score (Coefficient of Determination)
    ss_res = sum(r ** 2 for r in residuals)
    ss_tot = sum((p - mean_price) ** 2 for p in prices)
    r2 = 1 - (ss_res / ss_tot)

    # 2. Mean Absolute Error (MAE)
    mae = sum(abs(r) for r in residuals) / m

    # 3. Mean Squared Error (MSE) & Root MSE (RMSE)
    mse = ss_res / m
    rmse = mse ** 0.5

    return r2, mae, mse, rmse


def main():
    """Main function to evaluate the precision of the trained model."""

    kms, prices = load_data('data.csv')
    theta0, theta1 = load_thetas('thetas.csv')

    r2, mae, mse, rmse = evaluate_precision(kms, prices, theta0, theta1)

    print("════════════════════════════════════════════════")
    print("             Model Precision Report            ")
    print("════════════════════════════════════════════════")
    print(f"  Trained Parameters : θ0 = {theta0:.4f}, θ1 = {theta1:.8f}")
    print(
        f"  R² Score (Accuracy): {r2:.4f} ({r2 * 100:.2f}% "
        "of variance explained)\n"
        f"  Mean Absolute Error: {mae:.2f} € (average deviation)\n"
        f"  Root Mean Sq. Error: {rmse:.2f} €\n"
        f"  Mean Squared Error : {mse:.2f}\n"
        "════════════════════════════════════════════════"
    )


if __name__ == '__main__':
    main()
