import csv


# ──────────────────────────────────────────────
# 1. 讀取訓練後的 θ0, θ1
#    若 thetas.csv 不存在，依題目規定 θ0 = θ1 = 0
# ──────────────────────────────────────────────
def load_thetas(filepath='thetas.csv'):
    try:
        with open(filepath, newline='') as f:
            reader = csv.DictReader(f)
            row = next(reader)
            return float(row['theta0']), float(row['theta1'])
    except (FileNotFoundError, StopIteration, KeyError):
        print("(thetas.csv not found — using theta0=0, theta1=0)")
        return 0.0, 0.0


# ──────────────────────────────────────────────
# 2. 預測公式 (直接來自 Subject)
#    estimatePrice(mileage) = θ0 + θ1 × mileage
# ──────────────────────────────────────────────
def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage


# ──────────────────────────────────────────────
# 3. 主流程：提示輸入 → 預測 → 輸出
# ──────────────────────────────────────────────
def main():
    theta0, theta1 = load_thetas()

    try:
        mileage = float(input("Enter mileage (km): "))
    except ValueError:
        print("Error: Please enter a valid number.")
        return

    if mileage < 0:
        print("Error: Mileage cannot be negative.")
        return

    price = estimate_price(mileage, theta0, theta1)
    print(f"Estimated price: {price:.2f} €")


if __name__ == '__main__':
    main()
