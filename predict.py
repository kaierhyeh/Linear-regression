#!/usr/bin/env python3
import csv


# ──────────────────────────────────────────────
# 1. Read trained θ0, θ1 stored in thetas.csv.
#    If the file doesn't exist, default to θ0 = θ1 = 0.
# ──────────────────────────────────────────────
def load_thetas(filepath='thetas.csv'):
    """
    Read trained θ0, θ1 stored in thetas.csv.
    If the file doesn't exist, default to θ0 = θ1 = 0.
    """

    try:
        with open(filepath, newline='') as f:
            reader = csv.DictReader(f)
            row = next(reader)
            return float(row['theta0']), float(row['theta1'])
    except (FileNotFoundError, StopIteration, KeyError):
        print("(thetas.csv not found, using theta0=0, theta1=0.)")
        return 0.0, 0.0
# with:
#     離開 with 區域時自動關閉檔案。
# newline='':
#     告訴 open() 不要自行轉換換行符號，讓 csv 模組自己處理換行。
# reader = csv.DictReader(f) 建立時會自動把 CSV 的第一列當成欄位名稱
# （fieldnames / keys），後面的列才會被轉成 dictionary。同時回傳的 reader會是個類似
# iterator 的物件。
# → 只讀取第一行時可用：
#     rows = list(reader)
#     row = rows[0]
#   或
#     for row in reader:
#         print(row)
#         break
#     但 next() 最清楚、最節省資源。


# ──────────────────────────────────────────────
# 2. 預測公式 (來自 Subject)
#    estimatePrice(mileage) = θ0 + θ1 × mileage
# ──────────────────────────────────────────────
def estimate_price(mileage, theta0, theta1):
    """Linear Regression formula"""
    return theta0 + theta1 * mileage


# ──────────────────────────────────────────────
# 3. 主流程：提示輸入 → 預測 → 輸出
# ──────────────────────────────────────────────
def main():
    """
    Return the estimated price with given mileage.
    """

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
