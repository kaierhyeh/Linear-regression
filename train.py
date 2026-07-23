import csv
import math


# ──────────────────────────────────────────────
# 1. 讀取資料集
# ──────────────────────────────────────────────
def load_data(filepath):
    kms, prices = [], []
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            kms.append(float(row['km']))
            prices.append(float(row['price']))
    return kms, prices


# ──────────────────────────────────────────────
# 2. Min-Max 正規化 / 反正規化
#    作用：將 km (~240000) 與 price (~8000) 縮放至 [0, 1]
#    避免梯度下降因數量級差異而難以收斂
# ──────────────────────────────────────────────
def normalize(data):
    min_v = min(data)
    max_v = max(data)
    return [(x - min_v) / (max_v - min_v) for x in data], min_v, max_v


def denormalize_thetas(t0, t1, km_min, km_max, price_min, price_max):
    """將在正規化空間訓練的 θ 還原回原始尺度"""
    scale = price_max - price_min
    range_km = km_max - km_min
    theta1 = t1 * scale / range_km
    theta0 = t0 * scale + price_min - theta1 * km_min
    return theta0, theta1


# ──────────────────────────────────────────────
# 3. 梯度下降 (Gradient Descent)
#    公式來自 Subject：
#      tmpθ0 = lr × (1/m) × Σ (estimatePrice(km[i]) - price[i])
#      tmpθ1 = lr × (1/m) × Σ (estimatePrice(km[i]) - price[i]) × km[i]
#    重點：必須「同時更新」(Simultaneous Update)
# ──────────────────────────────────────────────
def train(kms_norm, prices_norm, learning_rate=0.1, iterations=1000):
    t0, t1 = 0.0, 0.0
    m = len(kms_norm)

    for _ in range(iterations):
        # 先計算所有偏差，再同時更新
        tmp0 = learning_rate * (1 / m) * sum(
            (t0 + t1 * kms_norm[i]) - prices_norm[i]
            for i in range(m)
        )
        tmp1 = learning_rate * (1 / m) * sum(
            ((t0 + t1 * kms_norm[i]) - prices_norm[i]) * kms_norm[i]
            for i in range(m)
        )
        t0 -= tmp0
        t1 -= tmp1

    return t0, t1


# ──────────────────────────────────────────────
# 4. 精準度指標：R² Score (Coefficient of Determination)
#    R² = 1 - SS_res / SS_tot
#    R² = 1.0 → 完美擬合；R² = 0 → 與用平均值預測相同
# ──────────────────────────────────────────────
def r_squared(kms, prices, theta0, theta1):
    mean_price = sum(prices) / len(prices)
    ss_res = sum((prices[i] - (theta0 + theta1 * kms[i])) ** 2 for i in range(len(prices)))
    ss_tot = sum((prices[i] - mean_price) ** 2 for i in range(len(prices)))
    return 1 - ss_res / ss_tot


# ──────────────────────────────────────────────
# 5. 儲存 θ0, θ1 至 thetas.csv
# ──────────────────────────────────────────────
def save_thetas(theta0, theta1, filepath='thetas.csv'):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['theta0', 'theta1'])
        writer.writerow([theta0, theta1])


# ──────────────────────────────────────────────
# 6. 主流程
# ──────────────────────────────────────────────
def main():
    kms, prices = load_data('data.csv')

    # 正規化
    kms_norm, km_min, km_max       = normalize(kms)
    prices_norm, price_min, price_max = normalize(prices)

    # 梯度下降於正規化空間
    t0_norm, t1_norm = train(kms_norm, prices_norm)

    # 反正規化回原始尺度
    theta0, theta1 = denormalize_thetas(
        t0_norm, t1_norm, km_min, km_max, price_min, price_max
    )

    # 計算精準度
    r2 = r_squared(kms, prices, theta0, theta1)

    print(f"Training complete!")
    print(f"  theta0 (intercept) = {theta0:.4f}")
    print(f"  theta1 (slope)     = {theta1:.8f}")
    print(f"  R² Score           = {r2:.4f}")

    save_thetas(theta0, theta1)
    print(f"\nParameters saved to thetas.csv")

    # Bonus：視覺化
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 6))

        # 散佈圖
        plt.scatter(kms, prices, color='steelblue', label='Data points', zorder=3)

        # 回歸線
        x_line = [min(kms), max(kms)]
        y_line = [theta0 + theta1 * x for x in x_line]
        plt.plot(x_line, y_line, color='tomato', linewidth=2,
                 label=f'Regression line (R²={r2:.4f})')

        plt.xlabel('Mileage (km)')
        plt.ylabel('Price (€)')
        plt.title('ft_linear_regression: Car Price vs Mileage')
        plt.legend()
        plt.tight_layout()
        plt.savefig('plot.png', dpi=150)
        plt.show()
        print("Plot saved to plot.png")

    except ImportError:
        print("(matplotlib not found — skipping visualization)")


if __name__ == '__main__':
    main()
