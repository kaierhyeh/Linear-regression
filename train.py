#!/usr/bin/env python3
import csv


# ──────────────────────────────────────────────
# 1. 讀取資料集
# ──────────────────────────────────────────────
def load_data(filepath):
    """Load data.csv"""

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
    """
    Normalize to eradicate the scale difference,
    accelerating the fitting process.
    """

    min_v = min(data)
    max_v = max(data)
    return [(x - min_v) / (max_v - min_v) for x in data], min_v, max_v


def denormalize_thetas(t0, t1, km_min, km_max, price_min, price_max):
    """Denormalize θs"""

    scale = price_max - price_min
    range_km = km_max - km_min
    theta1 = t1 * scale / range_km
    theta0 = t0 * scale + price_min - theta1 * km_min
    return theta0, theta1


# ──────────────────────────────────────────────
# 3. 梯度下降與最佳化 (Gradient Descent & Optimization)
# ──────────────────────────────────────────────
def compute_error(kms, prices, t0, t1):
    """Calculate the cost function J(θ0, θ1) = (1/2m) Σ (f(x) - y)^2"""
    m = len(kms)
    return (1 / (2 * m)) * sum(
        ((t0 + t1 * kms[i]) - prices[i]) ** 2 for i in range(m)
    )


def is_converged(tmp0, tmp1, tol=1e-4):
    """
    Calculate the length of the error vector (L2 norm) to check convergence.
    """
    error_length = (tmp0 ** 2 + tmp1 ** 2) ** 0.5
    return error_length <= tol, error_length
# tol for tolerance


def decay_learning_rate(lr):
    """1.5 -> 1.0 -> 0.5 -> 0.1 -> half"""

    if lr > 1.0:
        return 1.0
    elif lr > 0.5:
        return 0.5
    elif lr > 0.1:
        return 0.1
    else:
        return lr * 0.5


# ──────────────────────────────────────────────
# 3-1. Backtracking Line Search:
#    當步幅過大使誤差增加時，下調 learning rate 直到誤差下降。
# ──────────────────────────────────────────────
def backtracking_line_search(kms, prices, t0, t1,
                             grad0, grad1, current_error, lr):
    """
    Backtracking line search:
    Dynamically decrease the learning rate if a step increases error
    (overshooting), ensuring monotonic cost reduction toward the minimum.
    """

    while lr > 1e-4:
        tmp0 = lr * grad0
        tmp1 = lr * grad1
        next_error = compute_error(kms, prices, t0 - tmp0, t1 - tmp1)

        # 誤差下降，代表步長安全
        if next_error <= current_error:
            return lr, tmp0, tmp1, next_error

        # 誤差上升代表衝過頭，下調 lr 重新嘗試
        lr = decay_learning_rate(lr)

    # 若回溯至小於 1e-4 仍無法使誤差下降：
    # 代表線搜索失敗（可能已非常接近極小值，或是此方向步長過於粗糙）。
    return lr, lr * grad0, lr * grad1, current_error


# ──────────────────────────────────────────────
# 3-2. 梯度下降與最佳化 (Gradient Descent & Optimization)
#     tmpθ0 = learningRate × (1/m) Σ (estimatePrice(km[i]) - price[i])
#     tmpθ1 = lr × (1/m) Σ (estimatePrice(km[i]) - price[i]) × km[i]
# Update θ0、θ1 simultaneously until error length ||(tmp0, tmp1)|| <= tolerance
# or the upper bound `iterations` is reached.
# ──────────────────────────────────────────────
def train(kms_norm, prices_norm, lr=1.5, iterations=1000, tol=1e-4):
    """
    Train the linear regression model using gradient descent with:
    - Backtracking line search for adaptive learning rate
    - Simultaneous parameter updates (θ0, θ1)
    - Error length early stopping (||(tmp0, tmp1)|| <= tol)
    """

    t0, t1 = 0.0, 0.0
    m = len(kms_norm)
    current_error = compute_error(kms_norm, prices_norm, t0, t1)

    for i in range(1, iterations + 1):
        # 1. 計算誤差純梯度方向 (1/m) Σ (estimatePrice(km[i]) - price[i])
        grad0 = (1 / m) * sum(
            (t0 + t1 * kms_norm[j]) - prices_norm[j]
            for j in range(m)
        )
        grad1 = (1 / m) * sum(
            ((t0 + t1 * kms_norm[j]) - prices_norm[j]) * kms_norm[j]
            for j in range(m)
        )

        # 2. 回溯線搜索：確認步伐安全並取得當前最佳 lr 與更新量
        lr, tmp0, tmp1, current_error = backtracking_line_search(
            kms_norm, prices_norm, t0, t1, grad0, grad1, current_error, lr
        )

        # 3. 同時更新參數
        t0 -= tmp0
        t1 -= tmp1

        # 4. 收斂判定：誤差長度 <= tol 提前停止
        converged, error_length = is_converged(tmp0, tmp1, tol)
        if converged:
            print(
                f"Stopped at iteration {i}: "
                f"Error length ({error_length:.2e}) <= tolerance ({tol}) "
                f"(learningRate={lr})"
            )
            break

    return t0, t1


# ──────────────────────────────────────────────
# 4. 精準度指標：R² Score (Coefficient of Determination)
#    R² = 1 - SS_res / SS_tot
#    R² = 1.0 → 完美擬合；R² = 0 → 與用平均值預測相同
# ──────────────────────────────────────────────
def r_squared(kms, prices, theta0, theta1):
    """Calculate the R² score"""

    mean_price = sum(prices) / len(prices)
    ss_res = sum(
        (prices[i] - (theta0 + theta1 * kms[i])) ** 2
        for i in range(len(prices))
    )
    ss_tot = sum((prices[i] - mean_price) ** 2 for i in range(len(prices)))
    return 1 - ss_res / ss_tot


# ──────────────────────────────────────────────
# 5. 儲存 θ0, θ1 至 thetas.csv
# ──────────────────────────────────────────────
def save_thetas(theta0, theta1, filepath='thetas.csv'):
    """Save θ0, θ1 to thetas.csv"""

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['theta0', 'theta1'])
        writer.writerow([theta0, theta1])


# ──────────────────────────────────────────────
# 6. 主流程
# ──────────────────────────────────────────────
def main():
    """Main function to run the linear regression training."""

    kms, prices = load_data('data.csv')

    # 正規化
    kms_norm, km_min, km_max = normalize(kms)
    prices_norm, price_min, price_max = normalize(prices)

    # 梯度下降於正規化空間
    t0_norm, t1_norm = train(kms_norm, prices_norm)

    # 反正規化回原始尺度
    theta0, theta1 = denormalize_thetas(
        t0_norm, t1_norm, km_min, km_max, price_min, price_max
    )

    # 計算精準度
    r2 = r_squared(kms, prices, theta0, theta1)

    print(
        "Training complete!\n"
        f"  theta0 (intercept) = {theta0:.4f}\n"
        f"  theta1 (slope)     = {theta1:.8f}\n"
        f"  R² Score           = {r2:.4f}"
    )

    save_thetas(theta0, theta1)
    print("\nParameters saved to thetas.csv.")

    # Bonus：視覺化
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 6))

        # 散佈圖
        plt.scatter(kms, prices, color='steelblue',
                    label='Data points', zorder=3)

        # 回歸線
        x_line = [min(kms), max(kms)]
        y_line = [theta0 + theta1 * x for x in x_line]
        plt.plot(x_line, y_line, color='tomato', linewidth=2,
                 label=f'Regression line (R²={r2:.4f})')

        plt.xlabel('Mileage (km)')
        plt.ylabel('Price (€)')
        plt.title('Linear Regression: Car Price vs Mileage')
        plt.legend()
        plt.tight_layout()
        plt.savefig('plot.png', dpi=150)
        plt.show()
        print("Plot saved to plot.png.")

    except ImportError:
        print("(Matplotlib not found, skipping visualization)")


if __name__ == '__main__':
    main()
