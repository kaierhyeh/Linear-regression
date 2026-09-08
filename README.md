<p align="right">
  <a href="#ft_linear_regression-繁體中文">
    <img src="https://img.shields.io/badge/中文-sienna?style=for-the-badge" />
  </a>
</p>

# ft_linear_regression

---

An introduction to machine learning: the goal of **ft_linear_regression** is to implement a simple machine learning algorithm from scratch. By using **Gradient Descent**, we train a single-variable linear regression model to predict the price of a car based on its mileage.

This project reinforces core concepts behind AI algorithms without relying on heavy external libraries like `numpy.polyfit` or `scikit-learn`.

### Example Dataset (Mileage vs Price)
| Mileage (km) | Price (€) |
|--------------|-----------|
| 240,000      | 3,650     |
| 114,800      | 5,350     |
| 63,060       | 6,390     |
| 22,899       | 7,990     |

## ✨ Core Features

- **Gradient Descent Algorithm**: Implemented manually to iteratively minimize the Mean Squared Error (MSE) cost function.
- **Simultaneous Updates**: Strictly adheres to the mathematical foundation of gradient descent by calculating partial derivatives prior to updating $\theta_0$ and $\theta_1$.
- **Feature Normalization**: Utilizes Min-Max Scaling on both `km` and `price` to unify their magnitudes, preventing wildly oscillating gradients and enabling smooth convergence.
- **Model Evaluation ($R^2$ Score)**: Automatically computes the Coefficient of Determination to measure how well the regression line fits the actual data points.
- **Data Visualization**: Includes a plotting mechanism to overlay the theoretical regression line onto a scatter plot of the original dataset.

## 🚀 Installation & Execution

The project operates entirely in Python. (Optional: `matplotlib` is required for data visualization).

```bash
cd "Linear regression"
```

### 1. Training Phase
Train the model by reading `data.csv` and computing the optimal parameters. The resulting weights ($\theta_0$ and $\theta_1$) are naturally saved to `thetas.csv`.

```bash
python3 train.py
```

### 2. Prediction Phase
Load the saved weights and predict the price of a car based on a user-input mileage. (If the model has not been trained yet, the program safely defaults to returning `0`).

```bash
python3 predict.py
```

### 3. Precision Evaluation (Bonus)
Calculate the precision of the trained model (R² Score, MAE, RMSE).

```bash
python3 precision.py
```

## 📊 Evaluation Metrics
During the training phase, the terminal guarantees full transparency by outputting:
- $\theta_0$ (Intercept)
- $\theta_1$ (Slope / Weight)
- **$R^2$ Score**: Typically ranging between `0.70` and `0.80` for this particular dataset, proving a strong negative correlation between mileage and price.

<br/>

---

<br/>

<p align="right">
  <a href="#ft_linear_regression">
    <img src="https://img.shields.io/badge/-TOP-sienna?style=for-the-badge" />
  </a>
</p>

# ft_linear_regression (繁體中文)

這是一個基礎且經典的機器學習體驗專案：**ft_linear_regression**。本專案的目標是完全從零開始，利用 **梯度下降 (Gradient Descent)** 的數學原理，訓練一個單變數的線性回歸模型，藉由汽車的「里程數 (Mileage)」來預測其「價格 (Price)」。

本專案旨在深入理解 AI 演算法背後的邏輯與數學基礎，因此**嚴格禁止**依賴如 `numpy.polyfit` 或 `scikit-learn` 等直接吐出解答的高階外部函式庫。

### 數據範例 (里程數 vs 價格)
| 里程數 (km) | 預期價格 (€) |
|------------|-------------|
| 240,000    | 3,650       |
| 114,800    | 5,350       |
| 63,060     | 6,390       |
| 22,899     | 7,990       |

## ✨ 核心技術特點

- **梯度下降 (Gradient Descent)**：手刻迴圈邏輯，逐步調整模型權重以最小化「均方誤差 (MSE, Mean Squared Error)」。
- **同時更新 (Simultaneous Updates)**：嚴格遵守微積分推導，在一個迴圈節拍中先算出所有的偏微分梯度，再「同時」去更新 $\theta_0$ 與 $\theta_1$，避免收斂軌跡錯誤。
- **特徵正規化 (Feature Normalization)**：因為 `km` 與 `price` 的數量級相差過大，本專案引進了 Min-Max Scaling 技術，將兩者強制壓縮至 `[0, 1]` 區間內進行訓練，這解決了學習率 (Learning Rate) 難以同時滿足雙邊收斂的世紀難題。
- **模型準確度評估 ($R^2$ Score)**：自動計算判定係數 (Coefficient of Determination)，以此科學指標驗證模型解釋變異的能力。
- **資料視覺化 (Bonus)**：使用 `matplotlib` 將散佈圖 (Scatter Plot) 與訓練產生的回歸直線完美疊加，達到資料科學一目了然的操作質感。

## 🚀 安裝與執行

本專案由 Python 撰寫（若要啟用 Bonus 視覺化，環境中需安裝 `matplotlib` 套件）。

```bash
cd "Linear regression"
```

### 1. 訓練階段 (Training)
讀取 `data.csv` 並啟動梯度下降迴圈，最終會將算出最完美的線性權重儲存為 `thetas.csv`。

```bash
python3 train.py
```

### 2. 預測階段 (Prediction)
讀取訓練好的存檔參數，根據使用者輸入的里程數，即時吐出預估價格。（若在執行訓練程式前就啟動，系統會依照 Subject 要求防呆回傳 `0`）。

```bash
python3 predict.py
```

### 3. 精準度評估階段 (Bonus)
獨立執行精準度評估程式，計算 $R^2$ Score、MAE 與 RMSE。

```bash
python3 precision.py
```

## 📊 評估與輸出
在訓練結束的當下，終端機會提供乾淨清晰的模型報告：
- $\theta_0$ (截距/基礎價格)
- $\theta_1$ (斜率/折舊率)
- **$R^2$ Score**：針對這份資料集，標準精準度通常落在 `0.70` 到 `0.80` 之間，這從數學上證實了「里程數與價格之間具備強烈的負相關性」。