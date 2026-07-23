# ft_linear_regression Defense Guide

這份文件涵蓋了線性回歸專案的核心概念、數學原理、演算法設計選擇，以及各種可能被詢問的技術細節。

---

## 0. 語言選擇 (Language Choice)

本專案選擇使用 **Python** 實作。

| 語言 | 視覺化 | 數學運算 | 開發速度 | 綜合評估 |
|---|---|---|---|---|
| **Python** | ✅ `matplotlib` 原生支援 | ✅ 手刻即可 | ⭐⭐⭐ | **最適合本專案** |
| **C++** | ❌ 需第三方庫 | ✅ 高效能 | ⭐ | 過度工程 |

Subject 明確指出「你應該使用一個容易視覺化資料的語言（You should use a language that allows you to easily visualize your data）」，Python 搭配 `matplotlib` 是最自然的選擇。

> **關於函式庫限制**：Subject 禁止使用 `numpy.polyfit` 等直接做線性回歸的函式庫。本專案**手寫梯度下降演算法**，僅用 `csv`（讀資料）與 `matplotlib`（視覺化）兩個函式庫。

---

## 1. 核心概念與執行流程 (Workflow & Logic)

**目標**：給定 24 筆「里程 (km) vs 價格 (price)」的資料，學出最佳的一條直線：
$$\text{estimatePrice(mileage)} = \theta_0 + \theta_1 \times \text{mileage}$$

**執行流程**：
1. **`train.py`**：讀取 `data.csv` → 正規化 → 梯度下降 → 反正規化 → 儲存 θ0, θ1 至 `thetas.csv`
2. **`predict.py`**：讀取 `thetas.csv` → 提示使用者輸入里程數 → 輸出預測價格

---

## 2. 線性回歸 (Linear Regression)

**什麼是線性回歸？**
試圖找出一條直線（在 2D 中），使得「所有資料點到這條線的距離平方和」最小。這個衡量標準稱為 **均方誤差 (Mean Squared Error, MSE)**：

$$MSE = \frac{1}{m} \sum_{i=0}^{m-1} (\text{estimatePrice}(\text{km}_i) - \text{price}_i)^2$$

我們的目標就是找到讓 MSE 最小的 θ0 和 θ1。

---

## 3. 梯度下降 (Gradient Descent)

**為什麼用梯度下降？**
MSE 這個「成本函數 (Cost Function)」是一個對 θ0 和 θ1 的開口朝上的拋物面。梯度下降就是不斷沿著「最陡的下坡方向」走，直到走到最低點（全局最小值）。

**更新公式（直接來自 Subject）**：
$$\text{tmp}\theta_0 = \text{lr} \times \frac{1}{m} \sum_{i=0}^{m-1} (\text{estimatePrice}(\text{km}_i) - \text{price}_i)$$
$$\text{tmp}\theta_1 = \text{lr} \times \frac{1}{m} \sum_{i=0}^{m-1} (\text{estimatePrice}(\text{km}_i) - \text{price}_i) \times \text{km}_i$$

然後**同時**更新：
$$\theta_0 \mathrel{-}= \text{tmp}\theta_0, \quad \theta_1 \mathrel{-}= \text{tmp}\theta_1$$

### 📌 必備 Defense 問題：為什麼要「同時更新」？
如果你先更新 θ0 再用新的 θ0 去算 θ1 的更新量，你在每次迭代中的「出發點」就不一致了（你已經站在下一步的位置上了）。這會打破梯度下降的數學推導，可能導致震盪或無法收斂。**正確做法是先算出所有的 tmp，然後在同一時刻更新 θ0 和 θ1。**

---

## 4. 特徵正規化 (Feature Normalization)

### 為什麼需要正規化？
本資料集中，`km` 的數量級是 **~240,000**，而 `price` 的數量級是 **~8,000**。

如果直接用原始數值進行梯度下降：
- θ1 需要非常小的 learning rate 才不會在 `km` 方向上震盪。
- 但這樣 θ0 方向的學習會極度緩慢。
- 結果：無論你怎麼調 learning rate，梯度下降都很難同時在兩個方向上表現良好。

**解決方案：Min-Max Normalization**
$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

將 `km` 和 `price` 都縮放至 [0, 1]，這樣兩個方向的數量級就一致了，梯度下降能有效且穩定地收斂。

### 反正規化
在正規化空間中訓練出 θ0', θ1' 後，需要反推回原始尺度的 θ0, θ1，這樣儲存的參數才能直接用於預測原始里程數和價格。

---

## 5. 精準度指標：R² Score

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

- $SS_{res} = \sum(y_i - \hat{y}_i)^2$：殘差平方和（預測的不準確性）
- $SS_{tot} = \sum(y_i - \bar{y})^2$：總平方和（用「平均值預測」的不準確性）

**解讀**：
- $R^2 = 1.0$：完美擬合，所有預測都正確。
- $R^2 = 0.0$：你的模型和「直接用平均值預測」一樣差。
- $R^2 < 0$：你的模型甚至比直接說「平均值」還要差（通常代表模型出了問題）。

對本資料集，預期 R² 應在 **0.75 ~ 0.85** 之間，表示里程數能解釋約八成的價格變化。

---

## 6. 程式碼閱讀順序建議 (Code Reading Guide)

1. **`data.csv`**：先看資料，了解我們在處理什麼。24 筆 km vs price 的負相關資料。
2. **`train.py`**：
   - `load_data()` → `normalize()` → `train()` → `denormalize_thetas()` → `save_thetas()`
   - 展示重點：`train()` 函式裡的同時更新邏輯，以及正規化的必要性。
3. **`predict.py`**：
   - `load_thetas()` → `estimate_price()`
   - 展示重點：Subject 指定的 estimatePrice 公式的直接實作。
4. **視覺化 (Bonus)**：在 `train.py` 的 `main()` 中，展示散佈圖 + 回歸線的重疊效果，直觀地說明模型的品質。
