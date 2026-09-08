# ft_linear_regression Defense Guide

這份文件涵蓋了線性回歸專案的核心概念、數學原理、演算法設計選擇，以及各種可能被詢問的技術細節。

---

## 0. 語言選擇 (Language Choice)

本專案選擇使用 **Python** 實作。

| 語言 | 視覺化 | 數學運算 | 開發速度 | 綜合評估 |
|---|---|---|---|---|
| **Python** | ✅ `matplotlib` 原生支援 | ✅ 手刻即可 | ⭐⭐⭐ | **最適合本專案** |
| **C++** | ❌ 需第三方庫 | ✅ 高效能 | ⭐ | 過度工程 |

Subject 指出「你應該使用一個容易視覺化資料的語言（You should use a language that allows you to easily visualize your data）」，Python 搭配 `matplotlib` 是最自然的選擇。

> **關於函式庫限制**：Subject 禁止使用 `numpy.polyfit` 等直接做線性回歸的函式庫。本專案**手寫**梯度下降演算法，僅用 `csv`（讀資料）與 `matplotlib`（視覺化）兩個函式庫。

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
試圖找出一條直線（在 2D 中），使得「所有資料點到這條線的距離平方和」最小。這個衡量標準稱為 **方均誤差 (Mean Squared Error, MSE)**：

$$MSE = \frac{1}{m} \sum_{i=0}^{m-1} (\text{estimatePrice}(\text{km}_i) - \text{price}_i)^2$$

我們的目標就是找到讓 MSE 最小的 θ0 和 θ1。

---

## 3. 梯度下降 (Gradient Descent)

### 3.1 誤差函數 (Cost Function, $J$)
為了評估預測模型的好壞，我們定義誤差函數 $J(\theta_0, \theta_1)$ 為方均誤差（MSE）的一半：

$$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=0}^{m-1} [(\theta_0 + \theta_1 x_i) - y_i]^2$$

> **為什麼是一半？**<br>
> 為了在微分時，次方的 $2$ 可以和 $\frac{1}{2}$ 抵消，讓梯度公式更簡潔。

---

### 3.2 推導
梯度下降的精神在於計算誤差函數對各參數的斜率（偏導數），以掌握最陡的下坡方向。

令第 $i$ 筆資料的預測誤差為中間變數 $u_i$：
$$u_i = (\theta_0 + \theta_1 x_i) - y_i$$
$$J = \frac{1}{2m} \sum_{i=0}^{m-1} [(\theta_0 + \theta_1 x_i) - y_i]^2 = \frac{1}{2m} \sum_{i=0}^{m-1} u_i^2$$

根據微分的線性性質（求和與微分可交換）與連鎖律：

$$
\begin{aligned}
\frac{\partial J}{\partial \theta} 
&= \frac{\partial}{\partial \theta} \left( \frac{1}{2m} \sum_{i=0}^{m-1} u_i^2 \right) \\
&= \frac{1}{2m} \sum_{i=0}^{m-1} \frac{\partial (u_i^2)}{\partial \theta} \\
&= \frac{1}{2m} \sum_{i=0}^{m-1} \left( \frac{\partial (u_i^2)}{\partial u_i} \cdot \frac{\partial u_i}{\partial \theta} \right) \\
&= \frac{1}{2m} \sum_{i=0}^{m-1} \left( 2 u_i \cdot \frac{\partial u_i}{\partial \theta} \right) \\
&= \frac{1}{m} \sum_{i=0}^{m-1} \left( u_i \cdot \frac{\partial u_i}{\partial \theta} \right)
\end{aligned}
$$

現在分別對 $\theta_0$ 與 $\theta_1$ 微分：

1. **對 $\theta_0$（截距）微分**：
   $$\frac{\partial u_i}{\partial \theta_0} = \frac{\partial}{\partial \theta_0} [(\theta_0 + \theta_1 x_i) - y_i] = 1$$
   代回連鎖律公式：
   $$\frac{\partial J}{\partial \theta_0} = \frac{1}{m} \sum_{i=0}^{m-1} \left( u_i \cdot \frac{\partial u_i}{\partial \theta_0} \right) = \frac{1}{m} \sum_{i=0}^{m-1} (u_i \cdot 1) = \frac{1}{m} \sum_{i=0}^{m-1} [(\theta_0 + \theta_1 x_i) - y_i]$$
   乘上學習率 $\text{learningRate}$ 即為更新量：
   $$\text{tmp}\theta_0 = \text{lr} \times \frac{1}{m} \sum_{i=0}^{m-1} (\text{estimatePrice}(\text{km}_i) - \text{price}_i)$$

2. **對 $\theta_1$（斜率）微分**：
   $$\frac{\partial u_i}{\partial \theta_1} = \frac{\partial}{\partial \theta_1} [(\theta_0 + \theta_1 x_i) - y_i] = x_i$$
   代回連鎖律公式：
   $$\frac{\partial J}{\partial \theta_1} = \frac{1}{m} \sum_{i=0}^{m-1} \left( u_i \cdot \frac{\partial u_i}{\partial \theta_1} \right) = \frac{1}{m} \sum_{i=0}^{m-1} (u_i \cdot x_i) = \frac{1}{m} \sum_{i=0}^{m-1} [(\theta_0 + \theta_1 x_i) - y_i] \cdot x_i$$
   乘上學習率 $\text{lr}$ 即為更新量：
   $$\text{tmp}\theta_1 = \text{lr} \times \frac{1}{m} \sum_{i=0}^{m-1} (\text{estimatePrice}(\text{km}_i) - \text{price}_i) \times \text{km}_i$$

> ### 💡 核心觀念解析
> 
> **1. 為何乘上學習率（$\text{learningRate}$）為更新量？**<br>
> $\frac{\partial J}{\partial \theta}$ 僅代表當前點的「斜率（坡度與方向）」，並非實際移動的跨距。
> - 若直接用偏導數更新（相當於 $\text{lr}=1$），坡度陡峭時步幅會過大，導致在谷底兩側劇烈震盪甚至發散（Exploding Gradient）。
> - 乘上學習率（如 0.1）可以精確控制每一步邁出的「步長（$\text{lr} \times $梯度）」，確保演算法平穩且受控地收斂至極小值。
> 
> **2. 為何不用解析解（直接令偏導數 = 0）？**<br>
> 線性回歸其實可以直接令 $\frac{\partial J}{\partial \theta_0} = 0, \frac{\partial J}{\partial \theta_1} = 0$ 解出二元一次聯立方程式（或矩陣形式的正規方程式 $\theta = (X^T X)^{-1} X^T y$）。不採用而使用梯度下降的原因如下：
> - **計算複雜度過高**：解析解需要計算反矩陣 $(X^T X)^{-1}$，時間複雜度高達 $O(n^3)$。當特徵數量成千上萬時，記憶體與算力會直接當機；而梯度下降每次迭代僅需 $O(m \cdot n)$。
> - **無法處理巨量資料**：若資料有數百萬筆以上，反矩陣無法一次全部載入記憶體運算；梯度下降則可藉由批次（Mini-batch）分批訓練。
> - **通用性（深度學習必備）**：在邏輯回歸、神經網路等複雜非線性模型中，令「導數 = 0」是無法代數求解的超越方程式（不存在解析解）。採用梯度下降是為了學習通用於所有現代 AI 與機器學習的核心優化方法。

---

### 3.3 參數更新
計算出更新量後，**同時**更新兩參數：
$$\theta_0 \mathrel{-}= \text{tmp}\theta_0, \quad \theta_1 \mathrel{-}= \text{tmp}\theta_1$$

**為什麼是減號？**
* **當斜率（偏導）為正**：代表 $\theta$ 增加會使誤差上升。若要讓誤差下降，$\theta$ 必須**向左減少**。
* **當斜率（偏導）為負**：代表 $\theta$ 增加會使誤差下降。若要讓誤差下降，$\theta$ 必須**向右增加**。
因此更新方向必須永遠沿著**負梯度方向（Negative Gradient）** $\theta := \theta - \alpha \frac{\partial J}{\partial \theta}$。

---

### 3.4 收斂條件
在工程與數值分析中，為避免無窮迴圈或盲目運算，我們為梯度下降設定了雙重收斂機制：

1. **向量長度判定（L2 Norm）**：
   每輪迭代計算更新量步伐的向量長度（歐氏範數）：
   $$\|\Delta \theta\| = \sqrt{(\text{tmp}\theta_0)^2 + (\text{tmp}\theta_1)^2}$$
   當 $\|\Delta \theta\| \le \text{tol}$（預設值為 $10^{-4}$）時，代表當前斜率已趨近於 0，參數變化量極微小（小於 $0.01\%$），判定已抵達誤差範圍內的最優解，觸發 `break` 提前退出。
2. **最大次數保護（Iterations Cap）**：
   以 `iterations=1000` 作為預設安全上限，確保在任何異常資料集下程式必定會在有限時間內結束。

---

### 📌 必備 Defense 問題

#### Q1：為什麼要「同時更新 (Simultaneous Update)」？
如果先更新 $\theta_0$，接著算 $\theta_1$ 時就會用到「已經更新過的 $\theta_0$」，這破壞了兩者是在同一時間點狀態 $(\theta_0, \theta_1)$ 下計算梯度的數學假設。**正確做法是先計算出暫存量 `tmp0` 與 `tmp1`，再同時更新 `t0` 與 `t1`**。

#### Q2：既然要求極小值，為什麼不直接令「偏導數 = 0」解聯立方程式？為什麼要用梯度下降一步一步走？
1. **線性回歸確實有解析解（Closed-form Solution / 正規方程式 Normal Equation）**：$\theta = (X^T X)^{-1} X^T y$。在特徵少時可以一步到位。
2. **計算複雜度與記憶體限制**：公式解涉及矩陣求逆 $(X^T X)^{-1}$，複雜度高達 $O(n^3)$。當特徵數量成千上萬（如文字或影像）或樣本數達數百萬時，反矩陣運算會讓記憶體與算力崩潰，而梯度下降只需 $O(m \cdot n)$。
3. **通用性（深度學習必備）**：在邏輯回歸、神經網路或深度學習中，誤差函數高度非線性，令「導數 = 0」會形成超越方程式，在數學上**不存在代數解析解**。梯度下降法是唯一能普遍應用在各種複雜模型上的通用優化演算法。

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

## 5. 精準度指標 (Precision Metrics)

### 5.1 R² Score (判定係數)

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

其中：

- $y_i$：第 $i$ 筆資料的實際價格。
- $\hat{y}_i = \theta_0 + \theta_1 x_i$：模型對第 $i$ 筆資料的預測價格。
- $\bar{y} = \frac{1}{m}\sum_{i=0}^{m-1} y_i$：所有實際價格的平均值。
- $SS_{res} = \sum_{i=0}^{m-1}(y_i - \hat{y}_i)^2$：殘差平方和（Residual Sum of Squares），表示模型預測誤差的總量。
- $SS_{tot} = \sum_{i=0}^{m-1}(y_i - \bar{y})^2$：總平方和，表示只用平均價格預測時的誤差總量。

因此，$R^2$ 是「模型剩下的誤差」相對於「平均值基準的誤差」所改善的比例：

$$R^2 = 1 - \frac{\text{模型誤差}}{\text{平均值基準誤差}}$$

- $R^2 = 1.0$：代表模型預測與實際資料非常符合，沒有誤差。
- $R^2 = 0.0$：代表模型的效果跟直接取**平均值**沒有兩樣，沒有預測價值。
- $0 < R^2 < 1$：模型解釋了部分價格變化；例如 $R^2 = 0.7330$ 可理解為模型解釋了約 73.3% 的價格變異。
- $R^2 < 0$：模型比直接使用平均值預測更差。

---

### 5.2 其他評估指標（`precision.py`）

除了 $R^2$ 以外，在 `precision.py` 中我們額外計算了三種資料科學常見的精準度衡量標準：

#### 1. MAE (Mean Absolute Error, 平均絕對誤差)
$$\text{MAE} = \frac{1}{m}\sum_{i=0}^{m-1} |y_i - \hat{y}_i|$$
* **含意與物理意義**：直接計算預測值與真實值的「絕對差距平均」。
* **特點**：單位與車價相同（歐元 €）。最直觀好懂，不易受極端離群值過度拉偏。
* **專案實測數值**：約 **$557.65$ €**，代表模型對每輛車的預測平均誤差約 557 歐元。

#### 2. MSE (Mean Squared Error, 均方誤差)
$$\text{MSE} = \frac{1}{m}\sum_{i=0}^{m-1} (y_i - \hat{y}_i)^2$$
* **含意與物理意義**：誤差平方的平均值，也就是梯度下降訓練時的最佳化目標（損失函數）。
* **特點**：平方放大較大的失誤，對預測偏差嚴厲懲罰；但單位變成平方（€$^2$），較難直接對應現實直覺。
* **專案實測數值**：約 **$445,646.79$**。

#### 3. RMSE (Root Mean Squared Error, 均方根誤差)
$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{m}\sum_{i=0}^{m-1} (y_i - \hat{y}_i)^2}$$
* **含意與物理意義**：將 MSE 開根號，把單位還原為歐元（€）。
* **特點**：既保留了「嚴懲大誤差」的敏感度，又有真實貨幣單位的可讀性。
* **專案實測數值**：約 **$667.57$ €**。
  *(註：因為 $\text{RMSE} > \text{MAE}$，代表資料中存在少數偏離回歸線較遠的資料點)*

---

## 6. 程式碼閱讀順序建議 (Code Reading Guide)

1. **`data.csv`**：先看資料，了解我們在處理什麼。24 筆 km vs price 的負相關資料。
2. **`train.py`**：
   - `load_data()` → `normalize()` → `train()` → `denormalize_thetas()` → `save_thetas()`
   - 展示重點：`train()` 函式裡的同時更新邏輯，以及正規化的必要性。
3. **`predict.py`**：
   - `load_thetas()` → `estimate_price()`
   - 展示重點：Subject 指定的 estimatePrice 公式的直接實作，以及未訓練時預設 θ0=0, θ1=0。
4. **`precision.py` (Bonus)**：
   - 獨立評估模型準確度程式，計算 R² Score、MAE、RMSE。
5. **視覺化 (Bonus)**：在 `train.py` 的 `main()` 中，展示散佈圖 + 回歸線的重疊效果，直觀地說明模型的品質。
