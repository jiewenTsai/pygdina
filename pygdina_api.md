# pygdina API 說明文件

> Pure-Python GDINA 實作（identity link，MMLE/EM 估計）  
> 對應 R 套件：`GDINA`（Ma & de la Torre, 2020）、`CDM`（Robitzsch et al.）

---

## 依賴套件

```python
numpy
scipy.special.logsumexp
```

---

## 模組層級函式

### `_skill_profiles(K)`

生成所有 2^K 個二元屬性組合矩陣。

| 參數 | 型別 | 說明 |
|---|---|---|
| `K` | `int` | 屬性數量 |

**回傳**：`ndarray (2^K, K)`，依 binary counting 排序，與 R GDINA / CDM 一致。

```python
_skill_profiles(2)
# → [[0,0], [1,0], [0,1], [1,1]]
```

> 此排序是 `parloc` 正確對應的前提，不可更改。

---

## 主類別 `GDINA`

### 建構子 `GDINA(...)`

```python
mod = GDINA(
    att_dist   = "saturated",   # 屬性分佈結構
    max_iter   = 2000,           # 最大 EM 迭代次數
    conv_crit  = 1e-5,           # 收斂門檻 |ΔLL|
    n_starts   = 3,              # 起始值數量
    lower_p    = 1e-4,           # 題目機率下界
    upper_p    = 1 - 1e-4,       # 題目機率上界
    random_seed = 123456,        # 隨機種子
    verbose    = True,           # 是否印出迭代進度
)
```

#### 參數說明

| 參數 | 預設值 | 說明 |
|---|---|---|
| `att_dist` | `"saturated"` | `"saturated"`：對 2^K 個類別估計無結構混合比例（等同 R GDINA 的 `att.dist="saturated"`）；`"independent"`：K 個技能獨立，以 K 個邊際機率參數化，自由參數從 2^K−1 降到 K |
| `max_iter` | `2000` | EM 最大迭代次數，超過視為未收斂 |
| `conv_crit` | `1e-5` | 收斂準則：連續兩次 E-step 的 \|ΔLL\| 低於此值則停止 |
| `n_starts` | `3` | 起始值組數。Start 0 為確定性初始值（複製 R CDM `seed=0` 的 ANOVA delta 設計）；Start 1+ 為隨機 DINA / DINO / A-CDM 型起始值。複雜度為 O(n_starts × 1 E-step) + O(max_iter)，非 O(n_starts × EM) |
| `lower_p` / `upper_p` | `1e-4` / `1−1e-4` | M-step 題目機率的上下界限制，防止邊界退化 |
| `random_seed` | `123456` | 控制隨機起始值的 numpy 種子 |
| `verbose` | `True` | 每次迭代印出 `Iter / \|ΔLL\| / deviance` |

---

### 公開方法

#### `fit(dat, Q)` — 模型估計

```python
mod.fit(dat, Q)
```

| 參數 | 型別 | 說明 |
|---|---|---|
| `dat` | `array-like (N, J)` | 二元作答矩陣，0/1，不支援 missing |
| `Q` | `array-like (J, K)` | Q 矩陣，`Q[j,k]=1` 表示第 j 題需要第 k 個技能 |

**回傳**：`self`（支援 method chaining）

**流程摘要：**
1. 建立 `att_pattern`（所有 2^K 屬性組合）與 `parloc`（global class → reduced group 對應表）
2. 壓縮資料為唯一 response pattern + 頻率
3. 以 uniform prior 初始化 `log_prior`
4. 從 `n_starts` 組起始值中，選初始觀測 LL 最高者進入完整 EM
5. EM 每次迭代同步記錄最低 deviance 的參數（對應 R CDM `save.devmin=TRUE`），收斂後 rollback 至最佳狀態

**Fitted attributes（`.fit()` 後可用）：**

| 屬性 | 形狀 | 說明 |
|---|---|---|
| `item_parm` | `(J, Lj_max)` | 各 item 在各 reduced group 的答對機率；超出 2^Kj 的欄位補零 |
| `log_prior` | `(L,)` | 所有 L=2^K 個 class 的 log 混合比例 |
| `att_pattern` | `(L, K)` | 所有屬性組合矩陣，binary counting 排序 |
| `parloc` | `(J, L)` | 1-indexed，parloc[j,l] = item j 在 global class l 下對應的 reduced group 索引 |
| `log_post` | `(N, L)` | 每位受測者對每個 class 的 log 後驗機率 |
| `deviance` | `float` | −2 × LL（最低 deviance 解） |

---

#### `person_parm(what)` — 受測者屬性估計

```python
mp  = mod.person_parm("mp")    # 邊際後驗機率
eap = mod.person_parm("eap")   # EAP 二元分類
mapp = mod.person_parm("map")  # MAP 二元分類
```

| 參數 | 預設值 | 說明 |
|---|---|---|
| `what` | `"mp"` | 輸出類型（見下表） |

| `what` 值 | 輸出形狀 | 說明 |
|---|---|---|
| `"mp"` | `(N, K)` 連續值 | 各技能的邊際後驗精熟機率 P(α_k=1\|X)，∈ (0,1) |
| `"eap"` | `(N, K)` 0/1 | EAP 分類：mp > 0.5 視為精熟 |
| `"map"` | `(N, K)` 0/1 | MAP 分類：後驗最高的屬性組合 |

**回傳**：`ndarray (N, K)`

> 需先呼叫 `.fit()`，否則拋出 `RuntimeError`。

---

#### `item_table()` — 題目參數表

```python
rows = mod.item_table()
import pandas as pd
df = pd.DataFrame(rows)
```

**回傳**：`list[dict]`，每筆記錄包含：

| 欄位 | 型別 | 說明 |
|---|---|---|
| `item` | `int` | 1-indexed 題目編號 |
| `group` | `int` | 1-indexed reduced group 編號 |
| `pattern` | `tuple[int, ...]` | 該 group 對應的 required skills 組合（0/1） |
| `P` | `float` | 該 group 的估計答對機率 |

> 需先呼叫 `.fit()`，否則拋出 `RuntimeError`。

---

## 完整使用範例

```python
import numpy as np
from pygdina import GDINA, _skill_profiles

# 1. 準備資料
Q   = np.array([[1,0],[0,1],[1,1],[1,0],[0,1]])   # J=5, K=2
dat = np.random.binomial(1, 0.6, size=(500, 5))   # N=500

# 2. 估計
mod = GDINA(att_dist="saturated", n_starts=3, verbose=True)
mod.fit(dat, Q)

# 3. 題目參數
import pandas as pd
df_items = pd.DataFrame(mod.item_table())

# 4. 受測者分類
mp  = mod.person_parm("mp")    # shape (500, 2)
eap = mod.person_parm("eap")   # shape (500, 2)

# 5. 模型摘要
print(f"Deviance: {mod.deviance:.2f}")
print(f"Skill mastery rates: {mp.mean(axis=0).round(3)}")
```

---

## 設計對齊說明

| 設計面 | 對應 R 套件 |
|---|---|
| 屬性組合排序（binary counting） | R GDINA / CDM |
| Start 0 確定性初始值（ANOVA delta） | R CDM `seed=0` |
| Multi-start：比較初始 LL 再選一個跑完整 EM | R GDINA `nstarts` |
| Best-deviance rollback | R CDM `save.devmin=TRUE` |
| 收斂準則 \|ΔLL\| | 此實作獨有（R 用雙重條件：Δpar + Δdev） |
