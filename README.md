# 🧪 **Listwise Learning to Rank (LTR) for Lab Test Ranking**

Listwise Learning to Rank (LTR) optimizes the **entire ranking order** for a given query—unlike Pointwise or Pairwise approaches. It's especially effective for ranking **lab tests** by relevance to queries like:

> *"glucose in blood"*, *"bilirubin in plasma"*, *"white blood cells count"*

---

## 🔧 **Step 1: Define the Listwise LTR Model**

Listwise LTR models learn a *ranking function* that optimizes evaluation metrics such as **NDCG** (*Normalized Discounted Cumulative Gain*).

### ⚙️ Workflow:

1. **Input**: A list of lab tests (documents) for a given query.  
2. **Scoring Function**: A model predicts a *relevance score* per test.  
3. **Loss Function**:
   - **eXtreme NDCG** – a direct optimization of NDCG.
   - **LambdaRank** – also NDCG-focused.
4. **Output**: A ranked list of lab tests based on predicted relevance.

---

## 🧹 **Step 2: Data Preparation**

We calculate *relevance scores* for lab tests by computen two different scoring procedures.


### 1. Traditional Scoring

#### 🔍 1.1 Define Query Features
- **Component**: Substance measured (e.g., *Glucose*)
- **System**: Environment of measurement (e.g., *Blood*, *Serum/Plasma*)

#### 🧼 1.2. Preprocess Dataset
Each lab test includes:
- **Component**
- **System**

#### 🎯 1.3. Match Criteria
- **Exact Match**: Full match with the query term.
- **Partial Match**: Synonyms or semantically similar terms.

#### 🧮 1.4. Scoring Scheme
- **Exact Match** (Component) = weight(component) * weight(component)
- **Partial Match** (Component) = weight(component)/2 * weight(component)
- **Exact Match** (System) = weight(system) * weight(system)
- **Partial Match** (System) = weight(system)/2 * weight(system) No Match = 0

### ⚖️ 3. Normalize Scores
Normalize scores between 0 and 1 using:
- **Normalized Score** = score / max_score


### 💾 4. Export Data
Save the processed data and scores into a new **CSV** file for model training.

---

## 🛠️ **Step 3: Implement the Listwise LTR Model**

We use **LightGBM** due to its speed, simplicity, and support for listwise ranking.

### 📁 1. Dataset Preparation
- Load data from CSV.
- Encode categorical columns: `Query`, `Name`, `Component`, `System`, `Property`, `Measurement`.
- Create `Score_label` from `Normalized_Score`.
- Split into **train** and **test** sets.

### 📊 2. LightGBM Dataset Setup
- **Features**: Encoded columns.
- **Grouping**: Group by `Query` (listwise requirement).
- **Labels**: Use `Score_label`.

### 🧠 3. Train the Model
- **Objective**: `rank_xendcg`
- **Approach**: Simulate *AdaRank*-style boosting and reweighting using LightGBM parameters.

### 📈 4. Prediction
- Predict and normalize scores.
- Sort by `Query` and `Predicted Score`.
- Save results to `results.csv`.

---

## 🚀 **Step 4: Enhancing the Dataset**

To improve **NDCG**, we introduced new **features**, expanded **queries**, and added more **data**.

### 🔍 1. Expanded Queries
Added queries beyond the original three:
- `calcium in serum`
- `cells in urine`  
...including query variations like `calcium`, `urine`, `cells`, etc.

### 📦 2. Dataset Expansion
We queried **LOINC Search** for additional documents:
- bilirubin in plasma / bilirubin  
- calcium in serum / calcium  
- glucose in blood / glucose  
- leukocytes / white blood cells count  
- blood / urine / cells  

Saved results as CSVs.

---

## 📊 **Step 5: Model Evaluation**

We use multiple **metrics** to assess model performance:

| Metric           | Description                                           | Ideal Value |
|------------------|-------------------------------------------------------|-------------|
| **MSE**          | Mean Squared Error – lower is better                  | 0           |
| **R²**           | R-squared – explains variance, higher is better       | 1           |
| **Spearman's ρ** | Rank correlation – higher shows stronger ranking match| 1           |
| **NDCG**         | Normalized DCG – higher is better ranking quality     | 1           |

---

### 📉 **Dataset Performance Comparison**

| Dataset           | MSE    | R²      | Spearman ρ | NDCG   | Notes                          |
|-------------------|--------|---------|------------|--------|---------------------------------|
| **Basic**         | 0.1642 | -2.5187 | 0.7265     | 0.9086 | Initial 3 queries               |
| **First Enhanced**| 0.0479 | -1.9010 | 0.4700     | 0.8533 | Added `calcium in serum`        |
| **Second Enhanced**| 0.0461| -0.8984 | 0.6024     | 0.9421 | Added `bilirubin`, `glucose`, `leukocytes` |
| **Third Enhanced**| 0.0252 | -0.4765 | 0.4983     | 0.9398 | Added `blood`, `serum or plasma`|
| **Fourth Enhanced**| 0.0450| -1.4383 | 0.4323     | 0.9448 | Added `cells in urine`          |
| **Fifth Enhanced** | 0.0191| -0.6009 | 0.4615     | **0.9517** | Final version with `cells`, `urine` |

---

### 📌 **Per-Query NDCG (Fifth Dataset)**
- bilirubin in plasma: 0.9499
- calcium in serum: 0.9637
- cells in urine: 0.9448
- glucose in blood: 0.9663
- white blood cells count: 0.9339
