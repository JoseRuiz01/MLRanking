# **Listwise Learning to Rank (LTR) for Lab Test Ranking**  
Listwise Learning to Rank (LTR) optimizes the **entire ranking order** for a given query, rather than comparing individual items (Pointwise) or pairs (Pairwise). This is particularly useful for ranking **lab tests** based on their relevance to queries like *"glucose in blood"*, *"bilirubin in plasma"*, and *"white blood cells count"*.  

---

## **Step 1: Define the Listwise LTR Model**  
Listwise LTR models learn a **ranking function** that orders a list of items to maximize evaluation metrics like **NDCG (Normalized Discounted Cumulative Gain)**.

### **How It Works**  
1. **Input**: A list of lab test results (documents) for a query.  
2. **Scoring Function**: A machine learning model predicts a **relevance score** for each test result.  
3. **Loss Function**: The model optimizes the ranking order using a loss function such as:
   - **eXtreme NDCG** (variant of the Normalized Discounted Cumulative Gain (NDCG) metric, designed to optimize ranking models directly)  
   - **LambdaRank** (optimizes for NDCG directly)  
4. **Output**: A ranked list of test results for the query.  

---

## **Step 2: Data Preparation**  

We implement a method for calculating relevance scores for lab tests based on a given query. The query consists of two main features: the **component** (e.g., "Glucose") and the **system** (e.g., "Blood"). 

### **1. Define Key Query Features**
   The query is parsed into two primary elements:
   - **Component**: The substance being measured (e.g., "Glucose").
   - **System**: The environment where the measurement takes place (e.g., "Blood").

### **2. Preprocess the Data**
   Each lab test in the dataset contains:
   - **Component**: The substance being measured (e.g., "Glucose").
   - **System**: The location or context of the test (e.g., "Blood", "Serum/Plasma").

### **3. Match Criteria**  
   To calculate the relevance score for each test:
   - **Exact Match**: The component or system exactly matches the query term (e.g., "Glucose" matches "Glucose").
   - **Partial Match**: The component or system contains terms closely related or synonymous to the query (e.g., "Glucose in Urine" partially matches "Glucose").
   - **System Match**: The system field matches the query's system (e.g., "Blood" matches "Blood").

### **4. Scoring Scheme**
   Points are awarded based on the match type:
   - **Exact Match on Component**:  weight(component) * weight(component)
   - **Partial Match on Component**: weight(component)/2 * weight(component)
   - **Exact Match on System**: weight(system) * weight(system)
   - **Partial Match on System**: weight(system)/2 * weight(system)
   - **No Match**: 0 points.

### **5. Normalize Scores**
   To standardize the scores across tests, normalize them to a scale (e.g., from 0 to 1). For example, if the highest score in the dataset is 5, the formula to normalize is:  
   - **Normalized Score** = \( score \ max_score \).

By following this method, each test is assigned a relevance score based on how well it matches the query's component and system. This system can be adjusted by fine-tuning the scoring weights to better suit specific applications and queries.

### **6. Save the new CSV**
Save into a new csv file the data with the calculated scores following this format:
| Query              | LOINC Code | Test Name          | Relevance Score  |
|--------------------|------------|--------------------|------------------|


---

## **Step 3: Implementing the Listwise LTR Model**  

We use **LightGBM**, which is fast, supports listwise ranking, and is easy to implement.

### **1. Prepare the Dataset**  
Before training, we need to format our dataset appropriately for LightGBM:

Query groups: LTR requires grouping rows by queries.
Features & Labels: Extract relevant numerical features and labels (e.g., Score).

### **2. Train the Model**  
We use eXtreme NDCG for listwise ranking with a custom objective function.

### **3. Evaluating the Model**
Mean Squared Error (MSE): 0.1550
R-squared (R²): -2.3101
Spearman's Rank Correlation: 0.7134
NDCG Mean Score: 0.8219
- NDCG for 'bilirubin in plasma': 0.7764
- NDCG for 'glucose in blood': 0.7507
- NDCG for 'white blood cells count': 0.9388

---

## **Step 4: Enhancing the Dataset**  

To improve model NDCG, we enhance the dataset with more **features** and **expanded queries**.

### **1. Expanding Queries**  
To cover more search variations, we add **user-generated queries**.

### **2. Expanding Dataset**

## **3. Evaluating the Model**  


**First Enhanced Dataset**
Mean Squared Error (MSE): 0.0285
R-squared (R²): -0.0209
Spearman's Rank Correlation: 0.6285
NDCG Mean Score: 0.8845
- NDCG for 'bilirubin in plasma': 0.8615
- NDCG for 'calcium in serum': 0.9149
- NDCG for 'cells in urine': 0.8793
- NDCG for 'glucose in blood': 0.9091
- NDCG for 'white blood cells count': 0.8577


