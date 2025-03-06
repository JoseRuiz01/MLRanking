# **Listwise Learning to Rank (LTR) for Lab Test Ranking**  

## **Overview**  
Listwise Learning to Rank (LTR) optimizes the **entire ranking order** for a given query, rather than comparing individual items (Pointwise) or pairs (Pairwise). This is particularly useful for ranking **lab tests** based on their relevance to queries like *"glucose in blood"*, *"bilirubin in plasma"*, and *"white blood cells count"*.  

---

## **Step 1: Define the Listwise LTR Model**  
Listwise LTR models learn a **ranking function** that orders a list of items to maximize evaluation metrics like **NDCG (Normalized Discounted Cumulative Gain)**.

### **How It Works**  
1. **Input**: A list of lab test results (documents) for a query.  
2. **Scoring Function**: A machine learning model predicts a **relevance score** for each test result.  
3. **Loss Function**: The model optimizes the ranking order using a loss function such as:
   - **ListMLE** (optimizes the likelihood of a correct ranking order)  
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
   - **Exact Match on Component**: +3 points.
   - **Partial Match on Component**: +2 points.
   - **Exact Match on System**: +2 points.
   - **Partial Match on System**: +1 point.
   - **No Match**: 0 points.

### **5. Normalize Scores**
   To standardize the scores across tests, normalize them to a scale (e.g., from 0 to 1). For example, if the highest score in the dataset is 5, the formula to normalize is:  
   - **Normalized Score** = \( score \ max_score \).

By following this method, each test is assigned a relevance score based on how well it matches the query's component and system. This system can be adjusted by fine-tuning the scoring weights to better suit specific applications and queries.

### **6. Save the new CSV**
Save into a new csv file the data with the calculated scores following this format:
| Query              | LOINC Code | Test Name                    | Relevance Score  |
|--------------------|-----------|------------------------------|--------------------|

### **7. Construct Training Lists**
Each query should have a **list of test results** with assigned relevance scores.  

```yaml
Query: "Glucose in blood"
- (14749-6, Glucose in Serum or Plasma, Relevance: 0.6)
- (35184-1, Fasting glucose, Relevance: 0.4)
- (15076-3, Glucose in Urine, Relevance: 0.4)
- (18906-8, Ciprofloxacin Susceptibility, Relevance: 0)
```

These lists will be fed into the model during training.  

---

## **Step 3: Implementing the Listwise LTR Model**  

We use **LightGBM (LambdaMART)**, which is fast, supports listwise ranking, and is easy to implement.

### **1. Install LightGBM**  
```bash
pip install lightgbm pandas numpy scikit-learn
```

### **2. Prepare the Dataset**  
Convert the dataset into **LightGBM format**:  

```python
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Sample dataset: Lab tests with features and relevance scores
data = pd.DataFrame({
    'query_id': [1, 1, 1, 1],  # Query "Glucose in blood"
    'feature1': [0.8, 0.6, 0.4, 0.1],  # Feature vectors (dummy values)
    'feature2': [0.7, 0.5, 0.3, 0.2],
    'relevance': [3, 2, 1, 0]  # Relevance scores
})

# Group data by query
query_group = data.groupby('query_id').size().tolist()

# Split dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Convert to LightGBM format
train_dataset = lgb.Dataset(train_data[['feature1', 'feature2']], label=train_data['relevance'], group=query_group)
test_dataset = lgb.Dataset(test_data[['feature1', 'feature2']], label=test_data['relevance'], group=query_group, reference=train_dataset)
```

### **3. Train the Model**  
```python
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1
}

# Train model
model = lgb.train(params, train_dataset, num_boost_round=100, valid_sets=[test_dataset], early_stopping_rounds=10)
```

### **4. Make Predictions**  
```python
# Sample test features for new queries
new_tests = pd.DataFrame({
    'feature1': [0.9, 0.3],
    'feature2': [0.8, 0.4]
})

# Predict ranking scores
predictions = model.predict(new_tests)
print(predictions)  # Higher scores mean more relevant lab tests
```

---

## **Step 4: Enhancing the Dataset**  

To improve model accuracy, we enhance the dataset with **better features** and **expanded queries**.

### **1. Feature Engineering**  

#### **Text-Based Features (TF-IDF, Embeddings)**
- Extract **TF-IDF** features from test names:  

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
text_features = vectorizer.fit_transform(["Glucose in blood", "Fasting glucose"])
```

- Use **BERT embeddings** for test descriptions:  

```python
from sentence_transformers import SentenceTransformer

bert_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_features = bert_model.encode(["Glucose in blood", "Fasting glucose"])
```

- Compute **cosine similarity** between query and document:  

```python
from sklearn.metrics.pairwise import cosine_similarity

query_embedding = bert_model.encode(["Glucose in blood"])
similarity_scores = cosine_similarity(query_embedding, embedding_features)
```

#### **Metadata Features**  
- **LOINC Code Presence** (Binary feature)
- **Source Reliability** (Weight based on source, e.g., PubMed vs. less authoritative sources)
- **Test Age** (Newer tests may be more relevant)  

```python
import datetime

def compute_age(date):
    return (datetime.datetime.now() - date).days

data['doc_age'] = data['publication_date'].apply(compute_age)
```

#### **User Interaction Features (if available)**  
- **Click-through rate (CTR)**
- **Dwell time** (time spent on a test result)
- **Past user preferences**  

```python
data['click_rate'] = data['num_clicks'] / data['num_impressions']
```


### **2. Expanding Queries**  
To cover more search variations, we add **synonyms, LOINC codes, and user-generated queries**.

#### **Synonyms & Variations**  
Use **medical ontologies** like **SNOMED-CT, UMLS** to find synonyms:  

```python
import nltk
from nltk.corpus import wordnet

def get_synonyms(term):
    synonyms = set()
    for syn in wordnet.synsets(term):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

print(get_synonyms("glucose"))
```

#### **Automated Query Generation with GPT-4**  
Use **GPT-4** or **T5 models** to create paraphrased queries:  

```python
from transformers import pipeline

paraphrase = pipeline("text2text-generation", model="t5-small")
generated_query = paraphrase("Paraphrase 'glucose in blood'", max_length=30)
print(generated_query)
```

---

## **Step 5: Evaluating the Model**  
We measure ranking quality using **NDCG**:  

```python
from sklearn.metrics import ndcg_score

true_relevance = [[3, 2, 1, 0]]
predicted_scores = [[0.9, 0.7, 0.4, 0.1]]

ndcg = ndcg_score(true_relevance, predicted_scores)
print("NDCG Score:", ndcg)
```

