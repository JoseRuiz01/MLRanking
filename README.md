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


---

## **Step 3: Implementing the Listwise LTR Model**  

We use **LightGBM (LambdaMART)**, which is fast, supports listwise ranking, and is easy to implement.

### **1. Install LightGBM**  
```bash
pip install lightgbm pandas numpy scikit-learn
```

### **2. Prepare the Dataset**  
Before training, we need to format our dataset appropriately for LightGBM:

Query groups: LTR requires grouping rows by queries.
Features & Labels: Extract relevant numerical features and labels (e.g., Score).

### **3. Train the Model**  
We use LambdaMART for listwise ranking with a custom objective function.


### **4. Make Predictions**  


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


