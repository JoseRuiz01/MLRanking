# Basic Part
### 1. Choose a Type of MLR Approach
*You have three possible approaches to learning-to-rank (MLR)* 
- Pointwise: Treats ranking as a regression or classification problem for individual documents.  
- Pairwise: Considers pairs of documents and learns which one should be ranked higher.  
- Listwise: Uses entire ranked lists of documents and optimizes for global ranking metrics.  

*Your choice depends on the nature of your data and the final ranking quality you want*  
- Pointwise is simpler but may ignore relative rankings.  
- Pairwise balances ranking correctness without needing full list supervision.  
- Listwise is the most advanced but requires complete ground-truth rankings.  

### 2. Understand How It Works 
*Once you've chosen an approach, study its implementation details*
- What kind of loss function does it use? (e.g., MSE for Pointwise, hinge loss for Pairwise, NDCG optimization for Listwise)  
- What are the input features? (e.g., term frequency, document length, BM25 scores, etc.)  
- How does it handle relevance labels?  

### 3. Map a Set of Real-World Lab Tests to LOINC (Independent Subtask)
- Identify the relevant LOINC codes for lab tests related to your queries:  
  - Glucose in blood 
  - Bilirubin in plasma 
  - White blood cell count 
- This involves using resources like the [LOINC database](https://loinc.org/) to find standard codes.  
- Ensure mappings are accurate and standardized to avoid ambiguity in document retrieval.  

### 4. Build the Training Set  
*Given the three queries and their associated documents*
- Feature Engineering: Extract features for each document-query pair, such as:  
  - Term frequency (TF-IDF, BM25, embeddings)  
  - Metadata (document length, author, source)  
  - LOINC mappings (whether the document explicitly mentions the relevant test code)  
- Relevance Labeling: Assign relevance scores (e.g., scale of 0–3 or binary labels) to documents per query.  
- Training Data Format:  
  - Pointwise: Train a model with document-query pairs labeled independently.  
  - Pairwise: Convert the data into document pairs where one document is more relevant than another.  
  - Listwise: Rank entire sets of documents for each query and optimize using ranking metrics (e.g., NDCG).  



# Optional Part
### 5. Implement model (you can use public libraries)
*Since you're working on Learning-to-Rank (MLR), you can use libraries like*

 - LightGBM (LambdaMART) → lightgbm

 - XGBoost (Pairwise/Listwise Ranking) → xgboost

 - RankLib (Java-based, supports multiple ranking methods)
 
 - TF-Ranking (Deep Learning-based) → tensorflow_ranking

### 6. Extend dataset in # terms
*To improve model performance, add more ranking features*

 - Text Features
    TF-IDF, BM25, or word embeddings (e.g., BERT, FastText)
    Cosine similarity between query and document
    Length of document (shorter ones might be more relevant)
 
 - Metadata Features
    Presence of LOINC codes in the document
    Source of document (e.g., PubMed, clinical trials)
    Document publication date (newer docs may be more relevant)

 - User Interaction Features (if available)
    Click-through rate (CTR) for each document-query pair
    Dwell time (how long users stay on a document)
    Past user preferences

### 7. Extend dataset in queries
*Ways to Expand Queries*
 - Synonyms & Variations
    “glucose in blood” → “blood sugar test”, “serum glucose”
    “bilirubin in plasma” → “total bilirubin test”, “bilirubin blood test”
    Use SNOMED-CT, UMLS, or word embeddings for synonym expansion.
 
 - More LOINC-based Queries
    Search for additional LOINC codes related to the same concept
    Use query expansion techniques (e.g., spell correction, abbreviation expansion)
 
 - User-generated Queries
    If you have search logs, analyze real-world queries from doctors/patients

 - Automated Query Generation 
    GPT-4 or T5 models can generate paraphrased medical queries

# Submission
By deadline, submit a .zip file with training dataset files, report  stating the members of the group and what each one contributed to the work (in .pdf) up to 4 pages, presentation and any additional material used

