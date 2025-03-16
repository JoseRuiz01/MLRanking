This document outlines the necessary libraries and steps to set up a Python environment to run the Jupyter notebooks with the required libraries for the project.

## 1. Create and Activate Virtual Environment

### For Windows:

1. Create a new virtual environment:
   ```bash
   python -m venv myenv
   ```

2. Activate the virtual environment:
   ```bash
   myenv\Scripts\activate
   ```

### For macOS/Linux:

1. Create a new virtual environment:
   ```bash
   python3 -m venv myenv
   ```

2. Activate the virtual environment:
   ```bash
   source myenv/bin/activate
   ```

## 2. Install Required Libraries

After activating the virtual environment, you can install all the required libraries by running the following command:

```bash
pip install pandas numpy scikit-learn scipy nltk sentence-transformers lightgbm
```

Alternatively, you can create a `requirements.txt` file with the following content and use it to install dependencies:

```txt
pandas==1.5.3
numpy==1.23.5
scikit-learn==1.1.2
scipy==1.9.3
nltk==3.7
sentence-transformers==2.2.0
lightgbm==3.3.2
```

Then install using:

```bash
pip install -r requirements.txt
```

## 3. Set Up Jupyter Notebook

If you don't have Jupyter Notebook installed, run the following command to install it:

```bash
pip install notebook
```

### 4. Datasets

To run the preprocessing code in `Preprocessing_basic.ipynb`, ensure that the basic Excel file, `loinc_dataset-v2.xlsx`, is present in the project folder.

Similarly, for executing `Preprocessing_enhanced.ipynb`, you will need a folder named `LOINC_Dataset`, where you can place any number of CSV files downloaded from the LOINC search (available at https://loinc.org/search). These CSV files will be processed to create a new scored dataset.