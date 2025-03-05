import pandas as pd
import argparse


query_mapping = {
    "glucose in blood": {
        "component": "Glucose",
        "system": "Bld"
    },
    "bilirubin in plasma": {
        "component": "Bilirubin",
        "system": "Ser/Plas"
    },
    "white blood cells count": {
        "component": "Leukocytes",
        "system": "Bld"
    }
}


def calculate_score(query_component, query_system, component, system):
    """
    Function to calculate relevance score
    """
    
    score = 0
    # Exact match on component
    if query_component.lower() == component.lower():
        score += 3  
    
    # Partial match on component (synonym or closely related terms)
    elif query_component.lower() in component.lower():
        score += 2  

    # Exact match on system
    if query_system.lower() == system.lower():
        score += 2  

    # Partial match on system
    elif query_system.lower() in system.lower():
        score += 1  

    return score


def preprocess(excel_file):
    """
    Function to preprocess the excel file
    """
    
    results = []
    xl = pd.ExcelFile(excel_file)

    # Loop through each sheet in the Excel file (representing a different query)
    for sheet_name in xl.sheet_names:
        query_df = xl.parse(sheet_name, header=1)
        sheet_name = sheet_name.lower()
        
        # Check if the sheet name is in query_mapping
        if sheet_name in query_mapping:
            query_component = query_mapping[sheet_name]["component"]
            query_system = query_mapping[sheet_name]["system"]
        else:
            continue  # Skip if the sheet name doesn't match a query
        
        query_df.columns = query_df.columns.str.strip()  # Clean column names
        
        # Loop through each row in the sheet 
        for _, test_row in query_df.iterrows():
            component = test_row[2]
            system = test_row[3]
            
            # Calculate the relevance score for the query and lab test
            score = calculate_score(query_component, query_system, component, system)
            
            # Append the result (query, LOINC code, name, and score)
            results.append([sheet_name, test_row[0], test_row[1], score])

    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results, columns=["Query", "LOINC Code", "Name", "Score"])

    # Save the results to a new CSV file
    results_df.to_csv("dataset_with_scores.csv", index=False)
    print("CSV with relevance scores has been saved as 'dataset_with_scores.csv'.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate relevance scores for lab tests based on a query.")
    parser.add_argument('--dataset', type=str, help="Path to the Excel file containing the lab test data.")
    
    args = parser.parse_args()

    preprocess(args.dataset)
