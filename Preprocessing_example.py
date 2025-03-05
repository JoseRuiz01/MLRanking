import pandas as pd

# Function to calculate relevance score
def calculate_score(query_component, query_system, component, system):
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

excel_file = "loinc_dataset-v2.xlsx"
results = []
xl = pd.ExcelFile(excel_file)


# Loop through each sheet in the Excel file (representing a different query)
for sheet_name in xl.sheet_names:
    query_df = xl.parse(sheet_name, header=1)
    sheet_name = sheet_name.lower()
    print(sheet_name)
    # Extract the component and system for the current query from the query_mapping dictionary
    if sheet_name in query_mapping:
        query_component = query_mapping[sheet_name]["component"]
        query_system = query_mapping[sheet_name]["system"]
    else:
        continue  
    
    query_df.columns = query_df.columns.str.strip()
    print(query_df.columns)
    
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
results_df.to_csv("relevance_scores.csv", index=False)

print("CSV with relevance scores has been saved as 'relevance_scores.csv'.")
