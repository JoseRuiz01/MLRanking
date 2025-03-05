import pandas as pd

# Function to calculate relevance score
def calculate_score(query, component, system):
    score = 0
    
    # Extract the keywords from the query
    query_components = query.lower().split()  # Convert query to lowercase and split by spaces
    query_component, query_system = query_components[0], query_components[2]  # Extract component and system
    
    # Check for exact match on component
    if query_component in component.lower():
        score += 3  # Exact match on component
    
    # Check for partial match on component
    elif query_component in component.lower():
        score += 2  # Partial match on component

    # Check for exact match on system
    if query_system in system.lower():
        score += 2  # Exact match on system

    # Check for partial match on system
    elif query_system in system.lower():
        score += 1  # Partial match on system

    return score

# Read the dataset (replace with the actual path to your CSV)
df = pd.read_csv("lab_tests.csv")

# Query to match
query = "GLUCOSE IN BLOOD"

# Create a list to hold the result data
results = []

# Loop through each row in the DataFrame and calculate the relevance score
for index, row in df.iterrows():
    component = row['component']
    system = row['system']
    
    # Calculate the relevance score
    score = calculate_score(query, component, system)
    
    # Append results (query, LOINC code, name, and score)
    results.append([query, row['loinc_num'], row['long_common_name'], score])

# Create a DataFrame from the results list
results_df = pd.DataFrame(results, columns=["Query", "LOINC Code", "Name", "Score"])

# Save the results to a new CSV file
results_df.to_csv("relevance_scores.csv", index=False)

print("CSV with relevance scores has been saved as 'relevance_scores.csv'.")
