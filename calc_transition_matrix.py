import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_transition_matrix(file_paths):
    # If file_paths is a string (single file), convert it into a list for uniform handling
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    # Initialize an empty list to accumulate all the transitions across files
    all_transitions = []
    
    # Iterate through each file
    for file_path in file_paths:
        # Read the DataFrame from the file
        df = pd.read_hdf(file_path)
        df = df.replace('filopodia', 'filopodium')
        
        # Ensure the 'stage' column exists
        if 'stage' not in df.columns:
            raise ValueError(f"'stage' column is missing in {file_path}.")
        
        # Iterate through the rows of the dataframe and check for D -> P transitions
        for i in range(1, len(df)):
            # Look for transitions from 'D' to 'P'
            if df.iloc[i-1]['stage'] == 'D' and df.iloc[i]['stage'] == 'P':
                # Collect the string values for transition
                for col in [x for x in df.columns.values if x!='hrs' or x!='stage']:
                    transition_pair = (df.iloc[i-1][col], df.iloc[i][col])
                    all_transitions.append(transition_pair)
    
    # Convert the transitions list into a DataFrame to count occurrences
    transition_df = pd.DataFrame(all_transitions, columns=['from', 'to'])

    # Now, we will compute the transition probability matrix
    # Get all possible unique values for 'from' and 'to' columns
    unique_values = pd.concat([transition_df['from'], transition_df['to']]).unique()

    # Initialize the transition matrix with zeros
    transition_matrix = pd.DataFrame(0, index=unique_values, columns=unique_values)
    
    # Count transitions from 'from' -> 'to'
    for _, row in transition_df.iterrows():
        transition_matrix.at[row['from'], row['to']] += 1

    # Normalize the matrix by dividing by the row sums to get probabilities
    row_sums = transition_matrix.sum(axis=1)
    transition_matrix = transition_matrix.div(row_sums, axis=0)

    transition_matrix = transition_matrix.drop(labels=[x for x in transition_matrix.columns if x not in ['mushroom','stubby','filopodium','thin','NS']])
    transition_matrix = transition_matrix.drop(labels=[x for x in transition_matrix.columns if x not in ['mushroom','stubby','filopodium','thin','NS']], axis=1)
    # transition_matrix.drop([x for x in transition_matrix.columns.values if x not in ['mushroom','stubby','filopodium','thin','NS']],
    #                      0, inplace=True)
    
    return transition_matrix

if __name__ == '__main__':
    base_path = '/Users/dmartins/Dropbox/spine_modeling/clean_data/'
    file_paths = [os.path.join(base_path, x) for x in os.listdir('./clean_data')]
    transition_matrix = calculate_transition_matrix(file_paths).values[:,1:]

    np.save('/Users/dmartins/Dropbox/spine_modeling/DtoP_transition_matrix.npy', transition_matrix)

    # print(transition_matrix)
