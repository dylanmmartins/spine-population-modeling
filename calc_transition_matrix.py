import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def calculate_transition_matrix(file_paths, f, t):
    # file_paths is the list of hdf files
    # f is the 'from' state
    # t is the 'to' state
     
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
        
        # Iterate through the rows of the dataframe and check for f -> t transitions
        for i in range(0, len(df)-1):
            # Look for transitions from 'D' to 'P'
            if df.iloc[i]['stage'] == f and df.iloc[i+1]['stage'] == t:
                # Collect the string values for transition
                for col in [x for x in df.columns.values if x!='hrs' or x!='stage']:
                    transition_pair = (df.iloc[i][col], df.iloc[i+1][col])
                    all_transitions.append(transition_pair)

    state_options = ['NS','filopodium','thin','stubby','mushroom']
    state_to_idx = {s: i for i, s in enumerate(state_options)}

    # Initialize count matrix
    count_matrix = np.zeros((5, 5), dtype=np.float64)

    # Fill the count matrix
    for from_state, to_state in all_transitions:
        # Skip any transitions that do not match expected naming
        if (from_state not in state_options) or (to_state not in state_options):
            continue
        i = state_to_idx[from_state]
        j = state_to_idx[to_state]
        count_matrix[i, j] += 1

    # Normalize rows to get transition probabilities
    row_sums = count_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(count_matrix, row_sums, where=row_sums != 0)
    
    return transition_matrix


def main():

    base_path = '/Users/dmartins/Documents/GitHub/spine-population-modeling/clean_data'
    file_paths = sorted([os.path.join(base_path, x) for x in os.listdir('./clean_data')])
    file_paths = [x for x in file_paths if 'DS_Store' not in x]

    transition_matrix_DtoP = calculate_transition_matrix(file_paths, f='D', t='P')
    transition_matrix_PtoE = calculate_transition_matrix(file_paths, f='P', t='E')
    transition_matrix_EtoM = calculate_transition_matrix(file_paths, f='E', t='M')
    transition_matrix_MtoD = calculate_transition_matrix(file_paths, f='M', t='D')

    transition_matrix_list = [
        transition_matrix_DtoP,
        transition_matrix_PtoE,
        transition_matrix_EtoM,
        transition_matrix_MtoD
    ]

    state_options = ['NS','filopodium','thin','stubby','mushroom']

    title_list = [
        ['Diestrus','Proestrus'],
        ['Proestrus','Estrus'],
        ['Estrus','Metestrus'],
        ['Metestrus','Diestrus']
    ]
    for tind in range(4):

        print('visualizing for {}/4'.format(tind+1))

        name_pair = title_list[tind]

        fig_save_name = '{}to{}_transition.svg'.format(name_pair[0][0], name_pair[1][0])
        mat_save_name = '{}to{}_transition_matrix.npy'.format(name_pair[0][0], name_pair[1][0])

        transition_matrix = transition_matrix_list[tind]

        fig = plt.figure(figsize=(6,5),dpi=300)
        plt.imshow(
            transition_matrix.T,
            cmap='Blues',
            origin='lower'
        )
        for i in range(transition_matrix.shape[1]):
            for j in range(transition_matrix.shape[0]):
                text = plt.text(
                    j,
                    i,
                    '{:.2}'.format(transition_matrix[j,i]),
                    ha="center",
                    va="center",
                    color="k"
                )
        plt.xticks(range(len(state_options)), labels=state_options)
        plt.yticks(range(len(state_options)), labels=state_options)
        plt.title('{} to {}'.format(name_pair[0], name_pair[1]))
        plt.xlabel('{}'.format(name_pair[0]))
        plt.ylabel('{}'.format(name_pair[1]))
        plt.colorbar(label='transition probability')
        plt.tight_layout()
        fig.savefig(fig_save_name)

        np.save(mat_save_name, transition_matrix.T)

if __name__ == '__main__':

    main()
