
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 7
mpl.rcParams['svg.fonttype'] = 'none'



def find_start_ind(df):
    for i in range(len(df['stage'])):
        if df['stage'][i]=='D' and df['stage'][i+1]=='D':
            return i
    return np.nan


def resample_df(file_paths=None):

    # test spine turnover equation on the actual data to make sure i'm calculating it the same way nora did
    resampled_dfs = []

    if file_paths is None:
        file_paths = sorted([x for x in os.listdir('./clean_data') if '.h5' in x])
    file_paths, len(file_paths)

    for i, file_path in enumerate(file_paths):

        df = pd.read_hdf(os.path.join('clean_data', file_path))
        df = df.replace('filopodia', 'filopodium')

        row_value_counts = []
        for index, row in df.iterrows():
            row_value_counts.append(row.value_counts())
        counts = pd.DataFrame(row_value_counts).fillna(0)
        counts = counts[['filopodium','thin','stubby','mushroom']]
        counts.reset_index(inplace=True)
        counts['stage'] = df['stage'].copy()

        # for j in range(0, len(df)-1):
        #     if df.iloc[j]['stage'] == 'P' and df.iloc[j+1]['stage'] == 'E':
        #         ax.vlines(j*12, 0, 100, color='tab:grey', lw=1, ls='--')

        # startind = find_start_ind(df)

        # identify each cycles start
        cycle_id = []
        current_cycle = 0
        previous_stage = None

        for stage in counts['stage']:
            if previous_stage == 'M' and stage == 'D':
                current_cycle += 1
            cycle_id.append(current_cycle)
            previous_stage = stage

        counts['cycle'] = cycle_id

        # Container for results
        resampled_rows = []

        # Group by cycle and stage
        for (cycle, stage), group in counts.groupby(['cycle', 'stage']):
            group = group.reset_index(drop=True)
            
            # Get original positions and values
            x_original = np.linspace(0, 1, len(group))
            y_original1 = group['filopodium'].values
            y_original2 = group['thin'].values
            y_original3 = group['stubby'].values
            y_original4 = group['mushroom'].values

            # Interpolation function
            if len(group) == 1:
                # Only one value — duplicate it
                y_interp1 = [y_original1[0], y_original1[0]]
                y_interp2 = [y_original2[0], y_original2[0]]
                y_interp3 = [y_original3[0], y_original3[0]]
                y_interp4 = [y_original4[0], y_original4[0]]
            else:
                y_interp1 = scipy.interpolate.interp1d(x_original, y_original1)(np.linspace(0, 1, 2))
                y_interp2 = scipy.interpolate.interp1d(x_original, y_original2)(np.linspace(0, 1, 2))
                y_interp3 = scipy.interpolate.interp1d(x_original, y_original3)(np.linspace(0, 1, 2))
                y_interp4 = scipy.interpolate.interp1d(x_original, y_original4)(np.linspace(0, 1, 2))

            # Add interpolated rows
            for i in range(len(y_interp1)):
                resampled_rows.append({
                    'cycle': cycle,
                    'stage': stage,
                    'sample': i,
                    'filopodium': y_interp1[i],
                    'thin': y_interp2[i],
                    'stubby': y_interp3[i],
                    'mushroom': y_interp4[i]
                })

        resampled_df = pd.DataFrame(resampled_rows)

        resampled_dfs.append(resampled_df)


def turnover():

    resampled_dfs = resample_df()


    turnover_arr = []
    for i,x in enumerate(resampled_dfs):
        ss = []
        for t in range(1,len(x)):
            nt = x[['filopodium','thin','stubby','mushroom']].iloc[t].sum() - x[['filopodium','thin','stubby','mushroom']].iloc[t-1].sum()
            tot_avg_spines = x[['filopodium','thin','stubby','mushroom']].sum(axis=1).mean()
            ss.append(nt / tot_avg_spines)
        turnover_arr.append(ss)

    plt.figure(dpi=300, figsize=(6,3))
    plt.fill_betweenx(y=[-0.1,0.1], x1=4.75, x2=5.75, alpha=0.3, color='yellow')
    plt.fill_betweenx(y=[-0.1,0.1], x1=0.75, x2=1.75, alpha=0.3, color='yellow')
    plt.fill_betweenx(y=[-0.1,0.1], x1=-3, x2=-2, alpha=0.3, color='yellow')
    _plotbins= np.arange(-3.5,7,0.5)
    mean_turnover = np.zeros([len(_plotbins), len(turnover_arr)]) * np.nan
    micron_length = 51.8
    for i in range(len(turnover_arr)):
        index_for_cycle1_start = resampled_dfs[i].where(resampled_dfs[i]['cycle']==1).dropna().index.values[0]
        t_ = (np.arange(0, len(turnover_arr[i]))*0.5) - index_for_cycle1_start*0.5
        plt.plot(t_, np.array(turnover_arr[i])/(micron_length/10), color='tab:blue', alpha=0.5)
        for i_, t_i in enumerate(t_):
            ind_ = i_ + int(np.argwhere(_plotbins==t_[0])[0])
            mean_turnover[ind_, i] = (np.array(turnover_arr[i])/(micron_length/10))[i_]
    plt.xlabel('days')
    plt.plot(_plotbins, np.nanmean(mean_turnover,1), color='tab:red')
    # plt.xticks([-4,-2,0,2,4,6,8], labels=np.array([-4,-2,0,2,4,6,8])+4)
    plt.vlines(4,-0.1, 0.1, color='k')
    plt.vlines(1,-0.1, 0.1, color='k')
    plt.vlines(2,-0.1, 0.1, color='k')