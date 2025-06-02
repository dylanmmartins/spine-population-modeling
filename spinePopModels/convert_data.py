import os
import numpy as np
from scipy.io import loadmat
import pandas as pd

def convert_data(datadir, savedir):

    list_of_dendrites = sorted([f for f in os.listdir(datadir) if 'mat' in f])

    for f in list_of_dendrites:
        print('Converting {}'.format(f))
        d = loadmat(os.path.join(datadir, f))
        mouse_id = str(d['spine_data'][0][0][0][0])
        dendrite_id = int(d['spine_data'][0][0][1])
        stages = np.array([str(i[0]) for i in d['spine_data'][0][0][2][0]])

        spine_data = np.zeros([
            np.size(d['spine_data'][0][0][5], 0),
            np.size(d['spine_data'][0][0][5], 1)
        ])
        spine_data = spine_data.astype(str)
        for day in range(np.size(d['spine_data'][0][0][5], 0)):
            for spine_id in range(np.size(d['spine_data'][0][0][5], 1)):
                try:
                    spinetype = d['spine_data'][0][0][5][day,spine_id][0][0][0][0]
                    while type(spinetype) == np.ndarray:
                        spinetype = spinetype[0]
                except:
                    spinetype = 'NS'
                    
                spine_data[day, spine_id] = str(spinetype)

        df = pd.DataFrame(spine_data)
        df.columns = ['spine_{:02}'.format(n) for n in range(len(df.columns.values))]
        # drop last day for the only recording that a day was removed from spine data but not from cycle data.
        if (len(stages) > len(df.index.values)) and (mouse_id=='WTR042')  and (dendrite_id==4):
            stages = stages[:-1]

        df['hrs'] = np.arange(0, 12*len(df.index.values), 12)
        df['stage'] = stages
        df.attrs = {
            'mouse': mouse_id,
            'dendrite': dendrite_id-1
        }

        df.to_hdf(
            os.path.join(savedir, '{}.h5'.format(os.path.splitext(f)[0])),
            key='key'
        )

if __name__ == '__main__':

    datadir = '/Users/dmartins/Dropbox/spine_modeling/matlab_data'
    savedir = '/Users/dmartins/Dropbox/spine_modeling/clean_data'
    convert_data(datadir, savedir)