import pandas
from os import listdir
import pickle

path = './Datasets/data/clean_tac/'
df = pandas.read_csv('./Datasets/data/all_accelerometer_data_pids_13.csv')
files = listdir(path)

for file in files:

    tacs = pandas.read_csv(path + file)
    pid = file.split('_')[0]
    previous_time = 0

    for i in range(len(tacs)):
        item = dict()
        item['tac'] = tacs.TAC_Reading[i]
        item['time'] = tacs.timestamp[i]*1000
        readings = df[ (df['pid'] == pid) & (df['time'] <= item['time']) & (df['time'] > previous_time)]
        previous_time = item['time']
        if len(readings)>0:
            item['readings'] = readings
            with open('./Datasets/data/Samples/' + pid + '_' + str(item['time']) + '.pkl', 'wb') as f:
                pickle.dump(item, f, pickle.HIGHEST_PROTOCOL)