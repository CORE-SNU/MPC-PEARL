import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import pandas as pd

# Make mean of mulitple files
filename = []
for i, val in enumerate(sys.argv):
    if i >= 1:    
        filename.append(val)

# Load csvs
data_sum = {'return_mean':[], 'return_std':[], 'reach_mean':[], 'reach_std':[], 'collision_free_success':[], 'collision_mean':[], 'collision_std':[], 'navi_mean':[], 'navi_std':[], 'length':[]}

for data_csv in filename:
    data = pd.read_csv(data_csv)
    data_sum['return_mean'].append(data['AverageReturn_all_test_tasks_col_free'])
    data_sum['return_std'].append(data['StdReturn_all_test_tasks_col_free'])
	
    data_sum['reach_mean'].append(data['AverageReaches_all_test_tasks_col_free'])
    data_sum['reach_std'].append(data['StdReaches_all_test_tasks_col_free'])
	
    data_sum['collision_free_success'].append(data['Collision_free_success_ratio'])
	
    data_sum['collision_mean'].append(data['AverageCollisionHuman_all_test_tasks'] + data['AverageCollisionTable_all_test_tasks'])
    data_sum['collision_std'].append(data['StdCollisionHuman_all_test_tasks'] + data['StdCollisionTable_all_test_tasks'])
    
    data_sum['navi_mean'].append(data['AverageNaviParts_all_test_tasks_col_free'])
    data_sum['navi_std'].append(data['StdNaviParts_all_test_tasks_col_free'])
    
    data_sum['length'].append(len(data))

    print('Filename : ',data_csv)
    print('Data length : ',len(data))
    print('')

# Smooth data
data_sum['return_smooth'], data_sum['reach_smooth'], data_sum['collision_free_success_smooth'],  data_sum['collision_smooth'],  data_sum['navi_smooth'] = [], [], [], [], []
window_width = 50
for i, name in enumerate(filename):
    length = data_sum['length'][i]
    data_sum['return_smooth'].append(np.zeros(length))
    data_sum['reach_smooth'].append(np.zeros(length))
    data_sum['collision_free_success_smooth'].append(np.zeros(length))
    data_sum['collision_smooth'].append(np.zeros(length))
    data_sum['navi_smooth'].append(np.zeros(length))
    for j in range(length):
        begin = max(j - window_width + 1, 0)
        end = j + 1
        data_sum['return_smooth'][i][j] = np.mean(data_sum['return_mean'][i][begin:end])
        data_sum['reach_smooth'][i][j] = np.mean(80. - data_sum['reach_mean'][i][begin:end])
        data_sum['collision_free_success_smooth'][i][j] = np.mean(data_sum['collision_free_success'][i][begin:end])
        data_sum['collision_smooth'][i][j] = np.mean(data_sum['collision_mean'][i][begin:end])
        data_sum['navi_smooth'][i][j] = np.mean(data_sum['navi_mean'][i][begin:end])
        
# Plot return
for i, name in enumerate(filename):
    data, data_std = data_sum['return_mean'][i], data_sum['return_std'][i]
    length = data_sum['length'][i]
    plt.plot(data, label=name)
    plt.fill_between(np.linspace(0,length,length), data+data_std, data-data_std, alpha=0.15)

plt.legend()
plt.ylabel('Average Return')
plt.xlabel('iters')
plt.xlim(0, max(data_sum['length'])-1)

plt.grid()
plt.tight_layout()
plt.savefig('result_return_summary.png')
plt.clf()

for i, name in enumerate(filename):
    data = data_sum['return_smooth'][i]
    plt.plot(data, label=name)

plt.legend()
plt.ylabel('Average Return')
plt.xlabel('iters')
plt.xlim(0, max(data_sum['length'])-1)

plt.grid()
plt.tight_layout()
plt.savefig('result_return_smooth_summary.png')
plt.clf()

# Plot arrival time
for i, name in enumerate(filename):
    data, data_std = data_sum['reach_mean'][i], data_sum['reach_std'][i]
    length = data_sum['length'][i]
    plt.plot(80-data, label=name)
    plt.fill_between(np.linspace(0,length,length), 80-data+data_std, 80-data-data_std, alpha=0.15)

plt.legend()
plt.xlabel('iters')
plt.ylabel('Arrival time (w/o collision consideration)')
plt.xlim(0, max(data_sum['length'])-1)

plt.grid()
plt.tight_layout()
plt.savefig('result_arrival_time_summary.png')
plt.clf()

for i, name in enumerate(filename):
    data = data_sum['reach_smooth'][i]
    plt.plot(data, label=name)

plt.legend()
plt.xlabel('iters')
plt.ylabel('Arrival time (w/o collision consideration)')
plt.xlim(0, max(data_sum['length'])-1)

plt.grid()
plt.tight_layout()
plt.savefig('result_arrival_time_smooth_summary.png')
plt.clf()

# Plot collisions
for i, name in enumerate(filename):
    data, data_std = data_sum['collision_mean'][i], data_sum['collision_std'][i]
    length = data_sum['length'][i]
    plt.plot(data, label=name)
    plt.fill_between(np.linspace(0,length,length), data+data_std, data-data_std, alpha=0.15)

plt.legend()
plt.xlabel('iters')
plt.ylabel('Average # of Collisions with Human')
plt.xlim(0, max(data_sum['length'])-1)

plt.grid()
plt.tight_layout()
plt.savefig('result_collision_summary.png')
plt.clf()

for i, name in enumerate(filename):
    data = data_sum['collision_smooth'][i]
    plt.plot(data, label=name)

plt.legend()
plt.xlabel('iters')
plt.ylabel('Average # of Collisions')
plt.xlim(0, max(data_sum['length'])-1)

plt.grid()
plt.tight_layout()
plt.savefig('result_collision_smooth_summary.png')
plt.clf()

# Plot navigation reward
for i, name in enumerate(filename):
    data, data_std = data_sum['navi_mean'][i], data_sum['navi_std'][i]
    length = data_sum['length'][i]
    plt.plot(data, label=name)
    plt.fill_between(np.linspace(0,length,length), data+data_std, data-data_std, alpha=0.15)

plt.legend()
plt.xlabel('iters')
plt.ylabel('Average Navi Return')
plt.xlim(0, max(data_sum['length'])-1)

plt.grid()
plt.tight_layout()
plt.savefig('result_navi_summary.png')
plt.clf()

for i, name in enumerate(filename):
    data = data_sum['navi_smooth'][i]
    plt.plot(data, label=name)

plt.legend()
plt.xlabel('iters')
plt.ylabel('Average Navi Return')
plt.xlim(0, max(data_sum['length'])-1)

plt.grid()
plt.tight_layout()
plt.savefig('result_navi_smooth_summary.png')
plt.clf()

# Plot task compeletion rate
for i, name in enumerate(filename):
    data = data_sum['collision_free_success'][i]
    length = data_sum['length'][i]
    plt.plot(data, label=name)

plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Task Compeletion Rate')
plt.xlim(0, max(data_sum['length'])-1)

plt.grid()
plt.tight_layout()
plt.savefig('result_collision_free_success_summary.png')
plt.clf()

for i, name in enumerate(filename):
    data = data_sum['collision_free_success'][i]
    plt.plot(data, label=name)

plt.legend()
plt.xlabel('iters')
plt.ylabel('Average # of Collisions with Table')
plt.xlim(0, max(data_sum['length'])-1)

plt.grid()
plt.tight_layout()
plt.savefig('result_collision_free_success_summary.png')
plt.clf()
