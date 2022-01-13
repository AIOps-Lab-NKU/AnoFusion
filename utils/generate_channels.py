from multiprocessing import Pool
import numpy as np
import time
import pandas as pd
import os
from numpy import array
import sys
sys.path.append('../serialize/')
from log_to_sequence import log_to_seq
from trace_to_sequence import trace_to_seq
sys.path.append('../')
import config
import json
import re
from collections import OrderedDict

data_path = '../data/'
store_linear_interpolation_data = '../update_linear_interpolation_data/'
store_serilize_data = '../serilize_data'
ground_truth_path = '../labeled_service/'

if not os.path.exists(store_serilize_data):
    os.mkdir(store_serilize_data)
if not os.path.exists(store_serilize_data + '/metrics'):
    os.mkdir(store_serilize_data + '/metrics')
if not os.path.exists(store_serilize_data + '/log'):
    os.mkdir(store_serilize_data + '/log')
if not os.path.exists(store_serilize_data + '/trace'):
    os.mkdir(store_serilize_data + '/trace')


def alignment(service_s, metric, log, trace):
    metric = metric[(metric['timestamp']>=config.start_time[service_s]) & (metric['timestamp']<=config.end_time[service_s])]
    log = log[(log['timestamp']>=config.start_time[service_s]) & (log['timestamp']<=config.end_time[service_s])]
    trace = trace[(trace['timestamp']>=config.start_time[service_s]) & (trace['timestamp']<=config.end_time[service_s])]
    
    metric = metric.drop_duplicates(['timestamp'])
    trace = trace.drop_duplicates(['timestamp'])
    log = log.drop_duplicates(['timestamp'])
    
    trace = trace[trace['timestamp'].isin(metric.timestamp.values)]
    log = log[log['timestamp'].isin(metric.timestamp.values)]
    
    while trace.shape[0]!=log.shape[0] or log.shape[0]!=metric.shape[0] or trace.shape[0]!=metric.shape[0]:
        min_length = min([metric.shape[0], trace.shape[0], log.shape[0]])
        if trace.shape[0] == min_length:
            print('1--')
            log = log[log['timestamp'].isin(trace.timestamp.values)]
            metric = metric[metric['timestamp'].isin(trace.timestamp.values)]
        elif log.shape[0] == min_length:
            print('2--')
            metric = metric[metric['timestamp'].isin(log.timestamp.values)]
            trace = trace[trace['timestamp'].isin(log.timestamp.values)]
        else:
            print('3--')
            log = log[log['timestamp'].isin(metric.timestamp.values)]
            trace = trace[trace['timestamp'].isin(metric.timestamp.values)]
        
    metric = metric.reset_index(drop=True)
    log = log.reset_index(drop=True)
    trace = trace.reset_index(drop=True)
    
    return log, metric, trace
    

def write_log_csv(service_s, name, temp, stru):
    stru = stru.drop_duplicates(['timestamp'])
    stru['timestamp'] = stru['timestamp'].astype(int)
    print("stru:", stru)
    temp = temp[temp['EventTemplate'].notnull()]
    temp = temp[temp['EventTemplate'].str.contains('INFO')]
    print("temp:", temp)
    log_series = dict({i:[] for i in temp['EventTemplate'].values.tolist()})
    log_series['total_log_length'] = []
    num_cores = len(log_series.keys())
    pool = Pool(processes=num_cores)  # multi-threading
    data_subsets = []
    for i in range(num_cores):
        key_name = list(log_series.keys())[i]
        data_subsets.append(pool.apply_async(log_to_seq, args=(key_name, temp, stru, config.start_time[service_s], config.end_time[service_s])))

    pool.close()
    pool.join()
    
    results = pd.DataFrame()
    count = 0
    for res in data_subsets:
        data = res.get()
        data_timestamp = []
        data_num = []
        for d in data:
            data_timestamp.append(d[0])
            data_num.append(d[1])
        results.loc[:, str(count)] = pd.Series(data_num)
        results.loc[:, "timestamp"] = pd.Series(data_timestamp)
        count += 1
    results.to_csv(store_serilize_data + '/logs/' + name + '.csv', index=False)


def write_trace_json(service_s, name, data):
    num_cores = 100
    pool = Pool(processes=num_cores)
    trace_data = []
    data_subsets = []
    begin = 0
    for j in range(num_cores):
        data_subsets.append(data[begin:begin+int(data.shape[0]//num_cores)])
        begin = begin+int(data.shape[0]//num_cores)
        
    for i in range(num_cores):
        trace_data.append(pool.apply_async(trace_to_seq, args=(data_subsets[i], config.start_time[service_s], config.end_time[service_s])))
    pool.close()
    pool.join()
    results = OrderedDict()
    for res in trace_data:
        results.update(res.get())
    trace_data = results.copy()
    trace_dict = {
        'version': "1.0",
        'results': trace_data,
        'explain': {
            'used': True,
            'details': "this is for josn test",
        }
    }

    json_str = json.dumps(trace_dict, indent=4)
    with open(store_serilize_data + '/traces/' + name, 'w') as json_file:
        json_file.write(json_str)
        
    print('*'*100)


def generate_sequence_data(name):
    # log data
    log_dataset = pd.read_csv(store_serilize_data + '/logs/' + name + '.csv')
    log_dataset = linear_interpolation(name, log_dataset, 1)
    log_dataset.to_csv(store_linear_interpolation_data + '/logs/' + name + '.csv', index=False)
    print("log_dataset.shape: ", log_dataset.shape)

    # trace data
    trace_json = pd.read_json(store_serilize_data + '/traces/' + name + '.json')
    trace_json = pd.DataFrame(trace_json)
    trace_dataset = []
    for value in trace_json['results'].values:
        if isinstance(value, float) or not isinstance(value, list):
            continue
        trace_dataset.append(array(value))
    trace_dataset = np.array(trace_dataset)
    trace_dataset = pd.DataFrame(trace_dataset)
    trace_dataset['timestamp'] = trace_json.index[:-2]
    trace_dataset = linear_interpolation(name, trace_dataset, 1)
    trace_dataset.to_csv(store_linear_interpolation_data + '/trace/' + name + '.csv', index=False)
    print("trace_dataset.shape: ", trace_dataset.shape)


def fill_missing_range(df, field, range_from, range_to, range_step=1, fill_with=0):
    return df.merge(how='right', on=field,
            right = pd.DataFrame({field:np.arange(range_from, range_to, range_step)}))\
                .sort_values(by=field).reset_index().fillna(fill_with).drop(['index'], axis=1)
      

def linear_interpolation(service_s, data, interval):
    data['timestamp'] = data['timestamp'].astype('int')
    data = data.drop_duplicates(['timestamp'])
    print("Raw data:", data)
    data = fill_missing_range(data, 'timestamp', config.start_time[service_s], config.end_time[service_s], interval, np.nan)
    data = data.interpolate()
    print("After data:", pd.DataFrame(data))
    return data
    

def write(service_s, store_path):
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    df_1 = pd.read_csv(data_path + service_s + '_2021-07-01_2021-07-15.csv')
    df_2 = pd.read_csv(data_path + service_s + '_2021-07-15_2021-07-31.csv')
    df = pd.concat([df_1, df_2], sort=True)
    df['timestamp'] = df['timestamp'].astype('int')
    df = df.drop_duplicates(['timestamp'])
    print(df)
    df = df[df['timestamp']<=config.end_time[service_s]]
    if 'ground_truth' in df.columns:
        df = df.drop(['ground_truth'], axis=1)
    if 'label' in df.columns:
        df = df.drop(['label'], axis=1)
    if 'run_fault' in df.columns:
        df = df.drop(['run_fault'], axis=1)
    df = df.loc[:, (df != 0).any(axis=0)]
    df.to_csv(store_linear_interpolation_data + '/metrics/' + service_s + '.csv', index=False)
    
    print("df ok")
    trace = pd.read_csv(data_path + 'trace_table_' + service_s.split('_')[0] + '_2021-07.csv')
    trace = trace.drop_duplicates(['timestamp'])
    if not os.path.exists(store_path + '/trace/'):
        os.mkdir(store_path + '/trace/')
    write_trace_json(service_s, service_s.split('_')[0] + '.json', trace)
   
    stru = pd.read_csv(data_path + service_s.split('_')[0] + '_stru.csv')
    temp = pd.read_csv(data_path + service_s.split('_')[0] + '_temp.csv')
    print("stru.timestamp:", stru['timestamp'].values[0], stru['timestamp'].values[-1])
    write_log_csv(service_s, service_s.split('_')[0], temp, stru)

     
def get_channels(service_s, dirname, proportion):
    write(service_s, store_linear_interpolation_data)
    generate_sequence_data(service_s.split('_')[0])
    
    trace = pd.read_csv(store_linear_interpolation_data + '/trace/' + service_s +'.csv')
    stru = pd.read_csv(store_linear_interpolation_data + '/log/' + service_s +'.csv')
    metric = pd.read_csv(store_linear_interpolation_data + '/metrics/' + service_s +'.csv')
    label_with_time = pd.read_csv(config.label_path + service_s + '.csv')
 
    align_log, align_metric, align_trace = alignment(service_s, metric, stru, trace)
    label_with_time = label_with_time[label_with_time['timestamp'].isin(align_metric['timestamp'].values)]
    print("align_metric:", align_metric.shape)
    print("align_log:", align_log.shape)
    print("align_trace:", align_trace.shape)
    if 'train' in dirname:
        align_metric = align_metric[:int(proportion*align_metric.shape[0])]
        align_log = align_log[:int(proportion*align_log.shape[0])]
        align_trace = align_trace[:int(proportion*align_trace.shape[0])]
        label_with_time = label_with_time[:int(proportion*len(label_with_time))]
        # timestamp = timestamp[:int(proportion*len(timestamp))]
    elif 'test' in dirname:
        align_metric = align_metric[int(proportion*align_metric.shape[0]):]
        align_log = align_log[int(proportion*align_log.shape[0]):]
        align_trace = align_trace[int(proportion*align_trace.shape[0]):]
        label_with_time = label_with_time[int(proportion*len(label_with_time)):]
        # timestamp = timestamp[int(proportion*len(timestamp)):]
        
    align_metric = align_metric.drop('timestamp', axis=1)
    align_log = align_log.drop('timestamp', axis=1)
    align_trace = align_trace.drop('timestamp', axis=1)
    print(align_metric, align_log, align_trace)
    return label_with_time, align_metric, align_log, align_trace
