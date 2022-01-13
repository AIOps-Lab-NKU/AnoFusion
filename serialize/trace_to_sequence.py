from builtins import input, print, range
import numpy as np
import sys
sys.path.append('../')
import datetime
from collections import OrderedDict

def trace_to_seq(df, start_time, end_time):
    window_size = 1
    trace_series = OrderedDict()

    for i in df['timestamp'].values:
        trace_split_data = df[(df['timestamp']>=i) & (df['timestamp']<i+window_size)]
        span_data = []
        
        for m in range(trace_split_data.shape[0]):
            if '.' in trace_split_data['end_time'].values[m]:
                end = datetime.datetime.strptime(trace_split_data['end_time'].values[m], "%Y-%m-%d %H:%M:%S.%f")
            else:
                end = datetime.datetime.strptime(trace_split_data['end_time'].values[m], "%Y-%m-%d %H:%M:%S")
            if '.' in trace_split_data['start_time'].values[m]:
                start = datetime.datetime.strptime(trace_split_data['start_time'].values[m], "%Y-%m-%d %H:%M:%S.%f")
            else:
                start = datetime.datetime.strptime(trace_split_data['start_time'].values[m], "%Y-%m-%d %H:%M:%S")
            span_data.append((end- start).total_seconds())
            span_data.append(int(trace_split_data['status_code'].values[m]))
        
        # span_data = np.array(span_data)
        if len(span_data) != 0:
            span = span_data[0]
            span_mean = np.mean(span_data)
            span_ptp = np.ptp(span_data)
            span_std = np.std(span_data)
            span_25 = np.percentile(span_data, 25)
            span_75 = np.percentile(span_data, 75)
            values = [span, span_mean, span_ptp, span_std, span_25, span_75]
            trace_series[str(i)] = span_data
        else:
            trace_series[str(i)] = [0]*2
    
    return trace_series
