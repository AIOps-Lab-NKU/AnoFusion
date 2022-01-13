import pandas as pd

def log_to_seq(key, templates, structure, start_time, end_time):
    structure = pd.DataFrame(structure)
    print("structure:", structure.shape)
    print("structure.columns:", structure.columns)
    window_size = 60
    series = []
    
    for i in range(start_time, end_time):
        print("i:", i)
        struct_eventTemplate = structure[(structure['timestamp']>=i) & 
                                         (structure['timestamp']<i+window_size)]
        struct_eventTemplate = struct_eventTemplate['EventTemplate'].values
        struct_eventTemplate = struct_eventTemplate.tolist()
        
        if key == 'total_log_length':
            series.append((str(i), len(struct_eventTemplate)))
        else:
            if key not in struct_eventTemplate:
                series.append((str(i), 0))
            else:
                series.append((str(i), struct_eventTemplate.count(key)))

    return series
