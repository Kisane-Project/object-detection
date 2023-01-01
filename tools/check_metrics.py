import os, json

metric_path = '/SSDc/kisane_DB/metrics.json'
best_metric = '/SSDc/kisane_DB/best_metric.json'

json_data = []
iteration = None
best_bbox_ap = 0

with open(metric_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if 'bbox/AP' in data and data['bbox/AP'] > best_bbox_ap:
            best_bbox_ap = data['bbox/AP']
            iteration = data['iteration']
            best_data = data

with open(best_metric, 'w', encoding='utf-8') as f:
    json.dump(best_data, f, indent='\t')

print('best_bbox_ap : ', best_bbox_ap)
print('iteration : ', iteration)

print('\ndone')