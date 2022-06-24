import numpy as np
import csv

sample_rate = 32000
clip_samples = sample_rate * 5     # Audio clips are 10-second

# Load label
with open('./metadata/esc50_labels_indices.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    lines = list(reader)

labels = []
ids = []    # Each label has a unique id such as "/m/068hy"
for i1 in range(1, len(lines)):
    id = lines[i1][0]
    label = lines[i1][1]
    ids.append(id)
    labels.append(label)

classes_num = len(labels)

lb_to_ix = {label : i for i, label in enumerate(labels)}
ix_to_lb = {i : label for i, label in enumerate(labels)}

id_to_ix = {id : i for i, id in enumerate(ids)}
ix_to_id = {i : id for i, id in enumerate(ids)}
