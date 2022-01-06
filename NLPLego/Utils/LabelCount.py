def LabelsCount(filepath):
    labels_set = set()
    Index = {}
    filepath = filepath + "/test.tsv"
    with open(filepath, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            try:
                line = line.strip().split("\t")
                if line_id == 0:
                    for k,v in enumerate(line):
                        Index[v]=k
                    continue
                label = int(line[Index['label']])
                labels_set.add(label)
            except:
                pass
    label_size = len(labels_set)
    return label_size