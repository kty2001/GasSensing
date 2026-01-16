import os
import glob
import pickle

import numpy as np

SEED = 36
    


def build_samples():
    gas_to_label = {
        "acetone": 0,
        "benzene": 1,
        "toluene": 2,
    }

    df = {}
    # for path in glob.glob("data/pkl/*.pkl"):
    for path in glob.glob("data/del_rever/*.pkl"):
        filename = os.path.splitext(os.path.basename(path))[0]
        gas, data_type = filename.split("_", 1)

        if gas not in df:
            df[gas] = {}

        with open(path, "rb") as f:
            obj = pickle.load(f)

        df[gas][data_type] = obj

    samples = []
    labels = []

    for gas, label in gas_to_label.items():
        # merge = df[gas]["merge"].to_numpy()
        merge = df[gas]["filtered"].to_numpy()
        time = merge[:, 0]
        signal = merge[:, 1:]

        for i in range(signal.shape[1]):
            samples.append(signal[:, i].astype(np.float32))
            labels.append(label)

    x = np.stack(samples)
    num_classes = len(gas_to_label)
    y_index = np.array(labels)
    y_onehot = np.eye(num_classes, dtype=np.float32)[labels]

    print("x shape", x.shape)
    print("y_index shape", y_index.shape)
    print("y_onehot shape", y_onehot.shape)
    return x, y_index, y_onehot