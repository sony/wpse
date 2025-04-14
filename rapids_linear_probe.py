# Copyright © 2025 Sony Research Inc.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------
import argparse

import os
import numpy as np
import pandas as pd
import pickle
import time

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from cuml import LogisticRegression
from cuml.common.device_selection import set_global_device_type


def main():
    parser = argparse.ArgumentParser(description="Linear probing with CuML LogisticClassifier")
    parser.add_argument("workspace", type=str, help="a directory")
    parser.add_argument("task", type=str, help="dataset")
    parser.add_argument("--subdim", type=int, default=None, help="dimensions to be used")

    args = parser.parse_args()
    print(args)

    sub_dir = "model_cuml"
    if args.subdim is not None:
        sub_dir += f"_subdim{args.subdim}"

    os.makedirs(os.path.join(args.workspace, sub_dir), exist_ok=True)

    set_global_device_type("GPU")

    if args.task in ["aircraft", "pets", "caltech101", "flowers"]:
        metric = "mpca"
    else:
        metric = "accuracy"

    dump_train = np.load(os.path.join(args.workspace, "frozen_feats", f"{args.task}_train.npz"))
    dump_val   = np.load(os.path.join(args.workspace, "frozen_feats", f"{args.task}_val.npz"))
    dump_test  = np.load(os.path.join(args.workspace, "frozen_feats", f"{args.task}_test.npz"))

    trnval_features = np.concatenate((dump_train["x"], dump_val["x"]))
    trnval_labels   = np.concatenate((dump_train["y"], dump_val["y"]))
    test_features   = dump_test["x"]
    test_labels   = dump_test["y"]
    
    if args.subdim:
        sdim = args.subdim
        trnval_features = trnval_features[:,:sdim]
        test_features = test_features[:,:sdim]
    print(trnval_features.shape, test_features.shape)

    if args.task == "caltech101":
        # rebalancing data splits since some classes in caltech-101 contain only a small number of samples.
        trn_x, val_x, trn_y, val_y = train_test_split(
            trnval_features, trnval_labels,
            stratify=trnval_labels,
            test_size=0.1,
            random_state=42,
        )
    else:
        trn_x = dump_train["x"]
        trn_y = dump_train["y"]
        val_x = dump_val["x"]
        val_y = dump_val["y"]
        if args.subdim:
            sdim = args.subdim
            trn_x = trn_x[:,:sdim]
            val_x = val_x[:,:sdim]

    res_table, best_C, best_score = search_param(trn_x, trn_y, val_x, val_y, metric)
    print("search finished")
    print(f"best_C: {best_C}, best_score: {best_score}")

    list_idx = []
    list_C = []
    list_score = []
    for key, item in res_table.items():
        list_idx.append(key)
        list_C.append(item["C"])
        list_score.append(item["score"])
    
    table = pd.DataFrame.from_dict({
            "idx": list_idx,
            "C": list_C,
            "metric": [metric] * len(list_score),
            "score": list_score,
            "timestamp": [time.ctime()] * len(list_score),
        })
    table_filename = os.path.join(args.workspace, sub_dir, f"search_table_{args.task}.csv")
    if os.path.isfile(table_filename):
        table_prev = pd.read_csv(table_filename)
        table = pd.concat([table_prev, table])
    table.to_csv(table_filename, index=False)

    end = time.time()
    classifier, score = linear_probing(
        trnval_features, trnval_labels, test_features, test_labels, C=best_C, metric=metric)
    print(f"{metric}: {score:.3f}")
    print(f"Time: {time.time() - end}")
    
    filename = os.path.join(args.workspace, sub_dir, f"classifier_{args.task}_C{best_C}_{score:.3f}.pkl")
    pickle.dump(classifier, open(filename, "wb"))

    results = pd.DataFrame.from_dict({
            "task": [args.task],
            "metric": [metric],
            "score": [score],
            "timestamp": [time.ctime()],
        })
    csv_filename = os.path.join(args.workspace, sub_dir, "results_linear_probe.csv")
    if os.path.isfile(csv_filename):
        results_prev = pd.read_csv(csv_filename)
        assert {"task", "score", "timestamp"}.issubset(results_prev.columns)
        results = pd.concat([results_prev, results])
    results.to_csv(csv_filename, index=False)


def search_param(train_features, train_labels, test_features, test_labels, metric):
    space = 16
    rad = 3
    max_idx = 2 * rad * space
    min_idx = 0
    mid_point = rad * space

    list_C = np.logspace(-6, 6, num=max_idx+1)
    list_idx = np.arange(mid_point - rad * space, mid_point + rad * space + 1, space)

    res = {}
    while space >= 1:
        if mid_point - rad * space < min_idx:
            mid_point = rad * space
        if mid_point + rad * space > max_idx:
            mid_point = max_idx - rad * space
        list_idx = np.arange(mid_point - rad * space, mid_point + rad * space + 1, space)
    
        best_score = -1
        best_idx = None
        for idx in list_idx:
            if idx in res:
                score = res[idx]["score"]
            else:
                end = time.time()
                _, score = linear_probing(
                                train_features, train_labels,
                                test_features, test_labels,
                                C=list_C[idx], metric=metric)
                res[idx] = {
                    "C": list_C[idx],
                    "score": score,
                }
                print(f"{idx}, C: {list_C[idx]}, score: {score}")
                print(f"Time: {time.time() - end}")
            if best_score < score:
                best_idx = idx
                best_score = score

        space //= 2
        mid_point = best_idx

    return res, list_C[best_idx], best_score



def linear_probing(train_features, train_labels, test_features, test_labels, C=0.316, metric="accuracy"):
    if metric == "accuracy":
        class_weight = None
    elif metric == "mpca":
        class_weight = "balanced"
    else:
        raise ValueError(f"Invalid metric: {metric}")

    # Perform logistic regression
    classifier = LogisticRegression(C=C, max_iter=1000, class_weight=class_weight, verbose=1)
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    if metric == "accuracy":
        #accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
        score = accuracy(predictions, test_labels)
    else:  #mpca
        score = mean_per_class(predictions, test_labels)
    return classifier, score

def accuracy(output, target):
    return accuracy_score(target, output) * 100

def mean_per_class(output, target):
    conf_mat = confusion_matrix(target, output)
    per_classes = conf_mat.diagonal() / conf_mat.sum(axis=1)

    return 100 * per_classes.mean()

if __name__ == "__main__":
    main()