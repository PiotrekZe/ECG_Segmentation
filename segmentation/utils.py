import numpy as np
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
import json
import os


# chat did that, but it just checks how many differences there is, so how preds differ from masks
def iou(target, pred, num_classes, epsilon=1e-6):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append((intersection / (union + epsilon)).item())

    return np.nanmean(ious)


def calculate_metrices(y_true, y_pred):
    acc, prec, rec, f1 = 0, 0, 0, 0
    for i in range(len(y_true)):
        acc += accuracy_score(y_true.cpu()[i], y_pred.cpu()[i])
        prec += precision_score(y_true.cpu()[i], y_pred.cpu()[i], zero_division=0, average='weighted')
        rec += recall_score(y_true.cpu()[i], y_pred.cpu()[i], zero_division=0, average='weighted')
        f1 += f1_score(y_true.cpu()[i], y_pred.cpu()[i], zero_division=0, average='weighted')
    return acc / len(y_pred), prec / len(y_pred), rec / len(y_pred), f1 / len(y_pred)

def save_model_results(path, lists, num):
    path_to_save = os.path.join(path, str(num))
    os.mkdir(path_to_save)
    list_train_loss = lists["train_loss"]
    list_train_accuracy = lists["train_accuracy"]
    list_train_recall = lists["train_recall"]
    list_train_precision = lists["train_precision"]
    list_train_f1 = lists["train_f1"]
    list_train_iou = lists["train_iou"]
    list_test_loss = lists["test_loss"]
    list_test_accuracy = lists["test_accuracy"]
    list_test_recall = lists["test_recall"]
    list_test_precision = lists["test_precision"]
    list_test_f1 = lists["test_f1"]
    list_test_iou = lists["test_iou"]
    for key, val in lists.items():
        with open(f"{path_to_save}/{key}.txt", "w") as f:
            for el in val:
                f.write(f"{el};")

def read_config(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data
