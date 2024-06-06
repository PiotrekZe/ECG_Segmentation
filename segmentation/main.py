import Dataset
import CustomDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import RunModel
import Model
import utils


def main(num):
    config_data = utils.read_config("config_file.json")

    learning_rate = config_data['model']['learning_rate']
    weight_decay = config_data['model']['weight_decay']
    batch_size = config_data['model']['batch_size']
    epochs = config_data['model']['epochs']
    device = config_data['model']['device']
    num_channels = 2  # zależy ile jest sygnalow
    num_classes = 2  # dla segmentacji na peaki 2, ale jak będziemy szukać wszystkiego to zmiana

    path = config_data['file']['path']
    path_to_save = config_data['file']['path_to_save']

    dataset = Dataset.Dataset(path)
    X_train, X_test, y_train, y_test = dataset.read_dataset_peaks()

    train_dataset = CustomDataset.ECG(X_train, y_train)
    test_dataset = CustomDataset.ECG(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = Model.UNet(num_channels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    run_model = RunModel.RunModel(epochs, device, train_loader, test_loader, num_classes)

    (list_train_loss, list_train_accuracy, list_train_recall, list_train_precision,
     list_train_f1, list_train_iou) = [], [], [], [], [], []
    (list_test_loss, list_test_accuracy, list_test_recall, list_test_precision,
     list_test_f1, list_test_iou) = [], [], [], [], [], []

    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs}")
        train_running_loss, train_accuracy, train_precision, train_recall, train_f1_score, train_iou_scores = (
            run_model.train_model(model, criterion, optimizer))
        test_running_loss, test_accuracy, test_precision, test_recall, test_f1_score, test_iou_scores = (
            run_model.test_model(model, criterion))

        list_train_loss.append(train_running_loss)
        list_train_accuracy.append(train_accuracy)
        list_train_recall.append(train_recall)
        list_train_precision.append(train_precision)
        list_train_f1.append(train_f1_score)
        list_train_iou.append(train_iou_scores)

        list_test_loss.append(test_running_loss)
        list_test_accuracy.append(test_accuracy)
        list_test_recall.append(test_recall)
        list_test_precision.append(test_precision)
        list_test_f1.append(test_f1_score)
        list_test_iou.append(test_iou_scores)

    lists = {
        "train_loss": list_train_loss,
        "train_accuracy": list_train_accuracy,
        "train_recall": list_train_recall,
        "train_precision": list_train_precision,
        "train_f1": list_train_f1,
        "train_iou": list_train_iou,
        "test_loss": list_test_loss,
        "test_accuracy": list_test_accuracy,
        "test_recall": list_test_recall,
        "test_precision": list_test_precision,
        "test_f1": list_test_f1,
        "test_iou": list_test_iou
    }
    utils.save_model_results(path_to_save, lists, num)


if __name__ == "__main__":
    num = 0
    main(num)
