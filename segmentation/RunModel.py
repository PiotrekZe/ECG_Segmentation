import numpy as np
import torch
import utils


class RunModel:
    def __init__(self, epochs, device, train_loader, test_loader, num_classes):
        self.epochs = epochs
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_classes = num_classes

    def train_model(self, model, criterion, optimizer):
        model.train()
        running_loss = 0
        iou_scores = []
        accuracy_tab, recall_tab, precision_tab, f1_score_tab = [], [], [], []
        for inputs, targets in self.train_loader:
            inputs = inputs.to(torch.float32).to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()

            batch_outputs = model(inputs)
            outputs = batch_outputs.permute(0, 2, 1).contiguous()
            outputs = outputs.view(-1, self.num_classes)

            loss = criterion(outputs, targets.view(-1))
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            preds = torch.argmax(batch_outputs, dim=1)
            iou_scores.append(utils.iou(targets, preds, self.num_classes))
            accuracy, precision, recall, f1_score = utils.calculate_metrices(targets, preds)
            running_loss = running_loss / len(self.train_loader)
            accuracy_tab.append(accuracy)
            precision_tab.append(precision)
            recall_tab.append(recall)
            f1_score_tab.append(f1_score)
        print(
            f"Training model. Loss: {running_loss}, Accuracy: {np.mean(accuracy_tab)}, IOU score: {np.mean(iou_scores)}")
        return (running_loss, np.mean(accuracy_tab), np.mean(precision_tab),
                np.mean(recall_tab), np.mean(f1_score_tab), np.mean(iou_scores))

    def test_model(self, model, criterion):
        model.eval()
        running_loss = 0
        iou_scores = []
        accuracy_tab, recall_tab, precision_tab, f1_score_tab = [], [], [], []
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(torch.float32).to(self.device)
                targets = targets.to(self.device)

                batch_outputs = model(inputs)
                outputs = batch_outputs.permute(0, 2, 1).contiguous()
                outputs = outputs.view(-1, self.num_classes)

                loss = criterion(outputs, targets.view(-1))
                running_loss += loss.item()

                preds = torch.argmax(batch_outputs, dim=1)
                iou_scores.append(utils.iou(targets, preds, self.num_classes))
                accuracy, precision, recall, f1_score = utils.calculate_metrices(targets, preds)
                running_loss = running_loss / len(self.train_loader)
                accuracy_tab.append(accuracy)
                precision_tab.append(precision)
                recall_tab.append(recall)
                f1_score_tab.append(f1_score)

        print(
            f"Testing model. Loss: {running_loss}, Accuracy: {np.mean(accuracy_tab)}, IOU score: {np.mean(iou_scores)}")
        return (running_loss, np.mean(accuracy_tab), np.mean(precision_tab),
                np.mean(recall_tab), np.mean(f1_score_tab), np.mean(iou_scores))
