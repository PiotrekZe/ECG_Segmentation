import torch


class RunModel:
    def __init__(self, epochs, device, train_loader, test_loader):
        self.epochs = epochs
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
    
    def train_model(seld, model, criterion, optimizer, scheduler=None):
        model.train()
        running_loss = 0