import copy
from dl_utils import *
from torchinfo import summary
from torch.nn import MSELoss, BCELoss
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
from optim.losses import PerceptualLoss
import os
import torch
import numpy as np
import logging
from time import time

class EarlyStopping():
    def __init__(self, patience=25, min_delta=10e-9):
        self.patience = patience
        self.min_delta = min_delta
        print(f"INFO: Early stopping delta {min_delta}")
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience} with {self.best_loss - val_loss}")
            if self.counter >= self.patience:
                self.counter = 0
                print('INFO: Early stopping')
                return True

class Trainer:
    def __init__(self, training_params, model, data, device):
        if 'checkpoint_path' in training_params:
            self.client_path = training_params['checkpoint_path']
            if not os.path.exists(self.client_path):
                os.makedirs(self.client_path)

        self.training_params = training_params
        self.train_ds, self.val_ds = data.train_dataloader(), data.val_dataloader()
        self.num_train_samples = len(self.train_ds) * self.train_ds.batch_size
        self.device = device
        self.model = model.train().to(self.device)
        self.test_model = copy.deepcopy(model.eval().to(self.device))
        
        patience = training_params['patience'] if 'patience' in training_params.keys() else 25
        self.early_stopping = EarlyStopping(patience=patience)
        
        opt_params = training_params['optimizer_params']
        self.optimizer = Adam(self.model.parameters(), **opt_params)

        self.lr_scheduler = None
        lr_sch_type = training_params['lr_scheduler'] if 'lr_scheduler' in training_params.keys() else 'none'

        if lr_sch_type == 'cosine':
            self.optimizer = Adam(self.model.parameters(), lr=training_params['optimizer_params']['lr'],
                                  amsgrad=True, weight_decay=0.00001)
            self.lr_scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=100)
        elif lr_sch_type == 'plateau':
            self.lr_scheduler = ReduceLROnPlateau(optimizer=self.optimizer, mode='min', factor=0.1)
        elif lr_sch_type == 'exponential':
            self.lr_scheduler = ExponentialLR(optimizer=self.optimizer, gamma=0.97)

        loss_class = import_module(training_params['loss']['module_name'],
                                   training_params['loss']['class_name'])
        self.criterion_rec = loss_class(**(training_params['loss']['params'])) \
            if training_params['loss']['params'] is not None else loss_class()

        if 'transformer' not in training_params.keys():
            self.transform = None
        else:
            transform_class = import_module(training_params['transformer']['module_name'],
                                            training_params['transformer']['class_name']) \
                if 'module_name' in training_params['transformer'].keys() else None

            self.transform = transform_class(**(training_params['transformer']['params'])) \
                if transform_class is not None else None

        self.criterion_MSE = MSELoss().to(device)
        self.criterion_PL = PerceptualLoss(device=device)
        self.min_val_loss = np.inf
        self.alfa = training_params['alfa'] if 'alfa' in training_params.keys() else 0

        self.best_weights = self.model.state_dict()
        self.best_opt_weights = self.optimizer.state_dict()

    def get_nr_train_samples(self):
        return self.num_train_samples

    def train(self, model_state=None, opt_state=None, epoch=0):
        raise NotImplementedError("[Trainer::train]: Please Implement train() method")

    def test(self, model_weights, test_data, task='Val', optimizer_weights=None, epoch=0):
        raise NotImplementedError("[Trainer::test]: Please Implement test() method")

class PTrainer(Trainer):
    def __init__(self, training_params, model, data, device):
        super(PTrainer, self).__init__(training_params, model, data, device)
        self.criterion = BCELoss()
        self.accumulation_steps = 8  # Simulate a batch size 8 times larger

    def train(self, model_state=None, opt_state=None, start_epoch=0):
        if model_state is not None:
            self.model.load_state_dict(model_state)
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)

        for epoch in range(start_epoch, self.training_params['nr_epochs']):
            self.model.train()
            epoch_loss = 0
            for i, data in enumerate(self.train_ds):
                kspace, target_mask, _ = data
                kspace, target_mask = kspace.to(self.device), target_mask.to(self.device)

                prediction = self.model(kspace)
                loss = self.criterion(prediction, target_mask)
                loss = loss / self.accumulation_steps  # Normalize the loss
                loss.backward()

                if (i + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_loss += loss.item() * self.accumulation_steps

            avg_loss = epoch_loss / len(self.train_ds)
            print(f'Epoch {epoch}, Training Loss: {avg_loss:.4f}')

            val_loss = self.validate(epoch)
            if val_loss < self.min_val_loss:
                self.min_val_loss = val_loss
                torch.save(self.model.state_dict(), f'{self.client_path}/best_model.pt')
                self.best_weights = self.model.state_dict()
                self.best_opt_weights = self.optimizer.state_dict()

            if self.early_stopping(val_loss):
                print("Early stopping triggered")
                break

            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

        return self.best_weights, self.best_opt_weights

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in self.val_ds:
                kspace, target_mask, _ = data
                kspace, target_mask = kspace.to(self.device), target_mask.to(self.device)

                prediction = self.model(kspace)
                loss = self.criterion(prediction, target_mask)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_ds)
        print(f'Epoch {epoch}, Validation Loss: {avg_val_loss:.4f}')
        return avg_val_loss

    def test(self, model_weights, test_data, task='Val', optimizer_weights=None, epoch=0):
        self.test_model.load_state_dict(model_weights)
        self.test_model.eval()
        test_loss = 0
        num_samples = 0

        with torch.no_grad():
            for data in test_data:
                kspace, target_mask, _ = data
                kspace, target_mask = kspace.to(self.device), target_mask.to(self.device)
                num_samples += kspace.size(0)

                prediction = self.test_model(kspace)
                loss = self.criterion(prediction, target_mask)
                test_loss += loss.item() * kspace.size(0)

        avg_test_loss = test_loss / num_samples
        print(f'{task} Loss: {avg_test_loss:.4f}')

        metrics = {
            f'{task.lower()}_loss': avg_test_loss,
        }

        return metrics, num_samples