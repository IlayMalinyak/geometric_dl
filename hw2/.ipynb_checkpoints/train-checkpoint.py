import torch
import numpy as np
import time
import os
import yaml
from matplotlib import pyplot as plt
import glob
from collections import OrderedDict
from tqdm import tqdm
import torch.distributed as dist
import pytorch_lightning as pl 
from group_conv import GroupEquivariantCNN, CNN
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


model_dict = {
    'CNN': CNN,
    'GCNN': GroupEquivariantCNN
}

def create_model(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert False, f"Unknown model name \"{model_name}\". Available models are: {str(model_dict.keys())}"

class PlTrainer(pl.LightningModule):

    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = create_model(model_name, model_hparams)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        # AdamW is Adam with a correct implementation of weight decay (see here for details: https://arxiv.org/pdf/1711.05101.pdf)
        optimizer = optim.AdamW(
            self.parameters(), **self.hparams.optimizer_hparams)
        return [optimizer], []

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log('test_acc', acc, prog_bar=True)

        
class Trainer(object):
    """
    A class that encapsulates the training loop for a PyTorch model.
    """
    def __init__(self, model, optimizer, criterion, train_dataloader, device, num_classes=2, scheduler=None, val_dataloader=None,
                 optim_params=None, max_iter=np.inf, net_params=None, scaler=None, grad_clip=False,
                   exp_num=None, exp_name=None, plot_every=None, log_path=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler
        self.grad_clip = grad_clip
        self.num_classes = num_classes
        self.scheduler = scheduler
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.max_iter = max_iter
        self.device = device
        self.optim_params = optim_params
        self.net_params = net_params
        self.exp_num = exp_num
        self.exp_name = exp_name
        self.log_path = log_path
        self.best_state_dict = None
        self.plot_every = plot_every
        if not os.path.exists(f'{self.log_path}/exp{self.exp_num}'):
            os.makedirs(f'{self.log_path}/exp{self.exp_num}')
            with open(f'{self.log_path}/exp{exp_num}/net_params.yml', 'w') as outfile:
                yaml.dump(self.net_params, outfile, default_flow_style=False)
            with open(f'{self.log_path}/exp{exp_num}/optim_params.yml', 'w') as outfile:
                    yaml.dump(self.optim_params, outfile, default_flow_style=False)

    def fit(self, num_epochs, device,  early_stopping=None, best='loss'):
        """
        Fits the model for the given number of epochs.
        """
        min_loss = np.inf
        best_acc = 0
        train_loss, val_loss,  = [], []
        train_acc, val_acc = [], []
        self.optim_params['lr_history'] = []
        epochs_without_improvement = 0

        print(f"Starting training for {num_epochs} epochs with parameters: {self.optim_params}, {self.net_params}")
        for epoch in range(num_epochs):
            start_time = time.time()
            plot = (self.plot_every is not None) and (epoch % self.plot_every == 0)
            t_loss, t_acc = self.train_epoch(device, epoch=epoch, plot=plot,)
            t_loss_mean = np.mean(t_loss)
            train_loss.extend(t_loss)
            train_acc.append(t_acc)

            v_loss, v_acc = self.eval_epoch(device, epoch=epoch, plot=plot)
            v_loss_mean = np.mean(v_loss)
            val_loss.extend(v_loss)
            val_acc.append(v_acc)
            if self.scheduler is not None:
                self.scheduler.step(global_val_loss)
            criterion = min_loss if best == 'loss' else best_acc
            mult = 1 if best == 'loss' else -1
            objective = v_loss_mean if best == 'loss' else v_acc
            if mult*objective < mult*criterion:
                if best == 'loss':
                    min_loss = v_loss_mean
                else:
                    best_acc = v_acc
                # print("saving model...")
                torch.save(self.model.state_dict(), f'{self.log_path}/exp{self.exp_num}/checkpoint.pth')
                self.best_state_dict = self.model.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement == early_stopping:
                    print('early stopping!', flush=True)
                    break
            if epoch and epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss: {t_loss_mean:.6f}, Val Loss: {v_loss_mean:.6f}, Train Acc:  {t_acc:.4f}, Val Acc: {v_acc:.4f}, Time: {time.time() - start_time:.2f}s')
                print(os.system('nvidia-smi'))

            self.optim_params['lr'] = self.optimizer.param_groups[0]['lr']
            self.optim_params['lr_history'].append(self.optim_params['lr'])
        with open(f'{self.log_path}/exp{self.exp_num}/optim_params.yml', 'w') as outfile:
            yaml.dump(self.optim_params, outfile, default_flow_style=False)
        return {"num_epochs":num_epochs, "train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc}
    
    def train_epoch(self, device, epoch=None, plot=False, ):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        pbar = tqdm(self.train_dl)
        for i, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            self.optimizer.zero_grad()
            y_pred = self.model(x.float())
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            corrects = y_pred.argmax(-1).eq(y).sum().item()
            train_acc += corrects
            pbar.set_description(f"Epoch: {epoch}, train_loss:  {loss.item()}") 
            if i > self.max_iter:
                break
        # print("number of train_accs: ", train_acc)
        return train_loss, train_acc/len(self.train_dl.dataset)

    def eval_epoch(self, device, epoch=None,plot=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        pbar = tqdm(self.val_dl)
        for i,(x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            val_loss.append(loss.item())
            corrects = y_pred.argmax(-1).eq(y).sum().item()
            val_acc += corrects
            pbar.set_description(f"Epoch: {epoch}, val_loss:  {loss.item()}")
        return val_loss, val_acc/len(self.val_dl.dataset)

    def load_best_model(self, to_ddp=True, from_ddp=True):
        data_dir = f'{self.log_path}/exp{self.exp_num}'
        state_dict_files = glob.glob(data_dir + '/*.pth')
        print("loading model from ", state_dict_files[-1])
        
        state_dict = torch.load(state_dict_files[-1]) if to_ddp else torch.load(state_dict_files[0],map_location=device)
    
        if from_ddp:
            print("loading distributed model")
            # Remove "module." from keys
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    while key.startswith('module.'):
                        key = key[7:]
                new_state_dict[key] = value
            state_dict = new_state_dict
        # print("state_dict: ", state_dict.keys())
        # print("model: ", self.model.state_dict().keys())

        self.model.load_state_dict(state_dict, strict=False)


    def predict(self, test_dataloader, device, load_best=False, num_iter=np.inf):
        """
        Returns the predictions of the model on the given dataset.
        """
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        if load_best:
            self.load_best_model(from_ddp=False)
        self.model.eval()
        preds = np.zeros((0, self.num_classes))
        targets = np.zeros((0, self.num_classes))
        features = []
        for i,(x, y) in enumerate(test_dataloader):
            if i >= num_iter:
                break
            x = x.to(device)
            with torch.no_grad():
                y_pred = self.model(x, save_features=True)
                print(len(self.model.feature_maps))
                features.extend(self.model.feature_maps)
                self.model.clear_feature_maps()
                y_pred = y_pred.argmax(-1)
            print(preds.shape, y_pred.shape, targets.shape, y.shape)
            preds = np.concatenate((preds, y_pred.unsqueeze(1).cpu().numpy()))
            targets = np.concatenate((targets, y.unsqueeze(1).cpu().numpy()))
        print("target len: ", len(targets), "dataset: ", len(test_dataloader.dataset))
        return preds, targets, features

    