import torch
from torch import nn
from pathlib import Path
from time import time
import os

class Trainer:
    def __init__(self, model, train_data, eval_data, log_steps, epochs, device, checkpoint_dir=None):
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data

        self.log_steps = log_steps
        self.epochs = epochs
        self.device = device

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None

        # Loss function
        self.criterion = nn.BCELoss(reduction='mean').to(self.device)
        # Optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def train(self, epoch):
        print(f"Begin training epoch {epoch}")

        # Put the model to training mode
        self.model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0.0
        epoch_train_count = 0.0

        logging_loss = 0.0
        logging_correct = 0.0
        logging_count = 0.0
        stime = time()
        for i, lattice in enumerate(self.train_data):
            # Get input data
            targets = torch.tensor(lattice.target_nodes).view(-1, 1).float()
            targets = targets.to(self.device)

            # Reset gradients
            self.optimizer.zero_grad()

            # Get outputs from the model
            outputs = self.model(lattice, device=self.device)

            # Compute the loss
            loss = self.criterion(outputs, targets)

            # Back-propagation
            loss.backward()

            # Gradient descent
            self.optimizer.step()

            # Update the statistics
            logging_loss += loss.item()
            predicted_targets = (outputs>0.5).long()
            logging_correct += (predicted_targets == targets).sum().item()
            logging_count += targets.numel()

            # Logging
            if (i + 1) % self.log_steps == 0:
                accuracy = logging_correct*100/logging_count
                print(f"Epoch {epoch} | Step {i+1}/{len(self.train_data)} | Run time: {time()-stime:.2f}s | Loss: {logging_loss/self.log_steps:.3f} | Accuracy: {accuracy:.2f}%")
                logging_loss = 0.0
                logging_correct = 0.0
                logging_count = 0.0
            # update statistics
            epoch_train_loss += loss.item()
            epoch_train_correct += (predicted_targets == targets).sum().item()
            epoch_train_count += targets.numel()
        accuracy = epoch_train_correct*100/epoch_train_count
        print(f"Epoch {epoch} | Training | Run time: {time()-stime:.2f}s  | Loss: {epoch_train_loss/len(self.train_data):.2f} | Accuracy: {accuracy:.2f}%")
        return epoch_train_loss/len(self.train_data)

    def eval(self, epoch):
        print(f"Begin evaluate epoch {epoch}")

        # Put the model to eval mode
        self.model.eval()
        epoch_eval_loss = 0.0
        epoch_eval_correct = 0.0
        epoch_eval_count = 0.0
        for i, lattice in enumerate(self.eval_data):
            # Get input data
            targets = torch.tensor(lattice.target_nodes).view(-1, 1).float()
            targets = targets.to(self.device)

            #Get prediction:
            with torch.no_grad():
                outputs = self.model(lattice, device=self.device)
                epoch_eval_loss += self.criterion(outputs, targets)
            predicted_targets = (outputs>0.5).long()
            epoch_eval_correct += (predicted_targets == targets).sum().item()
            epoch_eval_count += targets.numel()
        accuracy = epoch_eval_correct/epoch_eval_count*100
        print(f"Epoch {epoch} | Validation | Loss: {epoch_eval_loss/len(self.eval_data):.2f} | Accuracy: {accuracy:.2f}%")
        return epoch_eval_loss/len(self.eval_data)

    def save_checkpoint(self, epoch, loss, eval_loss):
        if not self.checkpoint_dir:
            return

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        name = f"checkpoint_epoch{epoch}.pt"
        print(f"Saving model checkpoint epoch {epoch} to {self.checkpoint_dir.joinpath(name)}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'eval_loss': eval_loss,
        }, self.checkpoint_dir.joinpath(name))

        with open(self.checkpoint_dir.joinpath("checkpoint_last.txt"), "w") as f:
            f.write(name)

    def load_checkpoint(self, path: Path):
        print(f"Loading model checkpoint from {path}")
        checkpoint = torch.load(path)
        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded at epoch {self.current_epoch}.")

    def __call__(self):
        if self.checkpoint_dir and os.path.exists(self.checkpoint_dir.joinpath("checkpoint_last.txt")):
            with open(self.checkpoint_dir.joinpath("checkpoint_last.txt")) as f:
                name = f.readline().strip()
            self.load_checkpoint(self.checkpoint_dir.joinpath(name))
            start_epoch = self.current_epoch + 1
        else:
            start_epoch = 0

        for self.current_epoch in range(start_epoch, self.epochs):
            loss = self.train(self.current_epoch)
            eval_loss = self.eval(self.current_epoch)
            self.save_checkpoint(epoch=self.current_epoch, loss=loss, eval_loss=eval_loss)