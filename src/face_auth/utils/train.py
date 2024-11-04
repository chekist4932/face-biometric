import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm

from model import model, device
from dataset import train_loader

from config import EPOCHS, WD, LR
from utils import save_model

torch.backends.cudnn.benchmark = True


def train_model(best_loss=float('inf')):
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.BCELoss().to(device)

    for epoch in range(EPOCHS):
        loss_val = 0.0
        model.train()

        with tqdm(total=train_loader.__len__(), position=0) as progress:
            for tensor_image, label in train_loader:
                tensor_image = tensor_image.to(device)
                label = label.to(device)
                optimizer.zero_grad()

                output = model(tensor_image)

                loss = loss_fn(output, label)
                if loss < best_loss:
                    best_loss = loss
                    save_model(model_name='best_model_BCEloss.pth')

                loss.backward()

                loss_item = loss.item()
                loss_val += loss_item

                optimizer.step()
                progress.set_description(f"Epoch: {epoch} | Training loss(iter): {str(loss_item)[:6]}")
                progress.update()

            progress.set_description(f"Epoch: {epoch} | Training loss(total): {str(loss_val / len(train_loader))[:6]}")
