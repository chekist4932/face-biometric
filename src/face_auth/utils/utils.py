import os

import torch

from config import NOW_DATE
from src.config import BASE_DIR

from model import model


def save_model(model_name=f'{NOW_DATE}_model.pth'):
    dir_to_save = os.path.join(BASE_DIR / 'models', model_name)
    torch.save(model.state_dict(), dir_to_save)


def load_model(model_name='best_model.pth'):
    dir_to_load = os.path.join(BASE_DIR / 'models', model_name)
    model.load_state_dict(torch.load(dir_to_load, weights_only=True))


def bin_tens_to_hex(bin_tens: torch.Tensor) -> str:
    bin_line = ''.join([str(int(digit)) for digit in bin_tens.tolist()])
    octets = [bin_line[x:x + 8] for x in range(0, len(bin_line), 8)]

    hexs = [hex(int(octet, 2))[2:] for octet in octets]
    pad_hexs = [h if len(h) == 2 else '0' + h for h in hexs]
    hex_line = ''.join(pad_hexs)

    return hex_line
