import os
import io

import torch
from torchvision import transforms

from PIL import Image

from src.face_auth.utils.config import NOW_DATE
from src.config import BASE_DIR

from src.face_auth.utils.model import model


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


def predicate(content: bytes):
    img = FaceImage(content)

    load_model('best_model_BCEloss.pth')

    with torch.no_grad():
        model.eval()
        predicated = torch.round(model(img.get().unsqueeze(0)))

    return predicated


class FaceImage:
    def __init__(self, content: bytes):
        self.content = content
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def get(self) -> Image:
        image = Image.open(io.BytesIO(self.content))
        image = self.transform(image).cuda()
        return image


# keys = torch.load(BASE_DIR / 'models/keys.pth', weights_only=True)
# new_key = {f'{key}': hashlib.sha512(bytes.fromhex(bin_tens_to_hex(val))).hexdigest() for key, val in keys.items()}
# print(new_key)
# json.dump(new_key, open('some_keys.json', 'w', encoding='utf-8'), indent=4)
