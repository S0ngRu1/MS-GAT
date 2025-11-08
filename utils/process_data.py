import re
from torchvision import transforms

def preprocess_text(sen):
    sentence = re.sub(r'<[^>]+>', '', sen)
    sentence = sentence.strip()
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence


def get_transforms():
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
        ]
    )