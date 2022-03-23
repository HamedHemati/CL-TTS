from .cleaners import english_cleaners
from .symbols import symbols as _symbols


class EnglishTextProcessor:
    def __init__(self):
        self.symbols = _symbols

    def __call__(self, text):
        text = english_cleaners(text)
        encoded_text = [self.symbols.index(l) for l in text
                        if l in self.symbols]
        return encoded_text

    def process_for_inference(self, text):
        text = english_cleaners(text)
        encoded_text = [self.symbols.index(l) for l in text
                        if l in self.symbols]
        return encoded_text
