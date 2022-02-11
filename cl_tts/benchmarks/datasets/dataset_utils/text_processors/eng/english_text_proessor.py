from .cleaners import english_cleaners
from .symbols import symbols


class EnglishTextProcessor:
    def __call__(self, text):
        text = english_cleaners(text)
        encoded_text = [symbols.index(l) for l in text if l in symbols]
        return encoded_text
