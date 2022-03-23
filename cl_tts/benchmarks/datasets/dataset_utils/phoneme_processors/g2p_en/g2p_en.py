from ..phonemizer_api.phonemize import phonemize
from ..symbol_list import symbol_list, _punctuations, _pad


class G2PEN:
    def __init__(self):
        # Set char list
        self.symbols = symbol_list
        self.punctutations = _punctuations

        # Char to id and id to char conversion
        self.char_to_id = {s: i for i, s in enumerate(self.symbols)}
        self.id_to_char = {i: s for i, s in enumerate(self.symbols)}

    def __call__(self, text):
        return self.phone_to_index_list(text)

    def process_for_inference(self, text):
        return self.text_to_phone(text)[0]

    def text_to_phone(self, text):
        """Converts text to phoneme."""
        ph = phonemize(
            text,
            strip=False,
            with_stress=True,
            preserve_punctuation=True,
            punctuation_marks=self.punctutations,
            njobs=1,
            backend='espeak',
            language="en-us",
            language_switch="remove-flags"
        )
        return ph

    def _should_keep_char(self, 
                          p):
        r"""Checks if char is valid and is not pad char."""
        return p in self.symbols and p not in [_pad]

    def phone_to_index_list(self, phones):
        r"""Converts list of phonemes to index list."""
        sequence = [self.char_to_id[s] for s in list(phones) if
                    self._should_keep_char(s)]
        return sequence
        
    def text_to_phone_to_index_list(self, text):
        """Converts text to sequence of indices."""
        sequence = []
        # Get the phoneme for the text
        phones = self.text_to_phone(text)
        
        # Convert each phone to its corresponding index
        sequence = [self.char_to_id[s] for s in list(phones) if
                    self._should_keep_char(s)]

        return sequence, phones

    def get_symbol_list(self):
        return self.symbols
