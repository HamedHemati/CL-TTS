import torch
import torch.nn.functional as F
import numpy as np


class TTSColator:
    """
    Implementation of collator class for TTS datasets
    """

    def __init__(self, reduction_factor, audio_processor):
        self.reduction_factor = reduction_factor
        self.ap = audio_processor

    def __call__(self, batch):
        """
        Prepares batch.

        :param batch: (transcript, waveform, speaker_id)
        :return: (batch_data, speaker_ids)
        """
        # Compute text lengths and
        trans_lengths = torch.LongTensor([len(t[0]) for t in batch])

        # Sort items w.r.t. the transcript length for RNN efficiency
        trans_lengths, ids_sorted_decreasing = torch.sort(
            trans_lengths, dim=0, descending=True)

        # Create list of batch items sorted by text length
        transcripts = [batch[idx][0] for idx in ids_sorted_decreasing]
        waveforms = [batch[idx][1] for idx in ids_sorted_decreasing]
        speaker_ids = [batch[idx][2] for idx in ids_sorted_decreasing]

        # Compute Mel-spectrograms
        melspecs = [self.ap.melspec(waveform) for waveform in waveforms]
        melspec_lengths = [mel.shape[-1] for mel in melspecs]

        # Create stop labels
        stop_targets = [torch.FloatTensor(np.array([0.] * (mel_len - 1) + [1.]))
                        for mel_len in melspec_lengths]

        # Pad and preprate tensors
        transcripts = self.pad_and_prepare_transcripts(transcripts)
        melspecs = self.pad_and_prepare_spectrograms(melspecs)
        stop_targets = self.pad_and_prepare_stoptargets(stop_targets)

        # Convert numpy arrays to PyTorch tensors
        melspec_lengths = torch.LongTensor(melspec_lengths)
        speaker_ids = torch.LongTensor(speaker_ids)

        batch_data = {
            "transcripts": transcripts,
            "trans_lengths": trans_lengths,
            "melspecs": melspecs,
            "melspec_lengths": melspec_lengths,
            "stop_targets": stop_targets
        }

        return batch_data, speaker_ids

    def pad_and_prepare_transcripts(self, transcripts):
        """
        Pads and prepares transcript tensors.

        :param transcripts: transcripts (list of LongTensor):
        list of 1-D tensors
        :return: torch.LongTensor: padded and concated tensor
        of size B x max_len
        """
        # Find max len
        max_len = max([len(x) for x in transcripts])

        # Pad transcripts in the batch
        def pad_transcript(x):
            return F.pad(x,
                         (0, max_len - x.shape[0]),
                         mode='constant',
                         value=0)

        padded_transcripts = [pad_transcript(x).unsqueeze(0) for x in
                              transcripts]

        return torch.cat(padded_transcripts, dim=0)

    def pad_and_prepare_spectrograms(self, inputs):
        """
        Pads and prepares spectrograms.
        :param inputs: list of 3-D spectrogram tensors of shape 1 x C x L
         where C is number of energy channels
        :return: tensor.FloatTensor: Padded and concatenated tensor
        of shape B x C x max_len
        """
        max_len = max([x.shape[-1] for x in inputs])
        remainder = max_len % self.reduction_factor
        max_len_red = max_len + (self.reduction_factor - remainder) \
            if remainder > 0 else max_len

        def pad_spectrogram(x):
            return F.pad(x,
                         (0, max_len_red - x.shape[-1], 0, 0, 0, 0),
                         mode='constant',
                         value=0.0)

        padded_spectrograms = [pad_spectrogram(x) for x in inputs]

        return torch.cat(padded_spectrograms, dim=0)

    def pad_and_prepare_stoptargets(self, inputs):
        """
        Pads and prepares stop targets.

        :param inputs: inputs (list of FloatTensor): list of 1-D
        tensors of shape L.
        :return: tensor.FloatTensor: Padded and concatenated tensor
        of shape B x max_len
        """
        max_len = max([x.shape[-1] for x in inputs])
        remainder = max_len % self.reduction_factor
        max_len_red = max_len + (self.reduction_factor - remainder) \
            if remainder > 0 else max_len

        def pad_stop_label(x):
            return F.pad(x,
                         (0, max_len_red - x.shape[-1]),
                         mode='constant',
                         value=1.0)

        padded_stops = [pad_stop_label(x).unsqueeze(0) for x in inputs]

        return torch.cat(padded_stops, dim=0)
