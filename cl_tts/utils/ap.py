import torch
import torchaudio
import librosa

torchaudio.set_audio_backend("sox_io")


class AudioProcessor:
    def __init__(self,
                 params,
                 device=torch.device("cpu")):
        """
        Implementation of audio processor class. It is used to extract
        features from waveforms and convert features back to waveforms.

        :param params: dictionary of audio parameters
        :param device: device to run the transformations on (default: CPU)
        """
        self.params = params
        self.device = device

        # Spectrogram
        self.transform_spec = torchaudio.transforms.Spectrogram(
            n_fft=params["n_fft"],
            win_length=params["win_length"],
            hop_length=params["hop_length"],
            center=True,
            pad_mode="reflect",
            power=2.0).to(self.device)
        
        # Mel-Spectrogram
        self.transform_melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=params["sample_rate"],
            n_fft=params["n_fft"],
            win_length=params["win_length"],
            hop_length=params["hop_length"],
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm='slaney',
            onesided=True,
            n_mels=params["n_mels"],
            mel_scale="htk",
        )

        # Inverse Mel->Spec
        self.transform_inv_mel = torchaudio.transforms.InverseMelScale(
            n_stft=params["n_fft"] // 2 + 1,
            n_mels=params["n_mels"],
            sample_rate=params["sample_rate"],
            f_min=params["f_min"],
            f_max=params["f_max"],
            max_iter=1000,
            tolerance_loss=1e-10,
            tolerance_change=1e-10,
            mel_scale="htk",
            sgdargs={'lr': 10.9, "momentum": 0.9}
        )

        # Griffin-Lim
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=params["n_fft"],
            win_length=params["win_length"],
            hop_length=params["hop_length"]
        )

    def load_audio(self, audio_path):
        """

        :param audio_path: path to the wav file
        :return: Torch.Tensor
        """
        x, sr = torchaudio.load(audio_path)

        # Resample if sample rate of the input is different
        if sr != self.params["sample_rate"]:
            x = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.params["sample_rate"])(x)
        return x

    @staticmethod
    def trim_margin_silence(x, ref_level_db=26):
        r"""Trims margin silence of a waveform.

        Parameters:
        x (torch.tensor): input waveform
        ref_level_db: reference level in decibel

        Returns:
        torch.tensor: trimmed waveform
        """
        trimmed_x = librosa.effects.trim(x.numpy(),
                                         top_db=ref_level_db,
                                         frame_length=1024,
                                         hop_length=256)[0]
        x = torch.FloatTensor(trimmed_x)

        return x

    def spec(self, x):
        """
        Compute spectrogram of a waveform.
        :return:
        """
        return self.transform_spec(x)

    def melspec(self, x):
        """
        Computes mel-spectrogram a waveform.
        :param x:
        :return:
        """
        return self.transform_melspec(x)

    def mel_to_wav(self, mel):
        """
        Converts mel-spectrogram to waveform.
        :param mel: Tensor of shape B x C x L
        :return:
        """
        spec = self.transform_inv_mel(mel)
        return self.griffin_lim(spec)
