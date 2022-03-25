import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio


class MultiSpeakerDataset(Dataset):
    def __init__(
            self,
            datasets_root,
            dataset_name,
            speaker_list,
            speaker_to_id,
            audio_processor,
            *,
            trim_margin_silence_thr=-1,
            data_folder="audios",
            metafile_name="metadata.txt",
            file_format=None,
            transcript_processor=None
    ):
        self.datasets_root = datasets_root
        self.speaker_to_id = speaker_to_id
        self.dataset_path = os.path.join(datasets_root, dataset_name)
        self.trim_margin_silence_thr = trim_margin_silence_thr
        self.data_folder = data_folder
        self.file_format = file_format
        self.transcript_processor = transcript_processor
        self.ap = audio_processor

        self.load_metadata(metafile_name, speaker_list)

    def load_metadata(self, metafile_name, speaker_list):
        meta_data_path = os.path.join(self.dataset_path, metafile_name)
        with open(meta_data_path) as file:
            all_lines = file.readlines()
        all_lines = [l.strip() for l in all_lines]
        # Only keep valid speakers
        all_lines = [l for l in all_lines if l.split("|")[0] in speaker_list]

        # List of speakers (used as targets)
        speakers = [l.split("|")[0] for l in all_lines]

        # List of file names
        file_names = [l.split("|")[1] for l in all_lines]
        if self.file_format is not None:
            file_names = [l + "." + self.file_format for l in file_names]

        # List of transcripts
        transcripts = [l.split("|")[2] for l in all_lines]

        # Set dataset targets and data to retrieve them in __getitem__
        self.targets = speakers
        self.data = list(zip(file_names, transcripts))

        # Wav paths
        self.wav_paths = [self.get_wav_path(speaker, file_name) for
                          (speaker, file_name) in zip(speakers, file_names)]

    def get_wav_path(self, speaker, file_name):
        wav_path = os.path.join(
            self.dataset_path,
            self.data_folder,
            speaker,
            file_name
        )
        return wav_path

    def load_waveform(self, file_path):
        ext = file_path.split(".")[-1]
        if ext in ["wav"]:
            return self.ap.load_audio(file_path)
        else:
            raise NotImplementedError

    def get_durations(self, indices):
        durations = []
        for idx in indices:
            audio_meta_data = torchaudio.info(self.wav_paths[idx])
            duration = audio_meta_data.num_frames / audio_meta_data.sample_rate
            durations.append(duration)

        return durations

    def __getitem__(self, index):
        speaker = self.targets[index]
        speaker_id = self.speaker_to_id[speaker]
        file_name, transcript = self.data[index]

        # Process transcript
        if self.transcript_processor:
            transcript = self.transcript_processor(transcript)
        transcript = torch.LongTensor(transcript)

        # Load waveform
        file_path = self.wav_paths[index]
        waveform = self.load_waveform(file_path)

        # Trim silence
        if self.trim_margin_silence_thr != -1:
            waveform = self.ap.trim_margin_silence(
                waveform[0],
                ref_level_db=self.trim_margin_silence_thr
            ).unsqueeze(0)

        return transcript, waveform, speaker_id

    def __len__(self):
        return len(self.data)
