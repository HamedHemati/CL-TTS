import os
import glob

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np


from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDataset, \
    AvalancheSubset

from cl_tts.utils.ap import AudioProcessor
from cl_tts.utils.generic import load_params
from .dataset_utils.sampler import BinnedLengthSampler
from .dataset_utils.collator import TTSColator
from .dataset_utils.text_processors.eng.english_text_proessor import \
    EnglishTextProcessor


class MultiSpeakerDataset(Dataset):
    def __init__(
            self,
            datasets_root,
            dataset_name,
            speaker_list,
            audio_processor,
            *,
            data_folder="audios",
            metafile_name="metadata.txt",
            file_format=None,
            transcript_processor=None
    ):
        self.datasets_root = datasets_root
        self.dataset_path = os.path.join(datasets_root, dataset_name)
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
        self.speaker_to_id = {spk: i for i, spk in enumerate(set(speakers))}

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
            NotImplementedError

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

        # TODO: add trim-silence

        return transcript, waveform, speaker_id

    def __len__(self):
        return len(self.data)


def get_multispeaker_dataloader():
    def get_dataloader(text_processor,
                       eval=False,
                       **params):
            # Get dataset
        dataset = TTSDataset(text_processor, eval=eval, **params)

        # Get transcript lengths for the sampler
        audio_durations = dataset.get_audio_durations()

        # Define sampler
        if params["sampler"] == "binned_length_sampler" and not eval:
            sampler = BinnedLengthSampler(audio_durations, params["batch_size"],
                                          params["batch_size"])
        else:
            sampler = None

        # Create dataloader
        collator = TTSColator(
            reduction_factor=params["model"]["n_frames_per_step"],
            audio_params=params["audio_params"])

        dataloader = DataLoader(dataset,
                                collate_fn=collator,
                                batch_size=batch_size,
                                sampler=sampler,
                                num_workers=params["num_workers"],
                                drop_last=False,
                                pin_memory=True,
                                shuffle=False)

        return dataloader, dataset.speaker_to_id


if __name__ == "__main__":
    datasets_root = "./data/"
    dataset_name = "VCTK-mini"
    speaker_list = ["vctk_speaker335", "vctk_speaker336", "vctk_speaker339"]

    ap_params = load_params("hparams/speaker_incremental/naive.yml")["ap_params"]
    audio_processor = AudioProcessor(ap_params)
    english_text_processor = EnglishTextProcessor()

    multispk_dataset = MultiSpeakerDataset(
        datasets_root,
        dataset_name,
        speaker_list,
        audio_processor,
        data_folder="audios",
        metafile_name="metadata.txt",
        file_format="wav",
        transcript_processor=english_text_processor
    )

    ds = AvalancheDataset(multispk_dataset)

    target_speaker = "vctk_speaker335"
    indices = np.where(np.array(ds.targets) == target_speaker)[0]
    durations = multispk_dataset.get_durations(indices=indices)

    ds_speaker1 = AvalancheSubset(ds, indices=indices)

    batch_size = 32
    bin_size = 64
    sampler = BinnedLengthSampler(durations, batch_size, bin_size)
    collator = TTSColator(1, audio_processor)

    dataloader = DataLoader(ds_speaker1,
                            collate_fn=collator,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=0,
                            drop_last=False,
                            pin_memory=True,
                            shuffle=False)

    for i, (batch, speakers) in enumerate(dataloader):
        print(batch["transcripts"].shape)
        print(batch["trans_lengths"].shape)
        print(batch["melspecs"].shape)
        print(batch["melspec_lengths"].shape)
        break
