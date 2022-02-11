import unittest

from torch.utils.data import DataLoader
import numpy as np


from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDataset, \
    AvalancheSubset

from cl_tts.utils.ap import AudioProcessor
from cl_tts.utils.generic import load_params
from cl_tts.benchmarks.datasets.multispeaker_dataset import MultiSpeakerDataset
from cl_tts.benchmarks.datasets.dataset_utils.sampler import BinnedLengthSampler
from cl_tts.benchmarks.datasets.dataset_utils.collator import TTSColator
from cl_tts.benchmarks.datasets.dataset_utils.text_processors.\
    eng.english_text_proessor import EnglishTextProcessor


class MultiSpeakerDatasetTest(unittest.TestCase):
    def test_dataset(self):
        datasets_root = "./data/"
        dataset_name = "VCTK-mini"
        speaker_list = ["vctk_speaker335",
                        "vctk_speaker336",
                        "vctk_speaker339"]
        target_speaker = "vctk_speaker335"

        ap_params = load_params("hparams/speaker_incremental/naive.yml")[
            "ap_params"]

        audio_processor = AudioProcessor(ap_params)
        english_text_processor = EnglishTextProcessor()

        print("Creating an instance of MultiSpeakerDataset ...")
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

        print("Creating an instance of AvalancheDataset ...")
        ds = AvalancheDataset(multispk_dataset)

        # Select indices for the target speaker
        indices = np.where(np.array(ds.targets) == target_speaker)[0]

        print("Computing durations ...")
        durations = multispk_dataset.get_durations(indices=indices)

        print("Creating dataset subset for the target speaker ...")
        ds_speaker1 = AvalancheSubset(ds, indices=indices)

        print("Initializing dataloader ...")
        batch_size = 32
        bin_size = 64
        sampler = BinnedLengthSampler(durations, batch_size, bin_size)
        collator = TTSColator(1, audio_processor)
        dataloader = DataLoader(
            ds_speaker1,
            collate_fn=collator,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            drop_last=False,
            pin_memory=True,
            shuffle=False
        )

        for i, (batch, speakers) in enumerate(dataloader):
            break
        print("Dataloader worked successfully ...")
