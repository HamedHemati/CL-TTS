import os
import glob
from torch.utils.data import Dataset, DataLoader


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
        self.speakers = speakers
        self.data = list(zip(file_names, transcripts))

    def load_waveform(self, file_path):
        ext = file_path.split(".")[-1]
        if ext in ["wav"]:
            return self.ap.load_audio(file_path)
        else:
            NotImplementedError

    def __getitem__(self, index):
        speaker = self.speakers[index]
        speaker_id = self.speaker_to_id[speaker]
        file_name, transcript = self.data[index]

        # Process transcript
        if self.transcript_processor:
            transcript = self.transcript_processor(transcript)

        # Load waveform
        file_path = os.path.join(
            self.dataset_path,
            self.data_folder,
            speaker,
            file_name
        )
        waveform = self.load_waveform(file_path)

        # TODO: add trim-silence

        return transcript, speaker_id, waveform

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    datasets_root = "./data/"
    dataset_name = "VCTK-mini"
    speaker_list = ["vctk_speaker335", "vctk_speaker336", "vctk_speaker339"]
    audio_processor = None
    multispk_dataset = MultiSpeakerDataset(
        datasets_root,
        dataset_name,
        speaker_list,
        audio_processor,
        data_folder="audios",
        metafile_name="metadata.txt",
        file_format="wav"
    )

    print(multispk_dataset.data)
    print(multispk_dataset.speaker_to_id)
    print("Number of samples: ", len(multispk_dataset))
