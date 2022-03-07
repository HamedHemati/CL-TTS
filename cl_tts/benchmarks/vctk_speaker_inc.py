import numpy as np
import os

from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDataset, \
    AvalancheSubset

from cl_tts.utils.ap import AudioProcessor
from cl_tts.benchmarks.datasets.multispeaker_dataset import MultiSpeakerDataset
from cl_tts.benchmarks.datasets.dataset_utils.text_processors.\
    eng.english_text_proessor import EnglishTextProcessor
from cl_tts.benchmarks.datasets.dataset_utils.collator import TTSColator


def get_vck_speakers_dict(datasets_root, dataset_name, metafile_name):
    dataset_path = os.path.join(datasets_root, dataset_name)
    meta_data_path = os.path.join(dataset_path, metafile_name)
    with open(meta_data_path) as file:
        all_lines = file.readlines()
    all_lines = [l.strip() for l in all_lines]
    # Only keep valid speakers
    all_lines = [l for l in all_lines]

    # List of speakers (used as targets)
    speakers = [l.split("|")[0] for l in all_lines]
    speaker_to_id = {spk: i for i, spk in enumerate(set(speakers))}
    return speaker_to_id


def get_vctk_speaker_incremental_benchmark(
        datasets_root,
        dataset_name,
        data_folder,
        metafile_name,
        file_format,
        speaker_lists,
        audio_params,
        reduction_factor
):
    # Audio processor
    audio_processor = AudioProcessor(audio_params)

    # Text processor
    transcript_processor = EnglishTextProcessor()

    speaker_to_id = get_vck_speakers_dict(datasets_root, dataset_name,
                                          metafile_name)
    speaker_datasets = []
    speaker_durations = []
    for speaker_list in speaker_lists:
        # Create dataset
        multispk_dataset = MultiSpeakerDataset(
            datasets_root,
            dataset_name,
            speaker_list,
            speaker_to_id,
            audio_processor,
            trim_margin_silence_thr=audio_params["trim_margin_silence_thr"],
            data_folder=data_folder,
            metafile_name=metafile_name,
            file_format=file_format,
            transcript_processor=transcript_processor
        )
        ds = AvalancheDataset(multispk_dataset)

        # Select indices for the target speaker
        indices = []
        for target_speaker in speaker_list:
            indices_i = np.where(np.array(ds.targets) == target_speaker)[0]
            indices.append(indices_i)
        indices = np.concatenate(indices)

        # Compute durations
        durations = multispk_dataset.get_durations(indices=indices)
        speaker_durations.append(durations)

        # Target speaker dataset
        ds_target_speaker = AvalancheSubset(ds, indices=indices)
        speaker_datasets.append(ds_target_speaker)

    benchmark = dataset_benchmark(speaker_datasets, speaker_datasets)

    collator = TTSColator(reduction_factor, audio_processor)

    # Speakers in each experience
    speakers_per_exp = []
    speakerids_per_exp = []
    for exp in benchmark.train_stream:
        speakers = set(exp.dataset.targets)
        speakers_per_exp.append(speakers)
        speaker_ids = [speaker_to_id[spk] for spk in speakers]
        speakerids_per_exp.append(speaker_ids)

    benchmark_meta = {
        "speaker_durations": speaker_durations,
        "transcript_processor": transcript_processor,
        "n_symbols": len(transcript_processor.symbols),
        "n_speakers_benchmark": len(speaker_list),
        "n_speakers_dataset": len(speaker_to_id.keys()),
        "collator": collator,
        "speakers_per_exp": speakers_per_exp,
        "speakerids_per_exp": speakerids_per_exp,
    }

    return benchmark, benchmark_meta
