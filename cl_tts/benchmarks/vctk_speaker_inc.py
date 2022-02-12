import numpy as np

from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDataset, \
    AvalancheSubset

from cl_tts.utils.ap import AudioProcessor
from cl_tts.benchmarks.datasets.multispeaker_dataset import MultiSpeakerDataset
from cl_tts.benchmarks.datasets.dataset_utils.text_processors.\
    eng.english_text_proessor import EnglishTextProcessor
from cl_tts.benchmarks.datasets.dataset_utils.collator import TTSColator


def get_vctk_speaker_incremental_benchmark(
        datasets_root,
        dataset_name,
        data_folder,
        metafile_name,
        file_format,
        speaker_list,
        audio_params,
        reduction_factor
):
    # Audio processor
    audio_processor = AudioProcessor(audio_params)

    # Text processor
    english_text_processor = EnglishTextProcessor()

    speaker_datasets = []
    speaker_durations = []
    for target_speaker in speaker_list:
        # Create dataset
        multispk_dataset = MultiSpeakerDataset(
            datasets_root,
            dataset_name,
            speaker_list,
            audio_processor,
            data_folder=data_folder,
            metafile_name=metafile_name,
            file_format=file_format,
            transcript_processor=english_text_processor
        )
        ds = AvalancheDataset(multispk_dataset)

        # Select indices for the target speaker
        indices = np.where(np.array(ds.targets) == target_speaker)[0]

        # Compute durations
        durations = multispk_dataset.get_durations(indices=indices)
        speaker_durations.append(durations)

        # Target speaker dataset
        ds_target_speaker = AvalancheSubset(ds, indices=indices)
        speaker_datasets.append(ds_target_speaker)

    benchmark = dataset_benchmark(speaker_datasets, speaker_datasets)

    collator = TTSColator(reduction_factor, audio_processor)

    benchmark_meta = {
        "speaker_durations": speaker_durations,
        "n_symbols": len(english_text_processor.symbols),
        "n_speakers": len(speaker_list),
        "collator": collator,
    }
    return benchmark, benchmark_meta
