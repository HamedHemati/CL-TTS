from .speaker_incremental import get_vctk_speaker_incremental_benchmark,\
    get_ljspeech_single_speaker_benchmark


def get_benchmark(args, params):
    benchmark, benchmark_meta = None, None
    if params["benchmark"] == "VCTK-Speaker-Inc":
        benchmark, benchmark_meta = get_vctk_speaker_incremental_benchmark(
            params["datasets_root"],
            params["dataset_name"],
            params["data_folder"],
            params["metafile_name"],
            params["file_format"],
            params["speaker_lists"],
            params["ap_params"],
            params["model"]["n_frames_per_step"],
            params["input_type"]
        )
    elif params["benchmark"] == "LJSpeech-Single-Speaker":
        benchmark, benchmark_meta = get_ljspeech_single_speaker_benchmark(
            params["datasets_root"],
            params["dataset_name"],
            params["data_folder"],
            params["metafile_name"],
            params["file_format"],
            params["speaker_lists"],
            params["ap_params"],
            params["model"]["n_frames_per_step"],
            params["input_type"]
        )
    else:
        raise NotImplementedError()

    return benchmark, benchmark_meta
