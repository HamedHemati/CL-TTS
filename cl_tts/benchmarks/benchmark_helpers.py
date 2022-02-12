from .vctk_speaker_inc import get_vctk_speaker_incremental_benchmark


def get_benchmark(args, params):
    benchmark, benchmark_meta = None, None
    if params["benchmark"] == "VCTK-Speaker-Inc":
        benchmark, benchmark_meta = get_vctk_speaker_incremental_benchmark(
            params["datasets_root"],
            params["dataset_name"],
            params["data_folder"],
            params["metafile_name"],
            params["file_format"],
            params["speaker_list"],
            params["ap_params"]
        )

    else:
        raise NotImplementedError()

    return benchmark, benchmark_meta
