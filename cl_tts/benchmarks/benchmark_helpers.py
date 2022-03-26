import os

from .speaker_incremental import get_vctk_spk_inc_benchmark


def get_benchmark(params, ds_path, config):

    if params["benchmark"] == "vctk-spk-inc":
        benchmark, benchmark_meta, config = get_vctk_spk_inc_benchmark(
            params["speaker_lists"],
            ds_path,
            config,
        )
    else:
        raise NotImplementedError()

    return benchmark, benchmark_meta, config
