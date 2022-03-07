from cl_tts.utils.ap import AudioProcessor
from .loss import LossMetric
from .audio_sample import AudioSampleMetric


def get_metrics(metric_names, params, benchmark_meta, log_to_wandb):
    metric_list = []
    for metric_name in metric_names:
        if metric_name == "loss":
            metric_list.append(LossMetric())
        elif metric_name == "audio_sample":
            transcript_processor = benchmark_meta["transcript_processor"]
            ap = AudioProcessor(benchmark_meta["ap_params"])
            metric_list.append(AudioSampleMetric(
                transcript_processor=transcript_processor,
                ap=ap,
                transcript=params["audio_sample_transcript"],
                speakerids_per_exp=benchmark_meta["speakerids_per_exp"],
                log_to_wandb=log_to_wandb,
                synthesize_every=params["synthesize_every"])
            )
        else:
            raise NotImplementedError()
    return metric_list
