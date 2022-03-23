import torch
import wandb
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue

from cl_tts.utils.plot_utils import plot_spectrogram


class AudioSampleMetric(PluginMetric[float]):
    """
    This metric will return a `float` value after
    mini-batch.
    """

    def __init__(self, transcript_processor,
                 ap,
                 transcript,
                 speakerids_per_exp,
                 log_to_wandb=False,
                 synthesize_every=10):
        """
        Initialize the metric
        """
        super().__init__()
        self.transcript_processor = transcript_processor
        self.ap = ap
        self.transcript = transcript
        self.speakerids_per_exp = speakerids_per_exp
        self._log_to_wandb = log_to_wandb
        self._synthesize_every = synthesize_every
        self._iterator = -1

    def reset(self) -> None:
        """
        Reset the metric
        """
        self._loss_value = 0.0

    def result(self) -> float:
        """
        Emit the result
        """
        return None

    def after_training_epoch(
        self, strategy: "SupervisedTemplate"
    ) -> None:
        """
        Update the accuracy metric with the current
        predictions and targets
        """
        self._iterator += 1
        if self._iterator % self._synthesize_every == 0:
            self._synthesize_samples(strategy)

    def _synthesize_samples(self, strategy):
        current_exp = strategy.clock.train_exp_counter
        speakers_list = self.speakerids_per_exp[current_exp]
        for speaker in speakers_list:
            transcript = self.transcript
            inputs = self.transcript_processor.process_for_inference(transcript)
            inputs = torch.LongTensor(inputs).to(strategy.device).unsqueeze(0)
            input_lengths = torch.LongTensor(
                [len(inputs[0])]).to(strategy.device)
            speaker_ids = torch.LongTensor([speaker]).to(strategy.device)

            strategy.model.eval()
            with torch.no_grad():
                out = strategy.model.infer(inputs, input_lengths, speaker_ids)
            strategy.model.train()

            mel = out[0].squeeze(0).detach()

            if self._log_to_wandb:
                audio = self.ap.mel_to_wav(mel.unsqueeze(0).cpu())[0]
                step = strategy.clock.train_iterations
                wandb.log(
                    {f"Audio-Exp{current_exp}-{speaker}": wandb.Audio(
                        audio,
                        caption=transcript,
                        sample_rate=self.ap.params["sample_rate"])},
                    step=step)

                fig = plot_spectrogram(mel.cpu().numpy())
                wandb.log(
                    {f"Mel-Exp{current_exp}-{speaker}": fig},
                    step=step
                )

    def __str__(self):
        """
        Here you can specify the name of your metric
        """
        return "AudioSample"
