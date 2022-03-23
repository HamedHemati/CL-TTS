import os
import torch
from tqdm import tqdm
import wandb
import numpy as np
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from cl_tts.benchmarks.datasets.dataset_utils.sampler import BinnedLengthSampler
from cl_tts.utils.plot_utils import plot_spectrogram, plot_attention
from cl_tts.utils.ap import AudioProcessor


class Trainer(BaseTrainer):
    """
    Base class Trainer. All trainers should inherit from this class.
    """
    def __init__(self, args, params, experiment_name):
        super().__init__(args, params, experiment_name)

        self.model.to(self.device)
        self.log_to_wandb = args.wandb_proj != ""
        self.ap = AudioProcessor(self.benchmark_meta["ap_params"])
        if self.log_to_wandb:
            wandb.init(
                project=self.args.wandb_proj,
                config=self.params,
                name=experiment_name)

    def run(self):
        dataset = self.benchmark.train_stream[0].dataset
        durations = self.benchmark_meta["durations_per_exp"][0]
        sampler = BinnedLengthSampler(durations,
                                      self.params["train_mb_size"],
                                      self.params["train_mb_size"])

        dataloader = DataLoader(
            dataset,
            collate_fn=self.benchmark_meta["collator"],
            batch_size=self.params["train_mb_size"],
            sampler=sampler,  # For now no sampler is supported
            num_workers=0,
            drop_last=False,
            pin_memory=True,
            shuffle=False
        )

        self.global_step = 0

        for epoch in range(self.params["train_epochs"]):
            pbar = tqdm(dataloader)

            epoch_losses = []
            for mbatch in pbar:
                self.mbatch_to_device(mbatch)
                inputs, speaker_ids = mbatch
                self.optimizer.zero_grad()
                out = self.forward_func(self.model, inputs, speaker_ids)
                loss = self.criterion_func(out, inputs, speaker_ids)
                loss.backward()
                self.optimizer.step()

                # Logging
                pbar.set_description(
                    f"Epoch: {epoch} - Loss: {loss.item():.2f}")
                if self.log_to_wandb:
                    wandb.log({"Step Loss": loss.item()}, step=self.global_step)
                epoch_losses.append(loss.item())

                self.global_step += 1

            if self.log_to_wandb:
                epoch_loss = np.mean(epoch_losses)
                wandb.log({"Epoch Loss": epoch_loss}, step=self.global_step)

            if self.args.save_results:
                if epoch % 5 == 0:
                    self.save_checkpoint(epoch)

            if epoch % self.params["synthesize_every"] == 0:
                self.synthesize_samples(current_epoch=epoch)

        if self.log_to_wandb:
            wandb.finish()

    def mbatch_to_device(self, mbatch):
        """Move to device"""
        for k in mbatch[0].keys():
            mbatch[0][k] = mbatch[0][k].to(self.device)
        mbatch[1] = mbatch[1].to(self.device)

    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.checkpoints_path,
                                       f"ckpt_epoch_{epoch}.pt")
        torch.save(self.model.state_dict(), checkpoint_path)

    def load_checkpoint(self):
        # Load checkpoint
        print(f"Loading checkpoint from  " + \
              f"{self.params['load_checkpoint_path']}")
        ckpt = torch.load(self.params["load_checkpoint_path"],
                          map_location=self.device)
        for name, param in self.model.named_parameters():
            try:
                self.model.state_dict()[name].copy_(ckpt[name])
            except:
                print(f"Could not load weights for {name}")

    def synthesize_samples(self, current_epoch):
        transcript_processor = self.benchmark_meta["transcript_processor"]
        transcript = self.params["audio_sample_transcript"]
        speakers_list = self.benchmark_meta["speakerids_per_exp"][0]

        for speaker in speakers_list:
            inputs = transcript_processor(transcript)
            inputs = torch.LongTensor(inputs).to(self.device).unsqueeze(0)
            input_lengths = torch.LongTensor(
                [len(inputs[0])]).to(self.device)
            speaker_ids = torch.LongTensor([speaker]).to(self.device)

            self.model.eval()
            with torch.no_grad():
                out = self.model.infer(inputs, input_lengths, speaker_ids)
            self.model.train()

            mel = out[0].squeeze(0).detach()
            attn = out[2].squeeze(0).detach()

            if self.log_to_wandb:
                audio = self.ap.mel_to_wav(mel.unsqueeze(0).cpu())[0]
                step = self.global_step
                wandb.log(
                    {f"Audio": wandb.Audio(
                        audio,
                        caption=transcript,
                        sample_rate=self.ap.params["sample_rate"])},
                    step=step)

                fig = plot_spectrogram(mel.cpu().numpy())
                wandb.log(
                    {f"Mel": fig},
                    step=step
                )

                fig = plot_attention(attn.cpu().numpy())
                wandb.log(
                    {f"Attn": fig},
                    step=step
                )
