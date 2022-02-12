from typing import List
import torch
import torch.nn as nn
from math import sqrt

from .encoder import Encoder
from .decoder import Decoder, Postnet
from .modules import get_mask_from_lengths
from .tacotron2_loss import Tacotron2Loss


class Tacotron2NV(nn.Module):
    def __init__(self, params):
        super(Tacotron2NV, self).__init__()
        self.params = params
        self.mask_padding = params["mask_padding"]
        self.n_mel_channels = params["n_mel_channels"]
        self.n_frames_per_step = params["n_frames_per_step"]
        
        # ----- Char embedder
        self.embedding = nn.Embedding(params["n_symbols"],
                                      params["symbols_embedding_dim"])
        std = sqrt(2.0 / (params["n_symbols"] +
                          params["symbols_embedding_dim"]))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)

        # ----- Encoder
        self.encoder = Encoder(params["encoder_n_convolutions"],
                               params["encoder_embedding_dim"],
                               params["encoder_kernel_size"])
        encoder_embedding_dim = params["encoder_embedding_dim"]

        # ----- Speaker embedding
        if params["use_speaker_embedding"]:
            self.speaker_embedder = nn.Embedding(
                params["num_speakers"],
                params["speaker_embedding_dim"]
            )
            encoder_embedding_dim += params["speaker_embedding_dim"]

        # ----- Decoder
        self.decoder = Decoder(params["n_mel_channels"], 
                               params["n_frames_per_step"],
                               encoder_embedding_dim, 
                               params["attention_params"],
                               params["decoder_rnn_dim"],
                               params["attention_rnn_dim"],
                               params["prenet_dim"], 
                               params["max_decoder_steps"],
                               params["gate_threshold"], 
                               params["p_attention_dropout"],
                               params["p_decoder_dropout"],
                               not params["decoder_no_early_stopping"])

        # ----- Postnet
        self.postnet = Postnet(params["n_mel_channels"], 
                               params["postnet_embedding_dim"],
                               params["postnet_kernel_size"],
                               params["postnet_n_convolutions"])

    def parse_output(self, outputs, output_lengths):
        # type: (List[Tensor], Tensor) -> List[Tensor]
        if self.mask_padding and output_lengths is not None:
            mask = get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].masked_fill_(mask, 0.0)
            outputs[1].masked_fill_(mask, 0.0)
            outputs[2].masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, 
                inputs, 
                input_lengths, 
                melspecs, 
                melspec_lengths,
                speaker_ids):
                
        embedded_inputs = self.embedding(inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, input_lengths)

        if self.params["use_speaker_embedding"]:
            spk_emb_vec = self.speaker_embedder(speaker_ids).unsqueeze(1)
            spk_emb_vec = spk_emb_vec.expand(encoder_outputs.size(0),
                                             encoder_outputs.size(1), -1)
            encoder_outputs = torch.cat([encoder_outputs, spk_emb_vec], dim=-1)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, melspecs, input_lengths=input_lengths
        )

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output([mel_outputs, mel_outputs_postnet,
                                  gate_outputs, alignments], melspec_lengths)

    def infer(self, 
              inputs, 
              input_lengths,
              speaker_ids):

        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.infer(embedded_inputs, input_lengths)
        
        if self.params["use_speaker_embedding"]:
            spk_emb_vec = self.speaker_embedder(speaker_ids).unsqueeze(1)
            spk_emb_vec = spk_emb_vec.expand(encoder_outputs.size(0),
                                             encoder_outputs.size(1), -1)
            encoder_outputs = torch.cat([encoder_outputs, spk_emb_vec], dim=-1)

        mel_outputs, gate_outputs, alignments, mel_lengths = \
            self.decoder.infer(encoder_outputs, input_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        BS = mel_outputs_postnet.size(0)
        alignments = alignments.unfold(1, BS, BS).transpose(0, 2)

        return mel_outputs_postnet, mel_lengths, alignments


# Helper classes for forward and criterion functions

class Tac2NvForwardFunc:
    def __call__(self, model: Tacotron2NV, m_batch, speaker_ids):
        return model(
            inputs=m_batch["transcripts"],
            input_lengths=m_batch["trans_lengths"],
            melspecs=m_batch["melspecs"],
            melspec_lengths=m_batch["melspec_lengths"],
            speaker_ids=speaker_ids
        )

class Tac2NvCriterionFunc:
    def __init__(self, params, device):
        self.criterion = Tacotron2Loss(
            params["model"]["n_frames_per_step"],
            params["model"]["reduction"],
            params["model"]["pos_weight"],
            device
        )

    def __call__(self, model_outputs, m_batch, speaker_ids):
        targets = m_batch["melspecs"], m_batch["stop_targets"]
        return self.criterion(
            model_output=model_outputs,
            targets=targets,
            mel_len=m_batch["melspec_lengths"],
        )


# Helper function for model initialization
def get_tacotron2_nv(params, n_speakers, n_symbols, device):
    params["model"]["num_speakers"] = n_speakers
    params["model"]["n_symbols"] = n_symbols
    params["model"]["n_mel_channels"] = params["ap_params"]["n_mels"]
    model = Tacotron2NV(params["model"])
    forward_func = Tac2NvForwardFunc()
    criterion_func = Tac2NvCriterionFunc(params, device)

    return model, forward_func, criterion_func
