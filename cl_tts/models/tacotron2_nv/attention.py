import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from .modules import LinearNorm, ConvNorm


########################################
#                                      #
#           Forward Attention          #
#                                      #
########################################


class Linear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 init_gain='linear'):
        super(Linear, self).__init__()
        self.linear_layer = torch.nn.Linear(
            in_features, out_features, bias=bias)
        self._init_w(init_gain)

    def _init_w(self, init_gain):
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class LocationLayer(nn.Module):
    def __init__(self,
                 attention_dim,
                 attention_n_filters=32,
                 attention_kernel_size=31):
        super(LocationLayer, self).__init__()
        self.location_conv1d = nn.Conv1d(
            in_channels=2,
            out_channels=attention_n_filters,
            kernel_size=attention_kernel_size,
            stride=1,
            padding=(attention_kernel_size - 1) // 2,
            bias=False)
        self.location_dense = Linear(
            attention_n_filters, attention_dim, bias=False, init_gain='tanh')

    def forward(self, attention_cat):
        processed_attention = self.location_conv1d(attention_cat)
        processed_attention = self.location_dense(
            processed_attention.transpose(1, 2))
        return processed_attention


class ForwardAttention(nn.Module):
    """Following the methods proposed here:
        - https://arxiv.org/abs/1712.05884
        - https://arxiv.org/abs/1807.06736 + state masking at inference
        - Using sigmoid instead of softmax normalization
        - Attention windowing at inference time
    """

    # Pylint gets confused by PyTorch conventions here
    # pylint: disable=attribute-defined-outside-init
    def __init__(self, query_dim, embedding_dim, attention_dim,
                 location_attention, attention_location_n_filters,
                 attention_location_kernel_size, windowing, norm, forward_attn,
                 trans_agent, forward_attn_mask):
        super(ForwardAttention, self).__init__()
        self.query_layer = Linear(
            query_dim, attention_dim, bias=False, init_gain='tanh')
        self.inputs_layer = Linear(
            embedding_dim, attention_dim, bias=False, init_gain='tanh')
        self.v = Linear(attention_dim, 1, bias=True)
        if trans_agent:
            self.ta = nn.Linear(
                query_dim + embedding_dim, 1, bias=True)
        if location_attention:
            self.location_layer = LocationLayer(
                attention_dim,
                attention_location_n_filters,
                attention_location_kernel_size,
            )
        self._mask_value = -float("inf")
        self.windowing = windowing
        self.win_idx = None
        self.norm = norm
        self.forward_attn = forward_attn
        self.trans_agent = trans_agent
        self.forward_attn_mask = forward_attn_mask
        self.location_attention = location_attention

    def init_win_idx(self):
        self.win_idx = -1
        self.win_back = 2
        self.win_front = 6

    def init_forward_attn(self, inputs):
        B = inputs.shape[0]
        T = inputs.shape[1]
        self.alpha = torch.cat(
            [torch.ones([B, 1]),
             torch.zeros([B, T])[:, :-1] + 1e-7], dim=1).to(inputs.device)
        self.u = (0.5 * torch.ones([B, 1])).to(inputs.device)

    def init_location_attention(self, inputs):
        B = inputs.shape[0]
        T = inputs.shape[1]
        self.attention_weights_cum = inputs.data.new(B, T).zero_()

    def init_states(self, inputs):
        B = inputs.shape[0]
        T = inputs.shape[1]
        self.attention_weights = inputs.data.new(B, T).zero_()
        if self.location_attention:
            self.init_location_attention(inputs)
        if self.forward_attn:
            self.init_forward_attn(inputs)
        if self.windowing:
            self.init_win_idx()
        self.preprocess_inputs(inputs)

    def preprocess_inputs(self, inputs):
        self.processed_inputs = self.inputs_layer(inputs)

    def update_location_attention(self, alignments):
        self.attention_weights_cum += alignments

    def get_location_attention(self, query, processed_inputs):
        attention_cat = torch.cat((self.attention_weights.unsqueeze(1),
                                   self.attention_weights_cum.unsqueeze(1)),
                                  dim=1)
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_cat)
        energies = self.v(
            torch.tanh(processed_query + processed_attention_weights +
                       processed_inputs))
        energies = energies.squeeze(-1)
        return energies, processed_query

    def get_attention(self, query, processed_inputs):
        processed_query = self.query_layer(query.unsqueeze(1))
        energies = self.v(torch.tanh(processed_query + processed_inputs))
        energies = energies.squeeze(-1)
        return energies, processed_query

    def apply_windowing(self, attention, inputs):
        back_win = self.win_idx - self.win_back
        front_win = self.win_idx + self.win_front
        if back_win > 0:
            attention[:, :back_win] = -float("inf")
        if front_win < inputs.shape[1]:
            attention[:, front_win:] = -float("inf")
        # this is a trick to solve a special problem.
        # but it does not hurt.
        if self.win_idx == -1:
            attention[:, 0] = attention.max()
        # Update the window
        self.win_idx = torch.argmax(attention, 1).long()[0].item()
        return attention

    def apply_forward_attention(self, alignment):
        # forward attention
        fwd_shifted_alpha = F.pad(
            self.alpha[:, :-1].clone().to(alignment.device),
            (1, 0, 0, 0))
        # compute transition potentials
        alpha = ((1 - self.u) * self.alpha +
                 self.u * fwd_shifted_alpha + 1e-8) * alignment
        # force incremental alignment
        if not self.training and self.forward_attn_mask:
            _, n = fwd_shifted_alpha.max(1)
            val, n2 = alpha.max(1)
            for b in range(alignment.shape[0]):
                alpha[b, n[b] + 3:] = 0
                alpha[b, :(
                        n[b] - 1
                )] = 0  # ignore all previous states to prevent repetition.
                alpha[b,
                      (n[b] - 2
                       )] = 0.01 * val[b]  # smoothing factor for the prev step
        # renormalize attention weights
        alpha = alpha / alpha.sum(dim=1, keepdim=True)
        return alpha

    def forward(self, query, inputs, mask):
        """
        shapes:
            query: B x D_attn_rnn
            inputs: B x T_en x D_en
            processed_inputs:: B x T_en x D_attn
            mask: B x T_en
        """
        if self.location_attention:
            attention, _ = self.get_location_attention(
                query, self.processed_inputs)
        else:
            attention, _ = self.get_attention(
                query, self.processed_inputs)
        # apply masking
        # if mask is not None:
        #     attention.benchmarks.masked_fill_(~mask, self._mask_value)
        # apply windowing - only in eval mode
        if not self.training and self.windowing:
            attention = self.apply_windowing(attention, inputs)

        # normalize attention values
        if self.norm == "softmax":
            alignment = torch.softmax(attention, dim=-1)
        elif self.norm == "sigmoid":
            alignment = torch.sigmoid(attention) / torch.sigmoid(
                attention).sum(
                dim=1, keepdim=True)
        else:
            raise ValueError("Unknown value for attention norm type")

        if self.location_attention:
            self.update_location_attention(alignment)

        # apply forward attention if enabled
        if self.forward_attn:
            alignment = self.apply_forward_attention(alignment)
            self.alpha = alignment

        context = torch.bmm(alignment.unsqueeze(1), inputs)
        context = context.squeeze(1)
        self.attention_weights = alignment

        # compute transition agent
        if self.forward_attn and self.trans_agent:
            ta_input = torch.cat([context, query.squeeze(1)], dim=-1)
            self.u = torch.sigmoid(self.ta(ta_input))
        return context, alignment


########################################
#                                      #
#           GMM-V2 Attention           #
#                                      #
########################################


class GMMAttentionV2(nn.Module):
    """ Discretized Graves attention:
    (Code from Mozilla TTS https://github.com/mozilla/TTS)
        - https://arxiv.org/abs/1910.10288
        - https://arxiv.org/pdf/1906.01083.pdf
    """
    COEF = 0.3989422917366028  # numpy.sqrt(1/(2*numpy.pi))

    def __init__(self, query_dim, K):
        super().__init__()
        self._mask_value = 1e-8
        self.K = K
        # self.attention_alignment = 0.05
        self.eps = 1e-5
        self.J = None
        self.N_a = nn.Sequential(
            nn.Linear(query_dim, query_dim // 10, bias=True),
            nn.Tanh(),  # replaced ReLU with tanh
            nn.Linear(query_dim // 10, 3 * K, bias=True))

        self.softmax = nn.Softmax(dim=1)
        self.softplus = nn.Softplus()
        self.init_layers()

    def init_layers(self):
        # bias mean
        torch.nn.init.constant_(
            self.N_a[2].bias[(2 * self.K):(3 * self.K)], 1.)
        # bias std
        torch.nn.init.constant_(
            self.N_a[2].bias[self.K:(2 * self.K)], 10)

    def init_states(self, inputs):
        offset = 50
        if self.J is None or inputs.shape[1] + \
                offset > self.J.shape[-1] or self.J.shape[0] != inputs.shape[0]:
            self.J = torch.arange(
                0,
                inputs.shape[1] + offset
            ).to(inputs.device).expand_as(
                torch.Tensor(inputs.shape[0], self.K, inputs.shape[1] + offset)
            )
        self.attention_weights = torch.zeros(inputs.shape[0],
                                             inputs.shape[1]).to(inputs.device)
        self.mu_prev = torch.zeros(inputs.shape[0], self.K).to(inputs.device)

    # pylint: disable=R0201
    # pylint: disable=unused-argument

    def forward(self, query, inputs, mask):
        """
        Shapes:
            query: B x D_attention_rnn
            inputs: B x T_in x D_encoder
            processed_inputs: place_holder
            mask: B x T_in
        """
        gbk_t = self.N_a(query)
        gbk_t = gbk_t.view(gbk_t.size(0), -1, self.K)

        # attention model parameters
        # each B x K
        g_t = gbk_t[:, 0, :]
        b_t = gbk_t[:, 1, :]
        k_t = gbk_t[:, 2, :]

        # attention GMM parameters
        g_t = self.softmax(g_t) + self.eps
        mu_t = self.mu_prev + self.softplus(k_t)
        sig_t = self.softplus(b_t) + self.eps

        # update perv mu_t
        self.mu_prev = mu_t

        # calculate normalizer
        z_t = torch.sqrt(2 * np.pi * (sig_t ** 2)) + self.eps

        # location indices
        j = self.J[:g_t.size(0), :, :inputs.size(1)]

        # attention weights
        g_t = g_t.unsqueeze(2).expand(g_t.size(0), g_t.size(1), inputs.size(1))
        z_t = z_t.unsqueeze(2).expand_as(g_t)
        mu_t_ = mu_t.unsqueeze(2).expand_as(g_t)
        sig_t = sig_t.unsqueeze(2).expand_as(g_t)

        phi_t = (g_t / z_t) * torch.exp(
            -0.5 * (1.0 / (sig_t ** 2)) * (mu_t_ - j) ** 2)

        # discritize attention weights
        alpha_t = torch.sum(phi_t, 1).float()

        context_vec = torch.bmm(alpha_t.unsqueeze(1), inputs).squeeze(1)

        return context_vec, alpha_t


########################################
#                                      #
#           GMM-V2 Attention           #
#                                      #
########################################


class LocLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class LSAttention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim,
                 attention_dim, attention_location_n_filters,
                 attention_location_kernel_size):
        super(LSAttention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(2)
        return energies

    def forward(self, attention_hidden_state, memory, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded benchmarks
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        alignment = alignment.masked_fill(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights
