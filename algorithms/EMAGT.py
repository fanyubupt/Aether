import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from algorithms.util_algo import check, init
from algorithms.transformer_act import continuous_autoregreesive_act, continuous_autoregressive_act_approx
from algorithms.transformer_act import continuous_parallel_act
from gnnencoder.HirachicalGNN import hirachicalGNN


def init_(m, gain=0.01, activate=False):
    """
    Initializes the parameters of a neural network layer using orthogonal initialization.
    If 'activate' is True, calculates the gain for ReLU activation.

    :param m: (nn.Module) The layer to be initialized.
    :param gain: (float) Scaling factor for orthogonal initialization (default 0.01).
    :param activate: (bool) Whether to recalculate gain for 'relu'.
    :return: (nn.Module) The initialized layer.
    """

    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class Attention(nn.Module):
    """
    A fast-attention mechanism designed for efficient computation.

    :param n_embd: (int) Total embedding dimension.
    :param n_head: (int) Number of attention heads.
    :param masked: (bool) Whether to mask future tokens (causal attention).
    """

    def __init__(self, n_embd, n_head, masked=False):
        super(Attention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        hidden = 20
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, hidden))
        self.query = init_(nn.Linear(n_embd, hidden))
        self.value = init_(nn.Linear(n_embd, hidden))
        # output projection
        self.proj = init_(nn.Linear(hidden, n_embd))

        self.att_bp = None

    def forward(self, key, value, query):
        """
        Forward pass for the fast-attention module.

        :param key: (torch.Tensor) Key tensor of shape [B, L, D].
        :param value: (torch.Tensor) Value tensor of shape [B, L, D].
        :param query: (torch.Tensor) Query tensor of shape [B, L, D].
        :return: (torch.Tensor) Output of the attention mechanism with shape [B, L, D].
        """

        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        ks = self.key(key)# (B, nh, L, hs)
        qs = self.query(query)  # (B, nh, L, hs)
        vs = self.value(value)  # (B, nh, L, hs)

        if qs.sum()!=0 and ks.sum()!=0:
            qs = qs / torch.norm(qs, p=2)  # [N, H, M]
            ks = ks / torch.norm(ks, p=2)  # [L, H, M]
        N = qs.shape[0]

        # numerator
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
        all_ones = torch.ones([vs.shape[0]]).to(vs.device)
        vs_sum = torch.einsum("l,lhd->hd", all_ones, vs)  # [H, D]
        attention_num += vs_sum.unsqueeze(0).repeat(vs.shape[0], 1, 1)  # [N, H, D]

        # denominator
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer  # [N, H, D]

        # output projection
        y = self.proj(attn_output)
        return y


class SelfAttention(nn.Module):
    """
    A standard self-attention module using scaled dot-product attention.

    :param n_embd: (int) Total embedding dimension.
    :param n_head: (int) Number of attention heads.
    :param masked: (bool) Whether to apply causal masking for future tokens.
    """

    def __init__(self, n_embd, n_head, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence

        self.att_bp = None

    def forward(self, key, value, query):
        """
        Forward pass for standard self-attention.

        :param key: (torch.Tensor) Key tensor of shape [B, L, D].
        :param value: (torch.Tensor) Value tensor of shape [B, L, D].
        :param query: (torch.Tensor) Query tensor of shape [B, L, D].
        :return: (torch.Tensor) Attention outputs of shape [B, L, D].
        """

        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.masked:
            # Create mask dynamically
            mask = torch.tril(torch.ones(L, L, device=query.device)).view(1, 1, L, L)
            att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y


class EncodeBlock(nn.Module):
    """
    A single transformer encoder block consisting of:
    1) Multi-head (or Fast) Attention
    2) LayerNorm and residual connection
    3) MLP with a GELU activation
    4) Another LayerNorm and residual connection
    """

    def __init__(self, n_embd, n_head, use_normal_attn=False):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # self.attn = SelfAttention(n_embd, n_head, masked=True)
        if use_normal_attn:
            self.attn = SelfAttention(n_embd, n_head, masked=False)
        else:
            self.attn = Attention(n_embd, n_head, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x):
        """
        Forward pass of the encoder block.

        :param x: (torch.Tensor) Input tensor.
        :return: (torch.Tensor) Output of the block with the same shape.
        """

        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class DecodeBlock(nn.Module):
    """
    A single transformer decoder block consisting of:
    1) Masked multi-head (or fast) attention
    2) LayerNorm and residual connection
    3) Another attention from encoder outputs
    4) MLP with a GELU activation
    5) Additional LayerNorm and residual connection
    """

    def __init__(self, n_embd, n_head, use_normal_attn=False):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        if use_normal_attn:
            self.attn1 = SelfAttention(n_embd, n_head, masked=True)
            self.attn2 = SelfAttention(n_embd, n_head, masked=True)
        else:
            self.attn1 = Attention(n_embd, n_head, masked=True)
            self.attn2 = Attention(n_embd, n_head, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, rep_enc):
        """
        Forward pass of the decoder block.

        :param x: (torch.Tensor) Decoder input.
        :param rep_enc: (torch.Tensor) Encoder representation.
        :return: (torch.Tensor) Output of the block.
        """

        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x


class Encoder(nn.Module):
    """
    The encoder module of the Aether's multi-agent transformer.
    Projects input embeddings through multiple encoder blocks,
    then outputs a feature representation and a value estimate.
    """

    def __init__(self,decision_unit,state_dim, hidden_dim, n_block, n_embd, n_head, encode_state, use_normal_attn=False):
        super(Encoder, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_embd = n_embd
        self.encode_state = encode_state
        self.obs_encoder = nn.Sequential(nn.LayerNorm(hidden_dim),
                                         init_(nn.Linear(hidden_dim, n_embd), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, use_normal_attn) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, 1)))

    def forward(self, state, obs):
        """
        Forward pass of the encoder. uses obs for multi-agent tasks.

        :param state: (torch.Tensor) State input [unused if encode_state=False].
        :param obs: (torch.Tensor) Observations for each agent/path
        :return:
          - v_loc: (torch.Tensor) Scalar value predictions
          - rep: (torch.Tensor) Encoded feature representation
        """

        if self.encode_state:
            state_embeddings = self.state_encoder(state)
            x = state_embeddings
        else:
            obs_embeddings = self.obs_encoder(obs)#obs_encoder(hidden*du,hidden*du)
            x = obs_embeddings

        rep = self.blocks(self.ln(x))
        v_loc = self.head(rep)

        return v_loc, rep


class Decoder(nn.Module):
    """
    The decoder module of the Aether's multi-agent transformer.
    Takes the encoder representation and optionally the previous actions,
    then produces either discrete logits or continuous action parameters.
    """

    def __init__(self, decision_unit, hidden_dim, action_dim, n_block, n_embd, n_head,
                 action_type='Discrete', dec_actor=False, share_actor=False, use_normal_attn=False):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type

        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            self.log_std = torch.nn.Parameter(log_std)

        if action_type == 'Discrete':
            self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True),
                                                nn.GELU())
        else:
            self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(hidden_dim),
                                         init_(nn.Linear(hidden_dim, n_embd), activate=True), nn.GELU())
        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, use_normal_attn) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, action_dim)))

    def zero_std(self, device):
        """
        Sets the std of the continuous policy to zero.
        :param device: (torch.device) The device on which to place the log_std.
        """

        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    # state, action, and return
    def forward(self, action, obs_rep, obs):
        """
        Forward pass through the decoder. If dec_actor is True and share_actor is False,
        we apply a separate MLP per agent; otherwise, we autoregressively process the
        action embeddings alongside obs_rep.

        :param action: (torch.Tensor) Input actions.
        :param obs_rep: (torch.Tensor) Encoder representations.
        :param obs: (torch.Tensor) Observations if dec_actor logic is used (unused otherwise).
        :return: (torch.Tensor) Action logits or continuous action means.
        """

        if self.dec_actor:
            if self.share_actor:
                logit = self.mlp(obs)
            else:
                logit = []
                for n in range(len(self.mlp)):
                    logit_n = self.mlp[n](obs[:, n, :])
                    logit.append(logit_n)
                logit = torch.stack(logit, dim=1)
        else:
            action_embeddings = self.action_encoder(action)
            x = self.ln(action_embeddings)
            for block in self.blocks:
                x = block(x, obs_rep)
            logit = self.head(x)

        return logit


class MultiAgentTransformer(nn.Module):
    """
    Aether's Multi-agent transformer model that integrates an encoder, a decoder, and a subgraph-GNN
    for hierarchical feature extraction from graph-based observations (edge features).
    """

    def __init__(self, env, action_dim,
                 n_block, n_embd, n_head, encode_state=False, device=torch.device("cpu"),
                 action_type='Discrete', dec_actor=False, share_actor=False, use_normal_attn=False):
        super(MultiAgentTransformer, self).__init__()
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device
        self.env = env
        self.decision_unit=env.decision_unit
        self.expand_dim = self.decision_unit*self.action_dim
        state_dim = 37
        hidden_dim = 28
        self.subGNN = hirachicalGNN(env,self.decision_unit,hidden_dim,device)
        self.encoder = Encoder(self.decision_unit,state_dim*self.expand_dim, hidden_dim*self.expand_dim,
                               n_block, n_embd*self.expand_dim, n_head, encode_state, use_normal_attn)
        self.decoder = Decoder(self.decision_unit,hidden_dim*self.expand_dim, action_dim*self.decision_unit,
                               n_block, n_embd*self.expand_dim, n_head,
                               self.action_type, dec_actor=dec_actor, share_actor=share_actor, use_normal_attn=use_normal_attn)
        self.paths_to_edges = env.paths_to_edges
        self.to(device)


    def zero_std(self):
        """
        Utility to reset the log_std parameter to zero for continuous actions.
        """

        if self.action_type != 'Discrete':
            self.decoder.zero_std(self.device)

    def forward(self, state, obs, action, edge_features,indices,  **kwargs):
        """
        Forward pass for training/evaluation:
        1) Construct features using subGNN.
        2) Encode path features via Transformer encoder for value or representation.
        3) Decode actions using parallel decoding.

        :param state: (torch.Tensor) [Unused by default]
        :param obs: (torch.Tensor) Observations.
        :param action: (torch.Tensor) Actions.
        :param edge_features: (torch.Tensor) Edge features.
        :param indices: (torch.Tensor) Not used in this snippet, but kept for reference.
        :param kwargs: Topology-specific arguments.
        :return:
          - action_log: (torch.Tensor) Action log probabilities.
          - v_loc: (torch.Tensor) Estimated values.
          - entropy: (torch.Tensor) Entropy of the action distribution.
        """

        edge_adj = self.env.edge_adj if "edge_adj" not in kwargs else kwargs["edge_adj"]
        paths_to_edges = self.paths_to_edges if "paths_to_edges" not in kwargs else kwargs["paths_to_edges"]

        ori_shape = np.shape(state)

        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        edge_features = check(edge_features).to(**self.tpdv)

        edge_hidden = self.subGNN.edgegnn(edge_features,edge_adj)
        path_features = self.subGNN(obs,edge_hidden,paths_to_edges) #b,14,40*feature_dim

        batch_size = np.shape(obs)[0]
        v_loc, obs_rep= self.encoder(None, path_features)

        action_log, entropy = continuous_parallel_act(self.decoder,self.decision_unit, obs_rep, obs, action, batch_size,
                                                       self.action_dim, self.tpdv)

        action_log = action_log.reshape(action_log.shape[0],-1,self.action_dim).to(self.device)
        entropy = entropy.reshape(entropy.shape[0],-1,self.action_dim).to(self.device)
        v_loc = v_loc.reshape(v_loc.shape[0],-1,1).to(self.device)

        return action_log, v_loc, entropy

    def get_actions(self, state, obs,edge_features, indices, deterministic=False, **kwargs):
        """
        Retrieves actions autoregressively from the decoder using the encoded representation.

        :param state: (torch.Tensor) Not used by default.
        :param obs: (torch.Tensor) Observations.
        :param edge_features: (torch.Tensor) Edge features.
        :param indices: (torch.Tensor) Not used in this function.
        :param deterministic: (bool) Whether to output deterministic actions (mean) or sample from distribution.
        :param kwargs: Topology-specific arguments.
        :return:
          - output_action: (torch.Tensor) Sampled or deterministic actions.
          - output_action_log: (torch.Tensor) Log probabilities of the actions.
          - v_loc: (torch.Tensor) Value predictions.
        """

        edge_adj = self.env.edge_adj if "edge_adj" not in kwargs else kwargs["edge_adj"]
        paths_to_edges = self.paths_to_edges if "paths_to_edges" not in kwargs else kwargs["paths_to_edges"]
        n_nodes = self.env.n_nodes if "n_nodes" not in kwargs else kwargs["n_nodes"]

        ori_shape = np.shape(obs)

        obs = check(obs).to(**self.tpdv)
        edge_features = check(edge_features).to(**self.tpdv)

        batch_size = np.shape(obs)[0]
        edge_hidden = self.subGNN.edgegnn(edge_features, edge_adj)
        path_features = self.subGNN(obs,edge_hidden,paths_to_edges) #b,14,40*feature_dim

        v_loc, obs_rep= self.encoder(None, path_features)

        output_action, output_action_log = continuous_autoregressive_act_approx(self.decoder,self.decision_unit, obs_rep, obs, batch_size,
                                                                        self.action_dim, self.tpdv, n_nodes,
                                                                         deterministic)
        output_action = output_action.reshape(output_action.shape[0],-1,self.action_dim).to(self.device)
        output_action_log = output_action_log.reshape(output_action_log.shape[0],-1,self.action_dim).to(self.device)
        v_loc = v_loc.reshape(v_loc.shape[0],-1,1).to(self.device)
        return output_action, output_action_log, v_loc

    def get_values(self, state, obs,edge_features,indices, **kwargs):
        """
        Computes only the value predictions from the encoded representation.

        :param state: (torch.Tensor) [Unused by default].
        :param obs: (torch.Tensor) Observations.
        :param edge_features: (torch.Tensor) Edge features.
        :param indices: (torch.Tensor) Not used in this function.
        :param kwargs: Topology-specific arguments.
        :return: (torch.Tensor) Value predictions.
        """

        edge_adj = self.env.edge_adj if "edge_adj" not in kwargs else kwargs["edge_adj"]
        paths_to_edges = self.paths_to_edges if "paths_to_edges" not in kwargs else kwargs["paths_to_edges"]
        obs = check(obs).to(**self.tpdv)
        edge_features = check(edge_features).to(**self.tpdv)
        edge_hidden = self.subGNN.edgegnn(edge_features, edge_adj)
        path_features = self.subGNN(obs,edge_hidden, paths_to_edges) #b,14,40*feature_dim

        v_tot, obs_rep = self.encoder(None, path_features)
        return v_tot




