"""NFSP (Neural Fictitious Self-Play) network, state encoding, and action mapping.

Action space: 10 quantities (1..2*dice_count) x 6 faces (1..6), plus "liar" = 61 actions
for dice_count=5. Legal-action masking reuses Bid.__gt__ from game.py directly, so the
raise rules (including the asymmetric 1s rules) never get reimplemented/duplicated here.
"""
import random
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from game import Bid, Bot, GameState


@dataclass
class NFSPConfig:
    dice_count: int = 5
    gru_hidden: int = 64
    mlp_hidden: int = 128

    @property
    def max_quantity(self) -> int:
        return self.dice_count * 2

    @property
    def n_actions(self) -> int:
        return self.max_quantity * 6 + 1  # + "liar"

    @property
    def bid_feature_dim(self) -> int:
        return self.max_quantity + 6 + 1  # qty one-hot + face one-hot + bidder flag

    def to_dict(self):
        return asdict(self)


# ---------- Action <-> Bid ----------
def bid_to_action(bid: Bid, cfg: NFSPConfig) -> int:
    return (bid.quantity - 1) * 6 + (bid.face - 1)


def action_to_bid(idx: int, cfg: NFSPConfig):
    if idx == cfg.n_actions - 1:
        return "liar"
    q = idx // 6 + 1
    f = idx % 6 + 1
    return Bid(q, f)


def legal_action_mask(state: GameState, cfg: NFSPConfig) -> torch.BoolTensor:
    mask = torch.zeros(cfg.n_actions, dtype=torch.bool)
    cb = state.current_bid
    if cb is not None:
        mask[cfg.n_actions - 1] = True  # can always call liar once a bid exists
    for q in range(1, cfg.max_quantity + 1):
        for f in range(1, 7):
            bid = Bid(q, f)
            if cb is None or bid > cb:
                mask[bid_to_action(bid, cfg)] = True
    return mask


# ---------- State encoding ----------
def encode_hand(dice: list[int], cfg: NFSPConfig) -> torch.Tensor:
    counts = [0] * 6
    for d in dice:
        counts[d - 1] += 1
    return torch.tensor(counts, dtype=torch.float32) / cfg.dice_count


def encode_dice_counts(state: GameState, cfg: NFSPConfig) -> torch.Tensor:
    assert state.num_players == 2, "NFSP model currently only supports 1v1 games"
    opp_index = 1 - state.my_index
    my = state.dice_counts[state.my_index]
    opp = state.dice_counts[opp_index]
    return torch.tensor([my, opp], dtype=torch.float32) / cfg.dice_count


def encode_bid_history(state: GameState, cfg: NFSPConfig) -> torch.Tensor:
    rows = []
    for player_idx, bid in state.bid_history:
        q_onehot = [0.0] * cfg.max_quantity
        if 1 <= bid.quantity <= cfg.max_quantity:
            q_onehot[bid.quantity - 1] = 1.0
        # else: absurd/out-of-range quantity (engine allows it) -- leave all-zero, still
        # distinguishable via the face one-hot + bidder flag.
        f_onehot = [0.0] * 6
        f_onehot[bid.face - 1] = 1.0
        bidder_flag = 1.0 if player_idx == state.my_index else -1.0
        rows.append(q_onehot + f_onehot + [bidder_flag])
    if not rows:
        # Start-of-round "no bids yet" token: a single zero row. Combined with the GRU's
        # zero initial hidden state this gives a fixed, learnable "empty history" signal
        # without needing to special-case zero-length sequences.
        return torch.zeros((1, cfg.bid_feature_dim), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def encode_state(state: GameState, cfg: NFSPConfig):
    """Returns (hand[6], dice_counts[2], bid_seq[T, bid_feature_dim]) as CPU tensors."""
    return (
        encode_hand(state.my_dice, cfg),
        encode_dice_counts(state, cfg),
        encode_bid_history(state, cfg),
    )


def collate_states(states, device):
    """states: list of (hand, dice_counts, bid_seq) -> batched tensors on `device`."""
    hands = torch.stack([s[0] for s in states]).to(device)
    dice_counts = torch.stack([s[1] for s in states]).to(device)
    seqs = [s[2] for s in states]
    lengths = torch.tensor([seq.shape[0] for seq in seqs], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True).to(device)
    return hands, dice_counts, padded, lengths


# ---------- Network ----------
class NFSPNet(nn.Module):
    def __init__(self, cfg: NFSPConfig):
        super().__init__()
        self.cfg = cfg
        self.gru = nn.GRU(input_size=cfg.bid_feature_dim, hidden_size=cfg.gru_hidden, batch_first=True)
        feat_dim = cfg.gru_hidden + 6 + 2
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, cfg.mlp_hidden), nn.ReLU(),
            nn.Linear(cfg.mlp_hidden, cfg.mlp_hidden), nn.ReLU(),
        )
        self.head = nn.Linear(cfg.mlp_hidden, cfg.n_actions)

    def forward(self, hand, dice_counts, bid_seq, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(bid_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        hist = h_n[-1]
        feat = torch.cat([hist, hand, dice_counts], dim=-1)
        x = self.trunk(feat)
        return self.head(x)


def _single_forward(net: NFSPNet, feats, device):
    hand, dice_counts, seq, lengths = collate_states([feats], device)
    with torch.no_grad():
        return net(hand, dice_counts, seq, lengths).squeeze(0).cpu()


# ---------- Bots ----------
class NFSPTrainingBot(Bot):
    """Used only during self-play training. Samples a mode ('br' or 'avg') for the whole
    episode (anticipatory dynamics) and records its trajectory for later transition/SL
    reservoir construction by the training loop."""
    name = "NFSPTrain"

    def __init__(self, q_net: NFSPNet, sl_net: NFSPNet, cfg: NFSPConfig, mode: str, epsilon: float, device):
        self.q_net = q_net
        self.sl_net = sl_net
        self.cfg = cfg
        self.mode = mode
        self.epsilon = epsilon
        self.device = device
        self.trajectory = []   # (feats, action_idx, mask)
        self.sl_samples = []   # (feats, action_idx) -- only greedy best-response picks

    def act(self, state: GameState):
        feats = encode_state(state, self.cfg)
        mask = legal_action_mask(state, self.cfg)
        legal_idxs = mask.nonzero(as_tuple=True)[0]

        if self.mode == "avg":
            logits = _single_forward(self.sl_net, feats, self.device)
            logits = logits.masked_fill(~mask, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1).item()
        else:
            if random.random() < self.epsilon:
                idx = legal_idxs[random.randrange(len(legal_idxs))].item()
            else:
                q = _single_forward(self.q_net, feats, self.device)
                q = q.masked_fill(~mask, float("-inf"))
                idx = torch.argmax(q).item()
                self.sl_samples.append((feats, idx))

        self.trajectory.append((feats, idx, mask))
        return action_to_bid(idx, self.cfg)


class NFSPBot(Bot):
    """Deployable bot: acts via the average-policy (SL) network only. Samples from the
    masked distribution rather than argmax-ing, since argmax would collapse the learned
    mixed/bluffing strategy back into an exploitable pure strategy."""
    name = "NFSP"

    def __init__(self, sl_net: NFSPNet, cfg: NFSPConfig, device="cpu", sample: bool = True):
        self.sl_net = sl_net
        self.cfg = cfg
        self.device = device
        self.sample = sample

    @classmethod
    def from_checkpoint(cls, path: str, device="cpu", sample: bool = True):
        ckpt = torch.load(path, map_location=device)
        cfg = NFSPConfig(**ckpt["config"])
        net = NFSPNet(cfg).to(device)
        net.load_state_dict(ckpt["sl_net"])
        net.eval()
        return cls(net, cfg, device=device, sample=sample)

    def act(self, state: GameState):
        feats = encode_state(state, self.cfg)
        mask = legal_action_mask(state, self.cfg)
        logits = _single_forward(self.sl_net, feats, self.device)
        logits = logits.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, 1).item() if self.sample else torch.argmax(probs).item()
        return action_to_bid(idx, self.cfg)
