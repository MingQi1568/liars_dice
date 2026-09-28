"""NFSP (Neural Fictitious Self-Play) networks, state encodings, and action mapping.

Two architectures, selected by NFSPConfig.arch and stored in every checkpoint:
  "flat"         original: GRU over absolute bid history + MLP trunk + one flat 61-way head.
  "shared_face"  one small MLP scores every candidate face 2-6 with shared weights, fed
                 face-relative features (counts incl. wilds, legality gaps, opponent history)
                 instead of face-position-locked inputs; the bid history is encoded relative
                 to your own hand so no face numeral reaches the GRU.

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
    arch: str = "flat"

    @property
    def max_quantity(self) -> int:
        return self.dice_count * 2

    @property
    def n_actions(self) -> int:
        return self.max_quantity * 6 + 1  # + "liar"

    @property
    def bid_feature_dim(self) -> int:
        # flat: qty one-hot + face one-hot(6) + bidder flag; shared_face: qty one-hot + 7
        # relative features. Both come to the same width.
        return self.max_quantity + 7

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


# ---------- "flat" state encoding ----------
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


# ---------- "shared_face" state encoding ----------
FACE_FEATS = 10      # per-face input to SharedFaceMLP
FACE1_FEATS = 8      # input to Face1MLP
LIAR_SCALARS = 8
GLOBAL_SCALARS = 4
FACE_EMB = 16
Q_EMB = 8
SCORE_HIDDEN = 64
LIAR_HIDDEN = 32

# Layout of the packed per-state "static" vector.
S_GLOBAL = slice(0, 4)      # my_dice/dc, opp_dice/dc, count_1s/dc, has_current_bid
S_FACES = slice(4, 54)      # 5 faces (2..6) x FACE_FEATS
S_FACE1 = slice(54, 62)
S_LIAR = slice(62, 70)
S_CLAIM = slice(70, 76)     # one-hot of the claimed face; only used to select e_claim inside forward()
STATIC_DIM = 76


def _my_face_counts(dice: list[int]) -> list[int]:
    counts = [0] * 7  # index by face 1..6
    for d in dice:
        counts[d] += 1
    return counts


def _eff_count(counts: list[int], face: int) -> int:
    """Dice in my hand that count toward `face`: 1s are wild for faces 2-6, not for 1s."""
    return counts[1] if face == 1 else counts[face] + counts[1]


def encode_bid_history_relative(state: GameState, cfg: NFSPConfig, counts: list[int]) -> torch.Tensor:
    """Per-bid features with no face numeral: quantity one-hot, bidder, how many of that face
    I hold, how far the bid stretches past that, and how it relates to the previous bid."""
    qmax = cfg.max_quantity
    qcap = qmax + 1
    dc = cfg.dice_count
    rows = []
    prev_face = None
    for player_idx, bid in state.bid_history:
        q_onehot = [0.0] * qmax
        if 1 <= bid.quantity <= qmax:
            q_onehot[bid.quantity - 1] = 1.0
        eff = _eff_count(counts, bid.face)
        needed = min(max(0, bid.quantity - eff), qcap) / qcap
        same = up = down = 0.0
        if prev_face is not None:
            same = float(bid.face == prev_face)
            if bid.face != 1 and prev_face != 1:
                up = float(bid.face > prev_face)
                down = float(bid.face < prev_face)
        bidder = 1.0 if player_idx == state.my_index else -1.0
        rows.append(q_onehot + [bidder, eff / dc, needed, same, up, down, float(bid.face == 1)])
        prev_face = bid.face
    if not rows:
        return torch.zeros((1, cfg.bid_feature_dim), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def encode_state_shared(state: GameState, cfg: NFSPConfig, mask=None):
    """Returns (static[STATIC_DIM], bid_seq[T, bid_feature_dim]) as CPU tensors.

    Only own dice + public information are used. All quantity-valued features saturate at
    max_quantity+1 so absurd (out-of-grid) bids can't blow up the scale.
    """
    assert state.num_players == 2, "NFSP model currently only supports 1v1 games"
    if mask is None:
        mask = legal_action_mask(state, cfg)
    dc = cfg.dice_count
    qmax = cfg.max_quantity
    qcap = qmax + 1
    counts = _my_face_counts(state.my_dice)
    count_1s = counts[1]
    hand = len(state.my_dice)
    opp = state.dice_counts[1 - state.my_index]
    total = state.total_dice
    cb = state.current_bid

    opp_max = [0] * 7
    opp_times = [0] * 7
    for pidx, bid in state.bid_history:
        if pidx != state.my_index:
            opp_max[bid.face] = max(opp_max[bid.face], bid.quantity)
            opp_times[bid.face] += 1

    raise_ok = mask[: qmax * 6].tolist()

    def min_next(face: int) -> int:
        for q in range(1, qmax + 1):
            if raise_ok[(q - 1) * 6 + face - 1]:
                return q
        return qcap

    faces = []
    for f in range(2, 7):
        eff = counts[f] + count_1s
        if cb is None:
            above = below = equal = 0.0
        elif cb.face == 1:
            above, below, equal = 1.0, 0.0, 0.0  # any non-1 face may follow a 1s bid at the same quantity
        else:
            above, below, equal = float(f > cb.face), float(f < cb.face), float(f == cb.face)
        mn = min_next(f)
        faces += [
            eff / dc, hand / dc, total / (2 * dc), above, below, equal,
            min(opp_max[f], qcap) / qcap, min(opp_times[f], qmax) / qmax,
            mn / qcap, (mn - eff) / qcap,
        ]

    mn1 = min_next(1)
    face1 = [
        count_1s / dc, hand / dc, total / (2 * dc), float(cb is not None and cb.face == 1),
        min(opp_max[1], qcap) / qcap, min(opp_times[1], qmax) / qmax,
        mn1 / qcap, (mn1 - count_1s) / qcap,
    ]

    claim = [0.0] * 6
    if cb is None:
        liar = [0.0] * LIAR_SCALARS
    else:
        eff = _eff_count(counts, cb.face)
        needed = max(0, cb.quantity - eff)
        liar = [
            min(needed, qcap) / qcap, min(needed / opp, 2.0) / 2.0,
            eff / dc, total / (2 * dc), opp / dc,
            min(cb.quantity, qcap) / qcap, float(needed > opp), float(cb.face == 1),
        ]
        claim[cb.face - 1] = 1.0

    glob = [hand / dc, opp / dc, count_1s / dc, float(cb is not None)]
    static = torch.tensor(glob + faces + face1 + liar + claim, dtype=torch.float32)
    return static, encode_bid_history_relative(state, cfg, counts)


def encode_state(state: GameState, cfg: NFSPConfig, mask=None):
    """flat: (hand[6], dice_counts[2], bid_seq[T, D]); shared_face: (static[76], bid_seq[T, D])."""
    if cfg.arch == "shared_face":
        return encode_state_shared(state, cfg, mask)
    return (
        encode_hand(state.my_dice, cfg),
        encode_dice_counts(state, cfg),
        encode_bid_history(state, cfg),
    )


def collate_states(states, device):
    """flat: list of (hand, dice_counts, bid_seq) -> batched tensors on `device`."""
    hands = torch.stack([s[0] for s in states]).to(device)
    dice_counts = torch.stack([s[1] for s in states]).to(device)
    seqs = [s[2] for s in states]
    lengths = torch.tensor([seq.shape[0] for seq in seqs], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True).to(device)
    return hands, dice_counts, padded, lengths


def collate_shared(states, device):
    """shared_face: list of (static, bid_seq) -> batched tensors on `device`."""
    static = torch.stack([s[0] for s in states]).to(device)
    seqs = [s[1] for s in states]
    lengths = torch.tensor([seq.shape[0] for seq in seqs], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True).to(device)
    return static, padded, lengths


# ---------- Networks ----------
class NFSPNet(nn.Module):
    """Original ("flat") architecture."""

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


class SharedFaceNet(nn.Module):
    """Face-shared architecture: the 50 raise scores for faces 2-6 come from one MLP queried
    per (quantity, face) with weight tying, so what is learned about "three of a kind" on one
    face applies to every other face. Face 1 and "liar" have their own small heads."""

    def __init__(self, cfg: NFSPConfig):
        super().__init__()
        self.cfg = cfg
        g = cfg.gru_hidden + GLOBAL_SCALARS
        score_in = g + FACE_EMB + Q_EMB + 2
        self.gru = nn.GRU(input_size=cfg.bid_feature_dim, hidden_size=cfg.gru_hidden, batch_first=True)
        self.shared_face_mlp = nn.Sequential(
            nn.Linear(FACE_FEATS, FACE_EMB), nn.ReLU(), nn.Linear(FACE_EMB, FACE_EMB))
        self.face1_mlp = nn.Sequential(
            nn.Linear(FACE1_FEATS, FACE_EMB), nn.ReLU(), nn.Linear(FACE_EMB, FACE_EMB))
        self.quantity_embedding = nn.Embedding(cfg.max_quantity, Q_EMB)
        self.q_score_mlp = nn.Sequential(
            nn.Linear(score_in, SCORE_HIDDEN), nn.ReLU(), nn.Linear(SCORE_HIDDEN, 1))
        self.face1_score_mlp = nn.Sequential(
            nn.Linear(score_in, SCORE_HIDDEN), nn.ReLU(), nn.Linear(SCORE_HIDDEN, 1))
        self.liar_head = nn.Sequential(
            nn.Linear(g + LIAR_SCALARS + FACE_EMB + 2, LIAR_HIDDEN), nn.ReLU(), nn.Linear(LIAR_HIDDEN, 1))

    def forward(self, static, bid_seq, lengths, mask):
        cfg = self.cfg
        B = static.shape[0]
        Q = cfg.max_quantity
        dc = float(cfg.dice_count)

        packed = nn.utils.rnn.pack_padded_sequence(bid_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        scal = static[:, S_GLOBAL]
        ctx = torch.cat([h_n[-1], scal], dim=-1)                       # (B, g)

        fi = static[:, S_FACES].reshape(B, 5, FACE_FEATS)
        e_f = self.shared_face_mlp(fi)                                  # (B, 5, E), same weights for every face
        e_1 = self.face1_mlp(static[:, S_FACE1])                        # (B, E)

        qs = torch.arange(1, Q + 1, device=static.device, dtype=static.dtype)   # (Q,)
        q_emb = self.quantity_embedding.weight                                    # (Q, Q_EMB)
        opp = (scal[:, 1] * dc).view(B, 1, 1)

        def extras(eff):  # eff: (B, K) my effective count -> per-(face, quantity) gap and needed-ratio
            gap = qs.view(1, 1, Q) - eff.unsqueeze(-1)
            ratio = (gap.clamp(min=0) / opp).clamp(max=2.0) / 2.0
            return (gap / Q).unsqueeze(-1), ratio.unsqueeze(-1)

        gap_f, ratio_f = extras(fi[:, :, 0] * dc)                       # (B, 5, Q, 1)
        comb_f = torch.cat([
            ctx.view(B, 1, 1, -1).expand(B, 5, Q, -1),
            e_f.unsqueeze(2).expand(B, 5, Q, -1),
            q_emb.view(1, 1, Q, -1).expand(B, 5, Q, -1),
            gap_f, ratio_f], dim=-1)
        s_f = self.q_score_mlp(comb_f).squeeze(-1)                      # (B, 5, Q)

        gap_1, ratio_1 = extras((scal[:, 2] * dc).unsqueeze(-1))        # (B, 1, Q, 1)
        comb_1 = torch.cat([
            ctx.view(B, 1, 1, -1).expand(B, 1, Q, -1),
            e_1.view(B, 1, 1, -1).expand(B, 1, Q, -1),
            q_emb.view(1, 1, Q, -1).expand(B, 1, Q, -1),
            gap_1, ratio_1], dim=-1)
        s_1 = self.face1_score_mlp(comb_1).squeeze(-1).squeeze(1)       # (B, Q)

        # Faces ordered 1,2..6 so flat index = (q-1)*6 + (f-1), matching bid_to_action.
        flat_raise = torch.cat([s_1.unsqueeze(-1), s_f.permute(0, 2, 1)], dim=-1).reshape(B, Q * 6)

        raise_mask = mask[:, : Q * 6]
        with torch.no_grad():
            best = flat_raise.detach().masked_fill(~raise_mask, float("-inf")).max(dim=1).values
            best = torch.where(torch.isfinite(best), best, torch.zeros_like(best))
        n_legal = raise_mask.to(flat_raise.dtype).sum(dim=1) / (Q * 6)

        e_all = torch.cat([e_1.unsqueeze(1), e_f], dim=1)               # (B, 6, E), faces 1..6
        e_claim = (static[:, S_CLAIM].unsqueeze(-1) * e_all).sum(dim=1)  # zeros when no bid stands
        liar_in = torch.cat([ctx, static[:, S_LIAR], e_claim, best.unsqueeze(-1), n_legal.unsqueeze(-1)], dim=-1)
        liar = self.liar_head(liar_in)                                  # (B, 1)
        return torch.cat([flat_raise, liar], dim=-1)


def build_net(cfg: NFSPConfig) -> nn.Module:
    if cfg.arch == "shared_face":
        return SharedFaceNet(cfg)
    if cfg.arch == "flat":
        return NFSPNet(cfg)
    raise ValueError(f"unknown arch {cfg.arch!r} (expected 'flat' or 'shared_face')")


def collate_for(cfg: NFSPConfig, feats_list, device):
    if cfg.arch == "shared_face":
        return collate_shared(feats_list, device)
    return collate_states(feats_list, device)


def forward_batch(net, batch, masks):
    """Run either architecture on an already-collated batch. `masks`: (B, n_actions) bool."""
    if net.cfg.arch == "shared_face":
        static, seq, lengths = batch
        return net(static, seq, lengths, masks.to(static.device))
    hand, dice_counts, seq, lengths = batch
    return net(hand, dice_counts, seq, lengths)


def forward_states(net, feats_list, masks, device):
    return forward_batch(net, collate_for(net.cfg, feats_list, device), masks)


def _single_forward(net, feats, mask, device):
    with torch.no_grad():
        return forward_states(net, [feats], mask.unsqueeze(0), device).squeeze(0).cpu()


# ---------- Bots ----------
class NFSPTrainingBot(Bot):
    """Used only during self-play training. Samples a mode ('br' or 'avg') for the whole
    episode (anticipatory dynamics) and records its trajectory for later transition/SL
    reservoir construction by the training loop."""
    name = "NFSPTrain"

    def __init__(self, q_net, sl_net, cfg: NFSPConfig, mode: str, epsilon: float, device):
        self.q_net = q_net
        self.sl_net = sl_net
        self.cfg = cfg
        self.mode = mode
        self.epsilon = epsilon
        self.device = device
        self.trajectory = []   # (feats, action_idx, mask)
        self.sl_samples = []   # (feats, action_idx, mask) -- only greedy best-response picks

    def act(self, state: GameState):
        mask = legal_action_mask(state, self.cfg)
        feats = encode_state(state, self.cfg, mask)
        legal_idxs = mask.nonzero(as_tuple=True)[0]

        if self.mode == "avg":
            logits = _single_forward(self.sl_net, feats, mask, self.device)
            logits = logits.masked_fill(~mask, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1).item()
        else:
            if random.random() < self.epsilon:
                idx = legal_idxs[random.randrange(len(legal_idxs))].item()
            else:
                q = _single_forward(self.q_net, feats, mask, self.device)
                q = q.masked_fill(~mask, float("-inf"))
                idx = torch.argmax(q).item()
                self.sl_samples.append((feats, idx, mask))

        self.trajectory.append((feats, idx, mask))
        return action_to_bid(idx, self.cfg)


class NFSPBot(Bot):
    """Deployable bot: acts via the average-policy (SL) network only. Samples from the
    masked distribution rather than argmax-ing, since argmax would collapse the learned
    mixed/bluffing strategy back into an exploitable pure strategy."""
    name = "NFSP"

    def __init__(self, sl_net, cfg: NFSPConfig, device="cpu", sample: bool = True):
        self.sl_net = sl_net
        self.cfg = cfg
        self.device = device
        self.sample = sample

    @classmethod
    def from_checkpoint(cls, path: str, device="cpu", sample: bool = True):
        ckpt = torch.load(path, map_location=device)
        cfg = NFSPConfig(**ckpt["config"])
        net = build_net(cfg).to(device)
        net.load_state_dict(ckpt["sl_net"])
        net.eval()
        return cls(net, cfg, device=device, sample=sample)

    def act(self, state: GameState):
        mask = legal_action_mask(state, self.cfg)
        feats = encode_state(state, self.cfg, mask)
        logits = _single_forward(self.sl_net, feats, mask, self.device)
        logits = logits.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, 1).item() if self.sample else torch.argmax(probs).item()
        return action_to_bid(idx, self.cfg)
