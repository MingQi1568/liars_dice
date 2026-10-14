"""Policy + value network for R-NaD.

Same face-shared design as nfsp_model.SharedFaceNet (one small MLP scores every candidate
face 2-6 with shared weights; separate face-1 and liar heads), but it outputs policy LOGITS
and a scalar VALUE, and reads the bid history as a fixed window of the last K bids run
through a masked GRU cell, so a whole batch is processed with plain tensor ops.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vec_env import S_CLAIM, S_FACE1, S_FACES, S_GLOBAL, S_LIAR, STATIC_DIM, Spec

FACE_EMB, Q_EMB, SCORE_HIDDEN, LIAR_HIDDEN, VALUE_HIDDEN = 16, 8, 64, 32, 64


FACE_RANK = 6           # per-face rank inputs: normalized rank (f-2)/4 plus a one-hot of faces 2..6


class RNaDNet(nn.Module):
    """face_rank: give the shared per-face MLP the face's rank in the bid order (the only thing that
    distinguishes faces 2-6; probabilities are symmetric), so it can treat faces differently where the
    ordering matters while still sharing weights. info_ctx > 0: add a small MLP over the exact round
    encoding (vec_env.info_features) to the context every head sees - full history with real faces."""

    def __init__(self, spec: Spec, gru_hidden: int = 64, score_hidden: int = SCORE_HIDDEN,
                 face_rank: bool = False, info_ctx: int = 0):
        super().__init__()
        self.spec, self.gru_hidden, self.face_rank, self.info_ctx = spec, gru_hidden, face_rank, info_ctx
        G = gru_hidden + 4 + info_ctx
        self.G = G
        score_in = G + FACE_EMB + Q_EMB + 2
        self.cell = nn.GRUCell(spec.bid_dim, gru_hidden)
        self.shared_face_mlp = nn.Sequential(nn.Linear(10 + (FACE_RANK if face_rank else 0), FACE_EMB), nn.ReLU(),
                                             nn.Linear(FACE_EMB, FACE_EMB))
        if face_rank:
            rank = torch.cat([torch.arange(5, dtype=torch.float32)[:, None] / 4.0, torch.eye(5)], dim=1)   # (5, 6)
            self.register_buffer("rank_feats", rank)
            self.face_invariant_start()
        if info_ctx:
            self.info_mlp = nn.Sequential(nn.Linear(spec.info_dim, 2 * info_ctx), nn.ReLU(),
                                          nn.Linear(2 * info_ctx, info_ctx), nn.ReLU())
        self.face1_mlp = nn.Sequential(nn.Linear(8, FACE_EMB), nn.ReLU(), nn.Linear(FACE_EMB, FACE_EMB))
        self.quantity_embedding = nn.Embedding(spec.qmax, Q_EMB)
        self.q_score = nn.Sequential(nn.Linear(score_in, score_hidden), nn.ReLU(), nn.Linear(score_hidden, 2))
        self.face1_score = nn.Sequential(nn.Linear(score_in, score_hidden), nn.ReLU(), nn.Linear(score_hidden, 2))
        self.liar_head = nn.Sequential(nn.Linear(G + 8 + FACE_EMB + 2, LIAR_HIDDEN), nn.ReLU(), nn.Linear(LIAR_HIDDEN, 2))
        self.value_head = nn.Sequential(
            nn.Linear(G + FACE_EMB + FACE_EMB + 8 + FACE_EMB, VALUE_HIDDEN), nn.ReLU(), nn.Linear(VALUE_HIDDEN, 1))

    def face_invariant_start(self, haiku: bool = False):
        """Zero the rank inputs' weights, so the net starts exactly face-invariant and asymmetry grows only
        where gradients push it, and initialize the layer's other inputs like a plain 10-input layer
        (PyTorch's default init, or Haiku's when haiku=True). Call again after re-initializing weights."""
        first = self.shared_face_mlp[0]
        plain = nn.Linear(10, FACE_EMB)
        with torch.no_grad():
            if haiku:
                std = 1.0 / math.sqrt(10)
                nn.init.trunc_normal_(plain.weight, std=std, a=-2 * std, b=2 * std)
                nn.init.zeros_(plain.bias)
            first.weight.zero_()
            first.weight[:, :10].copy_(plain.weight)
            first.bias.copy_(plain.bias)

    def parts(self, static, win, wmask):
        B = static.shape[0]
        h = static.new_zeros(B, self.gru_hidden)
        for k in range(self.spec.window):
            h = torch.where(wmask[:, k: k + 1], self.cell(win[:, k], h), h)
        scal = static[:, S_GLOBAL]
        ctx = torch.cat([h, scal], dim=-1)
        if self.info_ctx:
            ctx = torch.cat([ctx, self.info_mlp(static[:, STATIC_DIM:])], dim=-1)
        fi = static[:, S_FACES].reshape(B, 5, 10)
        fin = torch.cat([fi, self.rank_feats.expand(B, 5, FACE_RANK)], dim=-1) if self.face_rank else fi
        e_f = self.shared_face_mlp(fin)
        e_1 = self.face1_mlp(static[:, S_FACE1])
        e_all = torch.cat([e_1[:, None], e_f], dim=1)
        e_claim = (static[:, S_CLAIM][:, :, None] * e_all).sum(1)
        return ctx, fi, scal, e_f, e_1, e_claim

    def _scores(self, mlp, ctx, e, gap, ratio):
        lin1, lin2 = mlp[0], mlp[2]
        W, G = lin1.weight, ctx.shape[1]
        pre = (F.linear(ctx, W[:, :G], lin1.bias)[:, None, None, :]
               + F.linear(e, W[:, G: G + FACE_EMB])[:, :, None, :]
               + F.linear(self.quantity_embedding.weight, W[:, G + FACE_EMB: G + FACE_EMB + Q_EMB])[None, None]
               + gap[..., None] * W[:, G + FACE_EMB + Q_EMB]
               + ratio[..., None] * W[:, G + FACE_EMB + Q_EMB + 1])
        return F.linear(F.relu(pre), lin2.weight, lin2.bias)          # (..., 2): [logit, q]

    def heads(self, static, win, wmask, mask, parts=None):
        """Returns (logits [B,A], q [B,A]); q is an auxiliary per-action critic (used only in critic mode)."""
        ctx, fi, scal, e_f, e_1, e_claim = parts if parts is not None else self.parts(static, win, wmask)
        B, spec = static.shape[0], self.spec
        Q, dc = spec.qmax, float(spec.dice)
        qs = torch.arange(1, Q + 1, device=static.device, dtype=static.dtype)
        opp = (scal[:, 1] * dc).view(B, 1, 1)

        def extras(eff):
            gap = qs.view(1, 1, Q) - eff.unsqueeze(-1)
            ratio = (gap.clamp(min=0) / opp).clamp(max=2.0) / 2.0
            return gap / Q, ratio

        gap_f, ratio_f = extras(fi[:, :, 0] * dc)
        s_f = self._scores(self.q_score, ctx, e_f, gap_f, ratio_f)                        # (B,5,Q,2)
        gap_1, ratio_1 = extras((scal[:, 2] * dc).unsqueeze(-1))
        s_1 = self._scores(self.face1_score, ctx, e_1[:, None], gap_1, ratio_1)[:, 0]     # (B,Q,2)
        flat = torch.cat([s_1.unsqueeze(2), s_f.permute(0, 2, 1, 3)], dim=2).reshape(B, Q * 6, 2)

        raise_mask = mask[:, : spec.n_raise]
        with torch.no_grad():
            best = flat[..., 0].detach().masked_fill(~raise_mask, float("-inf")).amax(1)
            best = torch.where(torch.isfinite(best), best, torch.zeros_like(best))
        n_legal = raise_mask.to(flat.dtype).sum(1) / spec.n_raise
        liar = self.liar_head(torch.cat([ctx, static[:, S_LIAR], e_claim, best[:, None], n_legal[:, None]], dim=-1))
        return torch.cat([flat[..., 0], liar[:, :1]], -1), torch.cat([flat[..., 1], liar[:, 1:]], -1)

    def policy_logits(self, static, win, wmask, mask, parts=None):
        return self.heads(static, win, wmask, mask, parts)[0]

    def value(self, static, win, wmask, parts=None):
        ctx, fi, scal, e_f, e_1, e_claim = parts if parts is not None else self.parts(static, win, wmask)
        v_in = torch.cat([ctx, e_f.sum(1), e_1, static[:, S_LIAR], e_claim], dim=-1)
        return self.value_head(v_in).squeeze(-1)

    def forward(self, static, win, wmask, mask):
        parts = self.parts(static, win, wmask)
        return self.policy_logits(static, win, wmask, mask, parts), self.value(static, win, wmask, parts)


class InfoSetNet(nn.Module):
    """The reference implementation's architecture: an MLP torso (two ReLU layers) on the exact per-round
    information-set encoding (vec_env.info_features, stored after STATIC_DIM in `static`), with a linear
    policy head and a linear value head. Same interface as RNaDNet; `win`/`wmask` are unused."""

    def __init__(self, spec: Spec, hidden: int = 256):
        super().__init__()
        self.spec = spec
        self.torso = nn.Sequential(nn.Linear(spec.info_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.pi_head = nn.Linear(hidden, spec.n_actions)
        self.v_head = nn.Linear(hidden, 1)

    def parts(self, static, win, wmask):
        return self.torso(static[:, STATIC_DIM:])

    def heads(self, static, win, wmask, mask, parts=None):
        h = parts if parts is not None else self.parts(static, win, wmask)
        logits = self.pi_head(h)
        return logits, torch.zeros_like(logits)

    def policy_logits(self, static, win, wmask, mask, parts=None):
        return self.heads(static, win, wmask, mask, parts)[0]

    def value(self, static, win, wmask, parts=None):
        h = parts if parts is not None else self.parts(static, win, wmask)
        return self.v_head(h).squeeze(-1)

    def forward(self, static, win, wmask, mask):
        h = self.parts(static, win, wmask)
        return self.policy_logits(static, win, wmask, mask, h), self.value(static, win, wmask, h)
