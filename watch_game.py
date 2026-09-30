"""Play one game between two NFSP bot instances and print, at every decision, the top-K
action probabilities the acting network assigned -- a quick intuition tool for *why* it
picked what it picked, not just what it picked. Uses the deployed average-policy (SL) net,
sampling the same way NFSPBot does.

Usage:
    python3 watch_game.py
    python3 watch_game.py --checkpoint-a checkpoints/shared_face/nfsp_ep458000.pt \\
                          --checkpoint-b checkpoints/shared_face/nfsp_ep30000.pt --seed 7
"""
import argparse
import os
import random

import torch

from game import Game, PlayerState
from nfsp_model import NFSPConfig, action_to_bid, build_net, encode_state, forward_states, legal_action_mask

DEFAULT_CKPT = "checkpoints/shared_face/nfsp_ep458000.pt"


def load_net(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = NFSPConfig(**ckpt["config"])
    net = build_net(cfg).to(device)
    net.load_state_dict(ckpt["sl_net"])
    net.eval()
    return net, cfg


class ExplainedBot:
    """Acts via the average-policy net (same as NFSPBot), printing the top-K action
    distribution at every decision before sampling from it."""

    def __init__(self, name, net, cfg, top_k=5, sample=True, device="cpu"):
        self.name, self.net, self.cfg = name, net, cfg
        self.top_k, self.sample, self.device = top_k, sample, device

    def act(self, state):
        mask = legal_action_mask(state, self.cfg)
        feats = encode_state(state, self.cfg, mask)
        with torch.no_grad():
            logits = forward_states(self.net, [feats], mask.unsqueeze(0), self.device).squeeze(0)
        logits = logits.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(logits, dim=-1)

        k = min(self.top_k, int(mask.sum()))
        top_p, top_i = torch.topk(probs, k)
        idx = int(torch.multinomial(probs, 1)) if self.sample else int(torch.argmax(probs))
        chosen = action_to_bid(idx, self.cfg)

        hand = sorted(state.my_dice)
        bid_str = str(state.current_bid) if state.current_bid else "(none yet)"
        print(f"[{self.name}] hand={hand}  current bid: {bid_str}")
        for p, i in zip(top_p.tolist(), top_i.tolist()):
            marker = " *" if i == idx else "  "
            print(f"  {p * 100:5.1f}%  {action_to_bid(i, self.cfg)}{marker}")
        print(f"  -> chooses {chosen}")
        return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-a", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--checkpoint-b", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--dice", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--greedy", action="store_true", help="argmax instead of sampling")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    net_a, cfg_a = load_net(args.checkpoint_a)
    net_b, cfg_b = load_net(args.checkpoint_b)
    name_a = f"A({os.path.basename(args.checkpoint_a)})"
    name_b = f"B({os.path.basename(args.checkpoint_b)})"

    bot_a = ExplainedBot(name_a, net_a, cfg_a, top_k=args.top_k, sample=not args.greedy)
    bot_b = ExplainedBot(name_b, net_b, cfg_b, top_k=args.top_k, sample=not args.greedy)
    players = [PlayerState(name_a, bot=bot_a), PlayerState(name_b, bot=bot_b)]

    print(f"=== {name_a} vs {name_b}, {args.dice} dice each ===\n")
    game = Game(players, starting_dice=args.dice, verbose=True)
    winner = game.play()
    print(f"\n=== {players[winner].name} wins! ===")


if __name__ == "__main__":
    main()
