"""Stateful game driver for live play.

Each invocation either starts a game or advances it one human-turn.
State is pickled between invocations.

Usage:
  python3 liveplay.py new [seed] [dice]   # start a game vs MasterBot
  python3 liveplay.py step "3 4"          # bid 3 fours
  python3 liveplay.py step liar           # call LIAR
"""
import sys
import pickle
import random
from dataclasses import dataclass, field
from game import Game, PlayerState, Bid
from best_bot import build_best_bot

STATE_FILE = "/tmp/liars_dice_live.pkl"


@dataclass
class LiveState:
    game: Game
    human_idx: int
    transcript: list[str] = field(default_factory=list)
    finished: bool = False
    winner: str = ""


def log(state: LiveState, msg: str):
    state.transcript.append(msg)


def show(state: LiveState):
    """Print the visible state to me."""
    g = state.game
    print(f"\n=== Game state ===")
    print(f"Players: " + ", ".join(
        f"{p.name}({p.num_dice}d)" + (" [ME]" if i == state.human_idx else "")
        for i, p in enumerate(g.players) if p.alive
    ))
    print(f"Total dice in play: {g.total_dice()}")
    if state.finished:
        print(f"FINISHED. Winner: {state.winner}")
        print("\n--- Transcript ---")
        for line in state.transcript:
            print(line)
        return

    my_dice = sorted(g.players[state.human_idx].dice)
    print(f"My dice: {my_dice}  (count by face: " +
          ", ".join(f"{f}:{my_dice.count(f)}" for f in range(1, 7) if my_dice.count(f) > 0) + ")")
    if g.current_bid:
        print(f"Current bid: {g.current_bid} (by {g.players[g.last_bidder].name})")
    else:
        print("No bid yet — I open.")
    print(f"\n--- Bid history this round ---")
    if not g.bid_history:
        print("  (none)")
    for bidder_idx, bid in g.bid_history:
        marker = " [ME]" if bidder_idx == state.human_idx else ""
        print(f"  {g.players[bidder_idx].name}{marker} bid {bid}")
    print(f"\n--- Recent log ---")
    for line in state.transcript[-8:]:
        print(line)


def advance_bots(state: LiveState):
    """Run bot turns until it's the human's turn or the round/game ends."""
    g = state.game
    while True:
        if state.finished:
            return
        player = g.players[g.current_player]
        if not player.alive:
            g.current_player = g.next_alive_player(g.current_player)
            continue
        if g.current_player == state.human_idx:
            return  # human's turn
        action = player.bot.act(g.make_state_view(g.current_player))
        if action == "liar":
            actual = g.count_face(g.current_bid.face)
            bid_valid = actual >= g.current_bid.quantity
            all_dice = [d for p in g.players if p.alive for d in p.dice]
            log(state, f"{player.name} calls LIAR on {g.current_bid}. "
                       f"All dice: {sorted(all_dice)}. Count {g.current_bid.face}s+wilds: {actual}. "
                       + ("BID TRUE." if bid_valid else "BID LIE."))
            challenge_loser = g.current_player if bid_valid else g.last_bidder
            challenge_winner = g.last_bidder if bid_valid else g.current_player
            # Variant: winner of challenge loses a die
            g.players[challenge_winner].dice.pop()
            log(state, f"  {g.players[challenge_winner].name} wins challenge and loses a die "
                       f"({g.players[challenge_winner].num_dice} left).")
            # Variant: first to 0 wins the game
            if g.players[challenge_winner].num_dice == 0:
                state.finished = True
                state.winner = g.players[challenge_winner].name
                log(state, f"  *** {g.players[challenge_winner].name} reached 0 dice — WINS! ***")
                return
            # Start next round (loser of challenge opens)
            g.current_bid = None
            g.bid_history = []
            g.last_bidder = None
            for p in g.players:
                if p.alive:
                    p.roll()
            log(state, f"--- new round, dice counts: " +
                       ", ".join(f"{p.name}:{p.num_dice}" for p in g.players if p.alive) + " ---")
            g.current_player = challenge_loser
        else:
            g.current_bid = action
            g.bid_history.append((g.current_player, action))
            g.last_bidder = g.current_player
            log(state, f"{player.name} bids {action}")
            g.current_player = g.next_alive_player(g.current_player)


def new_game(seed=None, dice=5):
    if seed is None:
        seed = random.randint(0, 1_000_000)
    random.seed(seed)
    print(f"Seed: {seed}")
    # Random seating
    bots = [build_best_bot()]
    players_seq = ["Master", "Claude"]
    random.shuffle(players_seq)
    players = []
    for name in players_seq:
        if name == "Master":
            players.append(PlayerState(name="Master", bot=build_best_bot()))
        else:
            players.append(PlayerState(name="Claude", bot=None))  # human-controlled
    g = Game(players, starting_dice=dice, verbose=False)
    # Roll initial
    for p in g.players:
        p.roll()
    human_idx = next(i for i, p in enumerate(g.players) if p.bot is None)
    state = LiveState(game=g, human_idx=human_idx)
    log(state, f"--- new round, dice counts: " +
               ", ".join(f"{p.name}:{p.num_dice}" for p in g.players if p.alive) + " ---")
    advance_bots(state)
    save(state)
    show(state)


def save(state):
    with open(STATE_FILE, "wb") as f:
        pickle.dump(state, f)


def load() -> LiveState:
    with open(STATE_FILE, "rb") as f:
        return pickle.load(f)


def step(action_str: str):
    state = load()
    if state.finished:
        print("Game already finished.")
        show(state)
        return
    g = state.game
    if g.current_player != state.human_idx:
        print("Not my turn? Advancing bots first.")
        advance_bots(state)
        save(state)
        show(state)
        return
    action_str = action_str.strip().lower()
    if action_str in ("liar", "l", "challenge"):
        if g.current_bid is None:
            print("Cannot call liar with no bid.")
            return
        actual = g.count_face(g.current_bid.face)
        bid_valid = actual >= g.current_bid.quantity
        all_dice = [d for p in g.players if p.alive for d in p.dice]
        log(state, f"Claude calls LIAR on {g.current_bid}. "
                   f"All dice: {sorted(all_dice)}. Count {g.current_bid.face}s+wilds: {actual}. "
                   + ("BID TRUE." if bid_valid else "BID LIE."))
        challenge_loser = state.human_idx if bid_valid else g.last_bidder
        challenge_winner = g.last_bidder if bid_valid else state.human_idx
        # Variant: winner of challenge loses a die
        g.players[challenge_winner].dice.pop()
        log(state, f"  {g.players[challenge_winner].name} wins challenge and loses a die "
                   f"({g.players[challenge_winner].num_dice} left).")
        # Variant: first to 0 wins
        if g.players[challenge_winner].num_dice == 0:
            state.finished = True
            state.winner = g.players[challenge_winner].name
            log(state, f"  *** {g.players[challenge_winner].name} reached 0 dice — WINS! ***")
            save(state)
            show(state)
            return
        # next round (loser of challenge opens)
        g.current_bid = None
        g.bid_history = []
        g.last_bidder = None
        for p in g.players:
            if p.alive:
                p.roll()
        log(state, f"--- new round, dice counts: " +
                   ", ".join(f"{p.name}:{p.num_dice}" for p in g.players if p.alive) + " ---")
        g.current_player = challenge_loser
        advance_bots(state)
        save(state)
        show(state)
        return

    # Parse bid
    parts = action_str.replace(",", " ").split()
    if len(parts) != 2:
        print("Format: 'Q F' or 'liar'")
        return
    try:
        q, f = int(parts[0]), int(parts[1])
    except ValueError:
        print("Q and F must be integers.")
        return
    if f < 2 or f > 6 or q < 1:
        print("Face must be 2-6, quantity >= 1.")
        return
    bid = Bid(q, f)
    if g.current_bid is not None and not (bid > g.current_bid):
        print(f"Bid {bid} doesn't beat {g.current_bid}.")
        return
    g.current_bid = bid
    g.bid_history.append((state.human_idx, bid))
    g.last_bidder = state.human_idx
    log(state, f"Claude bids {bid}")
    g.current_player = g.next_alive_player(state.human_idx)
    advance_bots(state)
    save(state)
    show(state)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "new":
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
        dice = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        new_game(seed=seed, dice=dice)
    elif cmd == "step":
        step(" ".join(sys.argv[2:]))
    elif cmd == "show":
        show(load())
    else:
        print(__doc__)
        sys.exit(1)
