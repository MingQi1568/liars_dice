"""Real-time Liar's Dice coach.

Mirror the live game state, get MasterBot's recommended move plus top
alternative bids with their (signal-adjusted) probabilities of being true.

Usage:
  python3 coach.py

You configure starting dice and who opens. Each round you enter the dice
you rolled, then on each turn you either type your friend's bid (e.g. '3 4')
or 'liar', and on your turns you accept Master's suggestion or override.
After a LIAR call, you tell the coach who lost a die so it tracks counts.
"""
import sys
from game import Bid, GameState
from best_bot import build_best_bot
from bots import next_bid_options, count_face_in_hand


def prompt(msg, default=None):
    if default is not None:
        s = input(f"{msg} [{default}]: ").strip()
        return s if s else str(default)
    return input(f"{msg}: ").strip()


def parse_dice(s):
    parts = s.replace(",", " ").split()
    dice = []
    for p in parts:
        try:
            d = int(p)
        except ValueError:
            raise ValueError(f"'{p}' isn't a die value")
        if d < 1 or d > 6:
            raise ValueError(f"Die must be 1-6, got {d}")
        dice.append(d)
    return dice


def parse_bid_or_liar(s):
    s = s.strip().lower()
    if s in ("liar", "l", "challenge", "c"):
        return "liar"
    parts = s.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError("Format is 'Q F' (e.g. '3 4') or 'liar'")
    q, f = int(parts[0]), int(parts[1])
    if f < 1 or f > 6:
        raise ValueError("Face must be 1-6")
    if q < 1:
        raise ValueError("Quantity must be >= 1")
    return Bid(q, f)


def make_state(my_dice, my_dice_count, friend_dice_count, current_bid, bid_history, last_bidder, my_index):
    return GameState(
        my_dice=list(my_dice),
        my_index=my_index,
        num_players=2,
        dice_counts=[my_dice_count, friend_dice_count] if my_index == 0 else [friend_dice_count, my_dice_count],
        total_dice=my_dice_count + friend_dice_count,
        current_bid=current_bid,
        bid_history=list(bid_history),
        last_bidder=last_bidder,
    )


def show_analysis(bot, state):
    """Show what MasterBot would do + top alternatives + probabilities."""
    print()
    print("  ╭── MasterBot analysis ──╮")
    if state.current_bid is not None:
        p_call = bot.signal_adjusted_prob(state, state.current_bid)
        threshold = bot.call_threshold
        ratio = state.current_bid.quantity / max(1, state.total_dice)
        if ratio > bot.ratio_breakpoint:
            threshold += bot.high_ratio_bonus
        my_count = len(state.my_dice)
        if my_count <= 2:
            threshold += bot.low_dice_call_bonus
        alive = sum(1 for c in state.dice_counts if c > 0)
        if alive == 2:
            threshold += bot.is_1v1_bonus
        print(f"  Current bid {state.current_bid} adjusted P(true): {p_call:.1%}")
        print(f"  Threshold to call LIAR: {threshold:.1%}  →  "
              f"{'CALL LIAR' if p_call < threshold else 'do not call'}")

    # All legal raises ranked
    options = next_bid_options(state.current_bid)
    options = [b for b in options if b.quantity <= state.total_dice + 1]
    scored = []
    for b in options:
        p = bot.signal_adjusted_prob(state, b)
        in_hand = count_face_in_hand(state.my_dice, b.face)
        same_qty = state.current_bid is not None and b.quantity == state.current_bid.quantity
        # use MasterBot's bid-scoring weights
        score = p + 0.05 * in_hand - 0.03 * (b.quantity - (state.current_bid.quantity if state.current_bid else 0))
        if in_hand == 0:
            score -= 0.04
        scored.append((score, p, b, in_hand))
    scored.sort(reverse=True)

    print(f"  Top 6 raise options (sorted by Master's score):")
    print(f"    {'Bid':>6} {'P(true)':>8} {'in_hand':>8} {'note'}")
    for s, p, b, in_hand in scored[:6]:
        safe = "✓ safe" if p >= bot.bid_threshold else ("✗ bluff" if p < 0.3 else "~ risky")
        print(f"    {str(b):>6} {p:>7.1%} {in_hand:>8} {safe}")

    suggested = bot.act(state)
    print(f"  ➤ Suggestion: {suggested if isinstance(suggested, str) else suggested}")
    print(f"  ╰────────────────────────╯")
    return suggested


def play_round(bot, my_dice_count, friend_dice_count, my_starts):
    """Run one round. Returns ('me' or 'friend') indicating who lost a die."""
    my_index = 0
    friend_index = 1
    print()
    print(f"=" * 50)
    print(f"NEW ROUND — You: {my_dice_count} dice, Friend: {friend_dice_count} dice")
    print(f"=" * 50)
    while True:
        try:
            my_dice = parse_dice(prompt(f"Enter your {my_dice_count} dice values (space-separated, faces 1-6)"))
            if len(my_dice) != my_dice_count:
                print(f"Expected {my_dice_count} dice, got {len(my_dice)}. Try again.")
                continue
            break
        except ValueError as e:
            print(f"Error: {e}")

    counts_in_hand = ", ".join(
        f"{f}:{count_face_in_hand(my_dice, f)}" for f in range(2, 7)
    )
    print(f"Your wild-adjusted counts: {counts_in_hand}  (raw 1s: {my_dice.count(1)})")

    current_bid = None
    bid_history = []
    last_bidder = None
    current_player = my_index if my_starts else friend_index

    while True:
        state = make_state(my_dice, my_dice_count, friend_dice_count, current_bid, bid_history, last_bidder, my_index)

        if current_player == my_index:
            suggested = show_analysis(bot, state)
            while True:
                txt = input("Your action (press Enter to use suggestion, or override): ").strip()
                if not txt:
                    action = suggested
                else:
                    try:
                        action = parse_bid_or_liar(txt)
                        if isinstance(action, Bid) and current_bid is not None and not (action > current_bid):
                            print(f"Bid {action} doesn't beat {current_bid}.")
                            continue
                    except (ValueError, IndexError) as e:
                        print(f"Error: {e}")
                        continue
                break
            label = "You"
        else:
            print()
            while True:
                txt = input("Friend's move ('Q F' or 'liar'): ").strip()
                try:
                    action = parse_bid_or_liar(txt)
                    if isinstance(action, Bid) and current_bid is not None and not (action > current_bid):
                        print(f"Bid {action} doesn't beat {current_bid}.")
                        continue
                except (ValueError, IndexError) as e:
                    print(f"Error: {e}")
                    continue
                break
            label = "Friend"

        if action == "liar":
            if current_bid is None:
                print("Cannot call LIAR with no bid.")
                continue
            print(f"  >>> {label} calls LIAR on {current_bid}.")
            while True:
                outcome = input("Was the bid TRUE or LIE? (t/l): ").strip().lower()
                if outcome.startswith("t"):
                    bid_true = True
                    break
                elif outcome.startswith("l"):
                    bid_true = False
                    break
                else:
                    print("Enter 't' or 'l'.")
            # In this variant, the WINNER of the challenge loses a die (progress toward 0)
            # and the LOSER opens the next round.
            if bid_true:
                # bidder (last_bidder) wins, caller (current_player) loses
                winner = "me" if last_bidder == my_index else "friend"
                challenge_loser = "me" if current_player == my_index else "friend"
            else:
                # caller (current_player) wins, bidder (last_bidder) loses
                winner = "me" if current_player == my_index else "friend"
                challenge_loser = "me" if last_bidder == my_index else "friend"
            print(f"  {winner.upper()} wins the challenge → {winner.upper()} loses a die.")
            return challenge_loser  # who will open next round
        else:
            current_bid = action
            bid_history.append((current_player, current_bid))
            last_bidder = current_player
            print(f"  {label} bids {current_bid}")
            current_player = friend_index if current_player == my_index else my_index


def main():
    print("=" * 50)
    print(" LIAR'S DICE COACH (1s wild, 2-player)")
    print("=" * 50)
    bot = build_best_bot()
    print(f"Using MasterBot config: call_threshold={bot.call_threshold}, "
          f"bid_threshold={bot.bid_threshold}, signal_boost={bot.signal_boost}")
    print()

    while True:
        try:
            my_dice_count = int(prompt("How many dice do YOU start with?", "5"))
            break
        except ValueError:
            print("Enter an integer.")
    while True:
        try:
            friend_dice_count = int(prompt("How many dice does your FRIEND start with?", "5"))
            break
        except ValueError:
            print("Enter an integer.")

    while True:
        s = prompt("Who opens the first round? (me/friend)", "me").lower()
        if s.startswith("m"):
            my_turn_first = True
            break
        elif s.startswith("f"):
            my_turn_first = False
            break
        print("Enter 'me' or 'friend'.")

    round_num = 0
    while my_dice_count > 0 and friend_dice_count > 0:
        round_num += 1
        challenge_loser = play_round(bot, my_dice_count, friend_dice_count, my_turn_first)
        # Variant: WINNER of challenge loses a die (progress toward 0).
        # LOSER of challenge opens the next round.
        if challenge_loser == "me":
            # I lost the challenge → friend wins → friend loses a die. I open next round.
            friend_dice_count -= 1
            my_turn_first = True
        else:
            # Friend lost the challenge → I win → I lose a die. Friend opens next round.
            my_dice_count -= 1
            my_turn_first = False
        print(f"\n>>> End of round {round_num}. You: {my_dice_count} dice, Friend: {friend_dice_count} dice <<<")

    # Variant: first to 0 wins
    if my_dice_count == 0:
        print("\n*** YOU WIN! (reached 0 dice first) ***")
    else:
        print("\n*** Friend wins. (reached 0 dice first) ***")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")
        sys.exit(0)
