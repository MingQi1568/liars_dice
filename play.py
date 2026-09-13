"""Interactive Liar's Dice — play against the best bot.

Rules:
- Each player starts with 5 dice. 1s are wild (count as any face).
- On your turn: either raise the bid, or call LIAR on the previous bid.
- A bid is "quantity face", e.g. 3 4s means at least three 4s exist across all dice (including 1s as wild).
- A raise must increase quantity, or keep quantity and raise the face. Faces 2-6 (1s wild can't be the named face).
- If you call LIAR: count actual dice of that face (incl. wilds). If the bid was true, you lose a die. If it was a lie, bidder loses a die.
- Last player with dice wins.
"""
import random
import sys
from game import Game, PlayerState, Bid, Bot, GameState


class HumanBot(Bot):
    name = "Human"

    def __init__(self, label="You"):
        self.label = label

    def act(self, state: GameState):
        # Display state
        print()
        print(f"== Your turn ({self.label}) ==")
        print(f"Your dice: {sorted(state.my_dice)}")
        parts = []
        for i, c in enumerate(state.dice_counts):
            if c == 0:
                continue
            label = "You" if i == state.my_index else f"Bot@P{i}"
            parts.append(f"{label}:{c}")
        print(f"Dice per player: {', '.join(parts)}  (total {state.total_dice})")
        if state.current_bid is None:
            print("No bid yet — you start.")
        else:
            print(f"Current bid: {state.current_bid}")

        while True:
            prompt = "Enter bid as 'Q F' (quantity face), or 'liar' to challenge: " if state.current_bid is not None else "Enter your opening bid as 'Q F': "
            try:
                raw = input(prompt).strip().lower()
            except EOFError:
                print("\nGoodbye.")
                sys.exit(0)
            if raw in ("l", "liar", "challenge", "c"):
                if state.current_bid is None:
                    print("There is no bid yet. You must make one.")
                    continue
                return "liar"
            parts = raw.replace(",", " ").split()
            if len(parts) != 2:
                print("Format: '<quantity> <face>', e.g. '3 4' for three 4s.")
                continue
            try:
                q = int(parts[0])
                f = int(parts[1])
            except ValueError:
                print("Quantity and face must be integers.")
                continue
            if f < 2 or f > 6:
                print("Face must be 2-6 (1s are wild and can't be the named face).")
                continue
            if q < 1:
                print("Quantity must be >= 1.")
                continue
            bid = Bid(q, f)
            if state.current_bid is not None and not (bid > state.current_bid):
                print(f"Bid must beat {state.current_bid}: higher quantity, or same quantity with higher face.")
                continue
            return bid


class AnnouncingGame(Game):
    """Game that announces bot actions for the human to see."""
    def __init__(self, players, starting_dice=5, human_index=0):
        super().__init__(players, starting_dice=starting_dice, verbose=False)
        self.human_index = human_index

    def play_round(self):
        for p in self.players:
            if p.alive:
                p.roll()
        self.current_bid = None
        self.bid_history = []
        self.last_bidder = None

        print(f"\n----- New round. Dice counts: {[(p.name, p.num_dice) for p in self.players if p.alive]} -----")

        while True:
            player = self.players[self.current_player]
            if not player.alive:
                self.current_player = self.next_alive_player(self.current_player)
                continue

            state = self.make_state_view(self.current_player)
            action = player.bot.act(state)

            if action == "liar":
                if self.current_bid is None:
                    raise ValueError(f"{player.name} called liar with no bid")
                actual = self.count_face(self.current_bid.face)
                bid_valid = actual >= self.current_bid.quantity
                all_dice = []
                for p in self.players:
                    if p.alive:
                        all_dice.extend(p.dice)
                print(f"\n  >>> {player.name} calls LIAR on {self.current_bid}.")
                print(f"  All dice (sorted): {sorted(all_dice)}")
                print(f"  Count of {self.current_bid.face}s (incl. 1s as wild): {actual}")

                def who(idx):
                    return self.players[idx].name

                # Variant: winner of challenge loses a die (toward goal of 0)
                if bid_valid:
                    challenge_winner = self.last_bidder
                    challenge_loser = self.current_player
                    print(f"  Bid was TRUE — {who(challenge_winner)} wins the challenge.")
                else:
                    challenge_winner = self.current_player
                    challenge_loser = self.last_bidder
                    print(f"  Bid was a LIE — {who(challenge_winner)} wins the challenge.")
                verb = "loses" if challenge_winner != self.human_index else "lose"
                pronoun = who(challenge_winner) if challenge_winner != self.human_index else "You"
                self.players[challenge_winner].dice.pop()
                print(f"  {pronoun} {verb} a die ({self.players[challenge_winner].num_dice} left).")
                if self.players[challenge_winner].num_dice == 0:
                    print(f"  *** {pronoun} reached 0 dice and WINS THE GAME! ***")
                # Variant: loser opens next round
                self.current_player = challenge_loser
                return challenge_winner
            else:
                new_bid: Bid = action
                self.current_bid = new_bid
                self.bid_history.append((self.current_player, new_bid))
                self.last_bidder = self.current_player
                print(f"  {player.name} bids {new_bid}")
                self.current_player = self.next_alive_player(self.current_player)


def play_game(num_bots=2, starting_dice=5, seed=None):
    if seed is not None:
        random.seed(seed)
    from best_bot import build_best_bot

    human = PlayerState(name="You", bot=HumanBot("You"))
    bots = [PlayerState(name=f"Bot{i+1}", bot=build_best_bot()) for i in range(num_bots)]
    players = [human] + bots
    # Random seating
    random.shuffle(players)
    human_index = next(i for i, p in enumerate(players) if isinstance(p.bot, HumanBot))

    game = AnnouncingGame(players, starting_dice=starting_dice, human_index=human_index)
    winner = game.play()
    print()
    if players[winner].bot.__class__.__name__ == "HumanBot":
        print("*** YOU WIN! ***")
    else:
        print(f"*** {players[winner].name} (the bot) wins. ***")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bots", type=int, default=2, help="Number of bot opponents (default 2)")
    p.add_argument("--dice", type=int, default=5, help="Starting dice per player (default 5)")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    args = p.parse_args()
    play_game(num_bots=args.bots, starting_dice=args.dice, seed=args.seed)
