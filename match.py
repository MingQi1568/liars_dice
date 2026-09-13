"""Play Claude vs Master, collect stats. Plays N 1v1 games and reports."""
import random
from dataclasses import dataclass, field
from game import Game, PlayerState, Bid
from best_bot import build_best_bot
from claude_bot import ClaudeBot


@dataclass
class GameStats:
    winner: str
    rounds: int
    dice_left_at_win: int
    liar_calls_by: dict[str, int] = field(default_factory=dict)
    correct_liars_by: dict[str, int] = field(default_factory=dict)
    bluffs_made_by: dict[str, int] = field(default_factory=dict)  # times bot was caught lying
    bids_made_by: dict[str, int] = field(default_factory=dict)


class TrackedGame(Game):
    """Game wrapper that records per-game stats."""
    def __init__(self, players, starting_dice=5):
        super().__init__(players, starting_dice=starting_dice, verbose=False)
        self.rounds = 0
        self.liar_calls_by = {p.name: 0 for p in players}
        self.correct_liars_by = {p.name: 0 for p in players}
        self.bluffs_caught_by = {p.name: 0 for p in players}  # they got caught lying
        self.bids_made_by = {p.name: 0 for p in players}

    def play_round(self):
        for p in self.players:
            if p.alive:
                p.roll()
        self.current_bid = None
        self.bid_history = []
        self.last_bidder = None
        self.rounds += 1

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
                self.liar_calls_by[player.name] += 1
                actual = self.count_face(self.current_bid.face)
                bid_valid = actual >= self.current_bid.quantity
                if bid_valid:
                    challenge_loser = self.current_player
                    challenge_winner = self.last_bidder
                else:
                    challenge_loser = self.last_bidder
                    challenge_winner = self.current_player
                    self.correct_liars_by[player.name] += 1
                    self.bluffs_caught_by[self.players[self.last_bidder].name] += 1
                # Variant: winner of challenge loses a die; loser opens next round
                self.players[challenge_winner].dice.pop()
                self.current_player = challenge_loser
                return challenge_winner
            else:
                new_bid: Bid = action
                self.current_bid = new_bid
                self.bid_history.append((self.current_player, new_bid))
                self.last_bidder = self.current_player
                self.bids_made_by[player.name] += 1
                self.current_player = self.next_alive_player(self.current_player)


def run_match(num_games=10, starting_dice=5, seed=0, verbose=False):
    random.seed(seed)
    all_stats = []
    for i in range(num_games):
        # Alternate seating to avoid first-player bias
        if i % 2 == 0:
            players = [
                PlayerState(name="Claude", bot=ClaudeBot()),
                PlayerState(name="Master", bot=build_best_bot()),
            ]
        else:
            players = [
                PlayerState(name="Master", bot=build_best_bot()),
                PlayerState(name="Claude", bot=ClaudeBot()),
            ]
        g = TrackedGame(players, starting_dice=starting_dice)
        winner_idx = g.play()
        winner = players[winner_idx]
        # Variant: winner ended at 0 dice; loser still has some
        loser_dice = max((p.num_dice for p in players if p is not winner), default=0)
        gs = GameStats(
            winner=winner.name,
            rounds=g.rounds,
            dice_left_at_win=loser_dice,  # opponent's remaining dice when game ended
            liar_calls_by=dict(g.liar_calls_by),
            correct_liars_by=dict(g.correct_liars_by),
            bluffs_made_by=dict(g.bluffs_caught_by),
            bids_made_by=dict(g.bids_made_by),
        )
        all_stats.append(gs)
        if verbose:
            print(f"Game {i+1}: winner={gs.winner}, rounds={gs.rounds}, "
                  f"dice_left={gs.dice_left_at_win}")
    return all_stats


def summarize(stats):
    n = len(stats)
    claude_wins = sum(1 for s in stats if s.winner == "Claude")
    master_wins = n - claude_wins

    # Average dice left at win, per winner side
    claude_win_dice = [s.dice_left_at_win for s in stats if s.winner == "Claude"]
    master_win_dice = [s.dice_left_at_win for s in stats if s.winner == "Master"]

    avg = lambda xs: sum(xs) / len(xs) if xs else 0

    print(f"\n{'='*55}")
    print(f"  {n} games, Claude vs MasterBot (1v1, 5 dice each)")
    print(f"{'='*55}\n")
    print(f"  Score: Claude {claude_wins} — {master_wins} Master")
    print(f"  Claude win rate: {claude_wins/n*100:.1f}%")
    print()
    print(f"  Avg dice left at win:")
    print(f"    Claude wins: {avg(claude_win_dice):.2f}  (n={len(claude_win_dice)})")
    print(f"    Master wins: {avg(master_win_dice):.2f}  (n={len(master_win_dice)})")
    print()
    avg_rounds = avg([s.rounds for s in stats])
    print(f"  Avg rounds per game: {avg_rounds:.1f}")
    print(f"    Shortest: {min(s.rounds for s in stats)}, longest: {max(s.rounds for s in stats)}")

    # Aggregate liar-call stats
    def total(key, who):
        return sum(getattr(s, key).get(who, 0) for s in stats)

    cl_calls = total("liar_calls_by", "Claude")
    cl_correct = total("correct_liars_by", "Claude")
    ms_calls = total("liar_calls_by", "Master")
    ms_correct = total("correct_liars_by", "Master")

    print()
    print(f"  LIAR calls (across all games):")
    print(f"    Claude: {cl_calls} calls, {cl_correct} correct  "
          f"({cl_correct/cl_calls*100 if cl_calls else 0:.1f}% accuracy)")
    print(f"    Master: {ms_calls} calls, {ms_correct} correct  "
          f"({ms_correct/ms_calls*100 if ms_calls else 0:.1f}% accuracy)")

    cl_bluffs = total("bluffs_made_by", "Claude")
    ms_bluffs = total("bluffs_made_by", "Master")
    cl_bids = total("bids_made_by", "Claude")
    ms_bids = total("bids_made_by", "Master")
    print()
    print(f"  Bids made / bluffs caught:")
    print(f"    Claude: {cl_bids} bids, {cl_bluffs} caught bluffing "
          f"({cl_bluffs/cl_bids*100 if cl_bids else 0:.1f}%)")
    print(f"    Master: {ms_bids} bids, {ms_bluffs} caught bluffing "
          f"({ms_bluffs/ms_bids*100 if ms_bids else 0:.1f}%)")

    print()
    print(f"  Per-game results:")
    for i, s in enumerate(stats):
        marker = "✓" if s.winner == "Claude" else " "
        print(f"    [{marker}] Game {i+1:2d}: {s.winner:6s} won, {s.rounds:2d} rounds, "
              f"{s.dice_left_at_win} dice left")


if __name__ == "__main__":
    stats = run_match(num_games=10, starting_dice=5, seed=2026, verbose=False)
    summarize(stats)
