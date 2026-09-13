"""Tournament simulator for Liar's Dice bots."""
import random
import itertools
from collections import Counter
from game import Game, PlayerState


def run_game(bots, labels=None, starting_dice=5, verbose=False):
    """Run one game. Returns (winning_label, all_labels).

    `labels` is an optional list of identifiers parallel to `bots`. If None, uses bot.name.
    Shuffles seating so position doesn't bias results.
    """
    if labels is None:
        labels = [b.name for b in bots]
    indices = list(range(len(bots)))
    random.shuffle(indices)
    seated_labels = [labels[i] for i in indices]
    seated_bots = [bots[i] for i in indices]
    players = [PlayerState(name=f"P{i}_{seated_labels[i]}", bot=seated_bots[i]) for i in range(len(seated_bots))]
    g = Game(players, starting_dice=starting_dice, verbose=verbose)
    winner_seat = g.play()
    return seated_labels[winner_seat], seated_labels


def tournament(bot_factories: dict, games_per_match: int = 200, players_per_game: int = 4, starting_dice: int = 5):
    """Round-robin: every multiset of bots plays many games."""
    names = list(bot_factories.keys())
    win_counts = Counter()
    play_counts = Counter()
    combos = list(itertools.combinations_with_replacement(names, players_per_game))

    for combo in combos:
        for _ in range(games_per_match):
            bots = [bot_factories[n]() for n in combo]
            labels = list(combo)
            winner, _ = run_game(bots, labels=labels, starting_dice=starting_dice)
            win_counts[winner] += 1
            for n in combo:
                play_counts[n] += 1

    return win_counts, play_counts


def print_results(win_counts, play_counts):
    print(f"{'Bot':<25} {'Wins':>8} {'Played':>8} {'Win%':>8}")
    print("-" * 55)
    rows = []
    for name in play_counts:
        wins = win_counts.get(name, 0)
        played = play_counts[name]
        rate = wins / played if played else 0
        rows.append((rate, name, wins, played))
    rows.sort(reverse=True)
    for rate, name, wins, played in rows:
        print(f"{name:<25} {wins:>8} {played:>8} {rate*100:>7.2f}%")


def head_to_head(bot_a_factory, bot_b_factory, label_a="A", label_b="B", games=500, starting_dice=5):
    wins = Counter()
    for _ in range(games):
        bots = [bot_a_factory(), bot_b_factory()]
        labels = [label_a, label_b]
        winner, _ = run_game(bots, labels=labels, starting_dice=starting_dice)
        wins[winner] += 1
    return wins


if __name__ == "__main__":
    from bots import RandomBot, NaiveBot, ProbabilisticBot, AggressiveBot, ConservativeBot, SmartBot

    random.seed(42)

    factories = {
        "Random": lambda: RandomBot(),
        "Naive": lambda: NaiveBot(),
        "Prob": lambda: ProbabilisticBot(),
        "Aggressive": lambda: AggressiveBot(),
        "Conservative": lambda: ConservativeBot(),
        "Smart": lambda: SmartBot(),
    }

    print("=== All bots, 4-player games, 100 games per combo ===")
    wins, played = tournament(factories, games_per_match=100, players_per_game=4)
    print_results(wins, played)
