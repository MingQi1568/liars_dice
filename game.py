"""Liar's Dice game engine. 1s are wild (count as any face value)."""
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Bid:
    quantity: int
    face: int  # 1-6. Face 1 = "stars/ones" with halving rule.

    def __gt__(self, other: "Bid") -> bool:
        """House rule for 1s (asymmetric):
        - Both non-1: standard — higher qty, or same qty with higher face.
        - Both 1s:    higher qty.
        - non-1 -> 1: new_qty > other.qty  (i.e. >= old_qty + 1).
        - 1 -> non-1: new_qty >= other.qty (any face 2-6 at same-or-higher qty).
        """
        if other is None:
            return True
        a_is_one = self.face == 1
        b_is_one = other.face == 1
        if a_is_one and b_is_one:
            return self.quantity > other.quantity
        if a_is_one and not b_is_one:
            # non-1 -> 1: need strict qty bump
            return self.quantity > other.quantity
        if not a_is_one and b_is_one:
            # 1 -> non-1: same qty allowed, any face 2-6
            return self.quantity >= other.quantity
        # both non-1
        if self.quantity != other.quantity:
            return self.quantity > other.quantity
        return self.face > other.face

    def __ge__(self, other: "Bid") -> bool:
        return self == other or self > other

    def __repr__(self):
        return f"{self.quantity}x{self.face}"


@dataclass
class PlayerState:
    name: str
    dice: list[int] = field(default_factory=list)
    bot: "Bot" = None  # noqa: F821

    @property
    def num_dice(self):
        return len(self.dice)

    @property
    def alive(self):
        return self.num_dice > 0

    def roll(self):
        self.dice = [random.randint(1, 6) for _ in range(self.num_dice)]


@dataclass
class GameState:
    """View of the game from a single player's perspective."""
    my_dice: list[int]
    my_index: int
    num_players: int
    dice_counts: list[int]  # number of dice each player has
    total_dice: int
    current_bid: Optional[Bid]
    bid_history: list[tuple[int, Bid]]  # (player_index, bid)
    last_bidder: Optional[int]


class Game:
    def __init__(self, players: list[PlayerState], starting_dice: int = 5, verbose: bool = False):
        self.players = players
        for p in self.players:
            p.dice = [0] * starting_dice
        self.starting_dice = starting_dice
        self.verbose = verbose
        self.current_player = 0
        self.current_bid: Optional[Bid] = None
        self.bid_history: list[tuple[int, Bid]] = []
        self.last_bidder: Optional[int] = None

    def total_dice(self):
        return sum(p.num_dice for p in self.players)

    def alive_players(self):
        return [i for i, p in enumerate(self.players) if p.alive]

    def count_face(self, face: int) -> int:
        """Count dice showing `face`, including 1s as wilds (unless face is 1)."""
        total = 0
        for p in self.players:
            for d in p.dice:
                if d == face or (d == 1 and face != 1):
                    total += 1
        return total

    def next_alive_player(self, from_index: int) -> int:
        i = (from_index + 1) % len(self.players)
        while not self.players[i].alive:
            i = (i + 1) % len(self.players)
        return i

    def make_state_view(self, player_index: int) -> GameState:
        return GameState(
            my_dice=list(self.players[player_index].dice),
            my_index=player_index,
            num_players=len(self.players),
            dice_counts=[p.num_dice for p in self.players],
            total_dice=self.total_dice(),
            current_bid=self.current_bid,
            bid_history=list(self.bid_history),
            last_bidder=self.last_bidder,
        )

    def play_round(self):
        """Play a single round (until someone calls liar). Returns loser index."""
        for p in self.players:
            if p.alive:
                p.roll()
        self.current_bid = None
        self.bid_history = []
        self.last_bidder = None

        # Start with the player who lost last round if set, else current_player
        while True:
            player = self.players[self.current_player]
            if not player.alive:
                self.current_player = self.next_alive_player(self.current_player)
                continue

            state = self.make_state_view(self.current_player)
            action = player.bot.act(state)

            if action == "liar":
                if self.current_bid is None:
                    # cannot call liar with no bid - force a bid
                    raise ValueError(f"{player.name} called liar with no bid")
                actual = self.count_face(self.current_bid.face)
                bid_valid = actual >= self.current_bid.quantity
                if self.verbose:
                    all_dice = [d for p in self.players if p.alive for d in p.dice]
                    print(f"  {player.name} calls LIAR on {self.current_bid}. "
                          f"Actual {self.current_bid.face}s (incl wilds): {actual}. "
                          f"Dice: {sorted(all_dice)}")
                if bid_valid:
                    challenge_loser = self.current_player  # caller was wrong
                    challenge_winner = self.last_bidder    # bidder was right
                else:
                    challenge_loser = self.last_bidder
                    challenge_winner = self.current_player
                # Variant: WINNER of challenge loses a die (progress toward goal of 0)
                self.players[challenge_winner].dice.pop()
                if self.verbose:
                    print(f"  {self.players[challenge_winner].name} wins challenge, loses a die. "
                          f"Now has {self.players[challenge_winner].num_dice}.")
                # Variant: LOSER of challenge opens the next round (penalty)
                self.current_player = challenge_loser
                return challenge_winner
            else:
                new_bid: Bid = action
                if self.current_bid is not None and not (new_bid > self.current_bid):
                    raise ValueError(f"{player.name} made invalid bid {new_bid} after {self.current_bid}")
                if new_bid.face < 1 or new_bid.face > 6:
                    raise ValueError(f"{player.name} bid invalid face {new_bid.face} (must be 1-6)")
                if new_bid.quantity < 1:
                    raise ValueError(f"{player.name} bid invalid quantity {new_bid.quantity}")
                self.current_bid = new_bid
                self.bid_history.append((self.current_player, new_bid))
                self.last_bidder = self.current_player
                if self.verbose:
                    print(f"  {player.name} bids {new_bid}")
                self.current_player = self.next_alive_player(self.current_player)

    def play(self) -> int:
        """Play until a player reaches 0 dice. Returns winner index (the one at 0)."""
        round_num = 0
        while True:
            # Variant: first player to 0 dice WINS
            for i, p in enumerate(self.players):
                if p.num_dice == 0:
                    return i
            round_num += 1
            if self.verbose:
                alive = [(p.name, p.num_dice) for p in self.players if p.alive]
                print(f"\n-- Round {round_num} -- {alive}")
            self.play_round()
            if round_num > 10000:
                raise RuntimeError("Game ran too long")


class Bot:
    """Base class. Subclasses override act()."""
    name = "Bot"

    def act(self, state: GameState):
        """Return either a Bid or the string 'liar'."""
        raise NotImplementedError
