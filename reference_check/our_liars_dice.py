"""Our 1-die Liar's Dice as an OpenSpiel (pyspiel) game, so OpenSpiel algorithms - including DeepMind's
reference R-NaD - can run on exactly our rules.

Rules (match game.py / vec_env.py with one die each): each player rolls one die; player 0 opens; a bid
(q, f) must be strictly higher in (quantity, face) order with face 1 the LOWEST face; 1s are wild for bids
on faces 2-6, a bid on 1s counts only 1s; "liar" is legal once a bid stands. A challenge ends the game:
if at least q dice match, the bidder wins (+1), otherwise the challenger wins.

Actions: bid (q, f) -> (q - 1) * 6 + (f - 1) for q in 1..2 (0..11); liar -> 12 (same numbering as vec_env).
Information-state tensor mirrors OpenSpiel's liars_dice: [player one-hot (2), own die one-hot (6),
bids made (12) + liar (1)] = 21.
"""
import numpy as np
import pyspiel

_N_BIDS, _LIAR = 12, 12
_GAME_TYPE = pyspiel.GameType(
    short_name="our_liars_dice_d1",
    long_name="Liar's Dice, 1 die each, 1s wild (our rules)",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True)
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_N_BIDS + 1, max_chance_outcomes=6, num_players=2,
    min_utility=-1.0, max_utility=1.0, utility_sum=0.0, max_game_length=_N_BIDS + 1)


def bid_of(action):
    return action // 6 + 1, action % 6 + 1


class OurLiarsDiceGame(pyspiel.Game):
    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    def new_initial_state(self):
        return OurLiarsDiceState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return OurLiarsDiceObserver(iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False), params)


class OurLiarsDiceState(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        self.dice = []          # faces 1..6, player 0 then player 1
        self.bids = []          # action indices 0..11, strictly increasing
        self.winner = None

    def current_player(self):
        if self.winner is not None:
            return pyspiel.PlayerId.TERMINAL
        if len(self.dice) < 2:
            return pyspiel.PlayerId.CHANCE
        return len(self.bids) % 2

    def _legal_actions(self, player):
        if not self.bids:
            return list(range(_N_BIDS))
        return list(range(self.bids[-1] + 1, _N_BIDS)) + [_LIAR]

    def chance_outcomes(self):
        return [(o, 1.0 / 6) for o in range(6)]

    def _apply_action(self, action):
        if self.is_chance_node():
            self.dice.append(action + 1)
            return
        if action != _LIAR:
            self.bids.append(action)
            return
        q, f = bid_of(self.bids[-1])
        matches = sum(1 for d in self.dice if d == f or (d == 1 and f != 1))
        caller = len(self.bids) % 2
        bidder = 1 - caller
        self.winner = bidder if matches >= q else caller

    def _action_to_string(self, player, action):
        if player == pyspiel.PlayerId.CHANCE:
            return f"Roll:{action + 1}"
        if action == _LIAR:
            return "Liar"
        q, f = bid_of(action)
        return f"{q}-{f}"

    def is_terminal(self):
        return self.winner is not None

    def returns(self):
        if self.winner is None:
            return [0.0, 0.0]
        return [1.0, -1.0] if self.winner == 0 else [-1.0, 1.0]

    def __str__(self):
        return f"dice={self.dice} bids={[self._action_to_string(0, b) for b in self.bids]}"


class OurLiarsDiceObserver:
    """Same layout as OpenSpiel liars_dice: player, own die, bids made (+ liar slot)."""

    def __init__(self, iig_obs_type, params):
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")
        pieces = [("player", 2, (2,))]
        if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
            pieces.append(("die", 6, (6,)))
        if iig_obs_type.public_info:
            pieces.append(("bids", _N_BIDS + 1, (_N_BIDS + 1,)))
        self.tensor = np.zeros(sum(size for _, size, _ in pieces), np.float32)
        self.dict, i = {}, 0
        for name, size, shape in pieces:
            self.dict[name] = self.tensor[i:i + size].reshape(shape)
            i += size

    def set_from(self, state, player):
        self.tensor.fill(0)
        self.dict["player"][player] = 1
        if "die" in self.dict and len(state.dice) > player:
            self.dict["die"][state.dice[player] - 1] = 1
        if "bids" in self.dict:
            for b in state.bids:
                self.dict["bids"][b] = 1
            if state.winner is not None:
                self.dict["bids"][_LIAR] = 1

    def string_from(self, state, player):
        parts = [f"p{player}"]
        if "die" in self.dict and len(state.dice) > player:
            parts.append(f"d{state.dice[player]}")
        if "bids" in self.dict:
            parts.append("b" + ",".join(str(b) for b in state.bids))
        return " ".join(parts)


pyspiel.register_game(_GAME_TYPE, OurLiarsDiceGame)
