"""
Copyright (c) 2023-present duhby
MIT License, see LICENSE for more details.

Copyright (c) 2021 Elo-MMR Project
MIT License, see LICENSE for more details.
"""

from dataclasses import dataclass, field
from functools import reduce
from itertools import chain
import math


TANH_MULTIPLIER = math.pi / math.sqrt(3)
DEFAULT_MU = 1500
DEFAULT_SIG = 350
DEFAULT_SIG_LIMIT = 80

@dataclass
class EloMMR:
    """Base Elo-MMR class. Used to interface with the Elo-MMR algorithm.

    .. note::

        All parameters are optional.

    Parameters
    ----------
    split_ties: bool
        Whtether to count ties as half a win plus half a loss. Defaults
        to False.
    drift_per_sec: float
        Uncertainty added passively between contests due to off-site
        practice or oxidation.
    weight_limit: float
        The maximum weight of a match. Values should be between 0 and 1.
    noob_delay: List[float]
        Amount to delay convergence for new players for beginning
        matches. This number gets multiplied by the weight which
        decreases the speed sigma decreases. Values should be between
        0 and 1.
    sig_limit: float
        The value sigma will converge to.
    transfer_speed: float
        How quickly to move from a logistic to gaussian belief, where
        lower values provide more robustness against one-time
        deviations, while higher values allow for less memory retention.
        It is important for the value to be positive and finite for
        theoretical reasons, but the default value of 1 is typically
        sufficient.
    max_history: int
        The maximum history to calculate per player. If None (default),
        there is no limit.

        .. note::

            This removes the oldest logistical factors from the
            calculation. If you're worried about memory usage, you can
            set this to a low value, but it will make the algorithm
            less accurate.
    """
    split_ties: bool = False
    drift_per_sec: float = 0
    weight_limit: float = .2
    noob_delay: list = field(default_factory=list)
    sig_limit: float = DEFAULT_SIG_LIMIT
    transfer_speed: float = 1
    max_history: int = None

    def __post_init__(self):
        self.mul = 1 if self.split_ties else 2

    def round_update(
        self,
        standings: list,
        contest_time: int = 0,
        weight: float = 1,
        perf_ceiling: float = None
    ):
        """Update the ratings of players in a round.

        Parameters
        ----------
        standings: List[Tuple[Player, int, int]]
            A list of tuples containing the player, and a placement
            range, with first place being 0. For example, if there 
            are 4 players, and the first player places 1st, the second
            player places 2nd, and the third and fourth players tie for
            3rd, the standings would look like this: ``[(player1, 0, 0),
            (player2, 1, 1), (player3, 2, 3), (player4, 2, 3)]``.

            .. warning::

                The order of the list matters. The placements must be
                in order from first to last.
        contest_time: int
            The time of the contest in seconds since the epoch.
        weight: float
            The weight of the match. I recommend keeping this at 1
            (default).

            .. note::

                This is supposed to work with a weight of 0, but it
                doesn't for some reason.
        perf_ceiling: float
            The maximum performance score a player can have. Defaults
            to None, which means there is no limit.
        """
        for player, lo, _ in standings:
            if player.update_time is None:
                player.update_time = contest_time
                player.delta_time = 0 # contest_time - player.update_time
            else:
                player.delta_time = contest_time - player.update_time
                player.update_time = contest_time
            player.event_history.append(
                PlayerEvent(
                    rating_mu=0, # Filled later
                    rating_sig=0, # Filled later
                    perf_score=0, # Filled later
                    place=lo,
                )
            )

        tanh_terms = []
        for player, _, _ in standings:
            sig_perf, discrete_drift = self.sig_perf_and_drift(
                weight, len(player.event_history) - 1
            )
            continuous_drift = self.drift_per_sec * player.delta_time
            sig_drift = math.sqrt(discrete_drift + continuous_drift)
            player.add_noise_best(sig_drift, self.transfer_speed)
            with_noise = player.approx_posterior.with_noise(sig_perf)
            tanh_term = TanhTerm.from_rating(with_noise.mu, with_noise.sig)
            tanh_terms.append(tanh_term)

        for player, lo, hi in standings:
            bounds = (-6000.0, 9000.0)
            f = lambda x: compute_likelihood_sum(x, tanh_terms, lo, hi, self.mul)
            solved = solve_newton(bounds, f)
            if perf_ceiling is not None:
                mu_perf = min(solved, perf_ceiling)
            else:
                mu_perf = solved
            sig_perf, _ = self.sig_perf_and_drift(weight, len(player.event_history) - 1)
            player.update_rating_with_logistic(
                Rating(mu_perf, sig_perf), self.max_history
            )

    def sig_perf_and_drift(self, weight: int, n: int) -> (float, float):
        weight *= self.weight_limit
        if n < len(self.noob_delay):
            weight *= self.noob_delay[n]
        sig_perf = self.sig_limit * math.sqrt(1 + 1 / weight)
        sig_drift_sq = weight * self.sig_limit ** 2
        return (sig_perf, sig_drift_sq)

@dataclass
class TanhTerm:
    """Represents... something internal."""
    mu: float
    w_arg: float
    w_out: float

    def get_weight(self) -> float:
        return self.w_out * self.w_arg * 2 / (TANH_MULTIPLIER ** 2)

    @staticmethod
    def from_rating(mu: float, sig: float) -> 'TanhTerm':
        w = TANH_MULTIPLIER / sig
        return TanhTerm(
            mu=mu,
            w_arg=w * 0.5,
            w_out=w,
        )

    def base_values(self, x: float) -> (float, float):
        z = (x - self.mu) * self.w_arg
        val = -math.tanh(z) * self.w_out
        val_prime = -math.cosh(z) ** -2 * self.w_arg * self.w_out
        return (val, val_prime)

@dataclass
class Rating:
    """Represents a player's rating.

    Attributes
    ----------
    mu: float
        The mean of the rating.
    sig: float
        The uncertainty level or standard deviation of the rating mu.
    """
    mu: float
    sig: float

    def with_noise(self, sig_noise: float) -> 'Rating':
        new_sig = math.sqrt(self.sig ** 2 + sig_noise ** 2)
        return Rating(
            mu=self.mu,
            sig=new_sig,
        )

@dataclass
class PlayerEvent:
    rating_mu: float
    rating_sig: float
    perf_score: float
    place: int

    def display_rating(
        self, stdevs: float = 2, sig_limit: float = DEFAULT_SIG_LIMIT
    ) -> float:
        """A string representation of the rating.

        Displays the mean (self.rating_mu) of the rating, plus or minus
        the number of standard deviations specified by ``stdevs`` and
        the uncertainty level (self.rating_sig), limited by
        ``sig_limit``.

        Parameters
        ----------
        stdevs: float
            The number of standard deviations to display. Defaults to
            2.
        sig_limit: float
            The minimum uncertainty level. Should be the same as the
            ``sig_limit`` parameter of the :class:`EloMMR` class.
            Defaults to 80.
        """
        return f"{self.rating_mu} ?? {stdevs * (self.rating_sig - sig_limit)}"

@dataclass
class Player:
    """Represents a player.

    .. warning::

        All of these attributes are modified in the
        :meth:`EloMMR.round_update` method. Do not modify them directly
        unless you know what you are doing.

    .. tip::

        If you want to change the initial rating of a player, you can
        provide custom ``Rating`` objects to the ``_normal_factor`` and
        ``approx_posterior`` attributes, making sure the classes are
        separate instances.

    Parameters
    ----------
    _normal_factor: Rating
        The normal factor of the player.
    _logistic_factors: list
        The logistic factors of the player.
    event_history: list
        The history of events for the player.
    approx_posterior: Rating
        The approximate posterior of the player.
    update_time: int
        The time of the last competition the player
        participated in. Represented as seconds since the epoch.
    delta_time: int
        The time since the last competition the player participated in.
        Represented as seconds.
    """
    _normal_factor: Rating = field(
        default_factory=lambda: Rating(mu=DEFAULT_MU, sig=DEFAULT_SIG)
    )
    _logistic_factors: list = field(default_factory=list)
    event_history: list = field(default_factory=list)
    approx_posterior: Rating = field(
        default_factory=lambda: Rating(mu=DEFAULT_MU, sig=DEFAULT_SIG)
    )
    update_time: int = None
    delta_time: int = None

    def add_noise_best(self, sig_noise: float, transfer_speed: float):
        new_posterior = self.approx_posterior.with_noise(sig_noise)

        decay = (self.approx_posterior.sig / new_posterior.sig) ** 2
        transfer = decay ** transfer_speed
        self.approx_posterior = new_posterior

        wt_norm_old = self._normal_factor.sig ** -2
        wt_from_norm_old = transfer * wt_norm_old
        wt_from_transfers = (1 - transfer) * (
            wt_norm_old + sum(r.get_weight() for r in self._logistic_factors)
        )
        wt_total = wt_from_norm_old + wt_from_transfers

        self._normal_factor.mu = (
            wt_from_norm_old * self._normal_factor.mu +
            wt_from_transfers * self.approx_posterior.mu
        ) / wt_total
        self._normal_factor.sig = (decay * wt_total) ** -0.5
        for r in self._logistic_factors:
            r.w_out *= transfer * decay

    def update_rating(self, rating, performance_score):
        # Assumes that a placeholder history item has been pushed
        last_event = self.event_history[-1]
        assert last_event.rating_mu == 0
        assert last_event.rating_sig == 0
        assert last_event.perf_score == 0

        self.approx_posterior = rating
        last_event.rating_mu = round(rating.mu)
        last_event.rating_sig = round(rating.sig)
        last_event.perf_score = round(performance_score)

    def update_rating_with_logistic(self, performance: Rating, max_history: int):
        if max_history is not None:
            if len(self._logistic_factors) >= max_history:
                logistic = self._logistic_factors.pop(0)
                wn = self._normal_factor.sig ** -2
                wl = logistic.get_weight()
                self._normal_factor.mu = (
                    wn * self._normal_factor.mu + wl * logistic.mu
                ) / (wn + wl)
                self._normal_factor.sig = (wn + wl) ** -0.5
        self._logistic_factors.append(
            TanhTerm.from_rating(performance.mu, performance.sig)
        )

        new_rating = self.approximate_posterior(performance.sig)
        self.update_rating(new_rating, performance.mu)

    def approximate_posterior(self, perf_sig: float) -> Rating:
        normal_weight = self._normal_factor.sig ** -2
        mu = robust_average(
            self._logistic_factors.copy(),
            -self._normal_factor.mu * normal_weight,
            normal_weight,
        )
        sig = (self.approx_posterior.sig ** -2 + perf_sig ** -2) ** -0.5
        return Rating(mu=mu, sig=sig)

    def __repr__(self):
        last_event = self.event_history[-1]
        return f"Player(mu={last_event.rating_mu}, sig={last_event.rating_sig})"

# Returns the unique zero of the following, strictly increasing function of x:
# offset + slope * x + sum_i weight_i * tanh((x-mu_i)/sig_i)
# We must have slope != 0 or |offset| < sum_i weight_i in order for the zero to exist.
# If offset == slope == 0, we get a robust weighted average of the mu_i's.
def robust_average(all_ratings: list, offset: float, slope: float) -> float:
    bounds = (-6000.0, 9000.0)

    def weighted_tanh_deriv_sum(x: float) -> tuple:
        s = sp = 0.0
        for term in all_ratings:
            tanh_z = math.tanh((x - term.mu) * term.w_arg)
            s += tanh_z * term.w_out
            sp += (1. - tanh_z * tanh_z) * term.w_arg * term.w_out
        return (s + offset + slope * x, sp + slope)

    return solve_newton(bounds, weighted_tanh_deriv_sum)

def eval_less(term: TanhTerm, x: float) -> tuple:
    val, val_prime = term.base_values(x)
    return (val - term.w_out, val_prime)

def eval_grea(term: TanhTerm, x: float) -> tuple:
    val, val_prime = term.base_values(x)
    return (val + term.w_out, val_prime)

def eval_equal(term: TanhTerm, x: float, mul: float) -> tuple:
    val, val_prime = term.base_values(x)
    return (mul * val, mul * val_prime)

def compute_likelihood_sum(x, tanh_terms, lo, hi, mul):
    itr1 = (eval_less(term, x) for term in tanh_terms[:lo])
    itr2 = (eval_equal(term, x, mul) for term in tanh_terms[lo:hi+1])
    itr3 = (eval_grea(term, x) for term in tanh_terms[hi+1:])
    return reduce(
        lambda acc, v: (acc[0] + v[0], acc[1] + v[1]), chain(itr1, itr2, itr3), (0, 0)
    )

def solve_newton(bounds: tuple, f: callable):
    lo, hi = bounds
    guess = 0.5 * (lo + hi)
    while True:
        sum_, sum_prime = f(guess)
        extrapolate = guess - sum_ / sum_prime
        if extrapolate < guess:
            hi = guess
            guess = max(extrapolate, hi - 0.75 * (hi - lo))
        else:
            lo = guess
            guess = min(extrapolate, lo + 0.75 * (hi - lo))
        if lo >= guess or guess >= hi:
            if abs(sum_) > 1e-10:
                print(f"Possible failure to converge @ {guess}: s={sum_}, s'={sum_prime}")
            return guess
