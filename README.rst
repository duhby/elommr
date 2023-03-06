elommr
======

.. image:: https://img.shields.io/pypi/dm/elommr?color=blueviolet&style=for-the-badge
   :target: https://pypi.python.org/pypi/elommr/
   :alt: PyPI downloads

.. image:: https://img.shields.io/pypi/v/elommr.svg?style=for-the-badge&logo=semantic-release&color=blue
   :target: https://pypi.python.org/pypi/elommr/
   :alt: PyPI version info

.. image:: https://img.shields.io/github/license/duhby/elommr?style=for-the-badge&color=bright-green
   :target: https://github.com/duhby/elommr/blob/master/LICENSE/
   :alt: License

A minimal, Python implementation of the Elo-MMR rating system as described in `this paper <https://arxiv.org/abs/2101.00400>`_.


Installation
^^^^^^^^^^^^

To install elommr, install it from pypi under the name ``elommr`` with
pip or your favorite package manager.

.. code:: sh

   pip install elommr --upgrade

Quick Example
^^^^^^^^^^^^^

You can view the docstrings for the
`EloMMR <https://github.com/duhby/elommr/blob/master/elommr/elommr.py#L21>`_ and
`Player <https://github.com/duhby/elommr/blob/master/elommr/elommr.py#L230>`_
classes for more information.

.. code:: python

    from elommr import EloMMR, Player
    from datetime import datetime, timezone

    def main():
        elo_mmr = EloMMR()
        player1 = Player()
        player2 = Player()
        standings = [
            (
                player1,
                0, 0 # Range of players that got or tied for first
            ),
            (
                player2,
                1, 1 # Range of players that got or tied for second
            ),
        ]

        # Note that the contest_time does not do anything in this example
        # because EloMMR.drift_per_sec defaults to 0, so contest_time
        # can be omitted from the round_update call, but it is included
        # here to show how it can be used.
        # Do note, though, that you should either always include
        # contest_time or never include it, because if you include it
        # in some rounds and not others, the ratings will be skewed
        # incorrectly.
        contest_time = round(datetime.now(timezone.utc).timestamp())
        elo_mmr.round_update(standings, contest_time)

        contest_time = round(datetime.now(timezone.utc).timestamp()) + 1000
        # Assumes the outcome of the next competition is the same as the
        # previous, so the standings aren't changed.
        elo_mmr.round_update(standings, contest_time)

        for player in [player1, player2]:
            print("\nrating_mu, rating_sig, perf_score, place")
            for event in player.event_history:
                print(f"{event.rating_mu}, {event.rating_sig}, {event.perf_score}, {event.place}")
            print(f"Final rating: {player.event_history[-1].display_rating()}")

        # >>>
        # rating_mu, rating_sig, perf_score, place
        # 1629, 171, 1654, 0
        # 1645, 130, 1663, 0
        # Final rating: 1645 ± 100
        #
        # rating_mu, rating_sig, perf_score, place
        # 1371, 171, 1346, 1
        # 1355, 130, 1337, 1
        # Final rating: 1355 ± 100

    if __name__ == '__main__':
        main()
