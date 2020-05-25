from config import config
import errors as e
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
import plotly.express as px
import score_functions as score


class MultiELO:
    """
    Generalized ELO for multiplayer matchups (also simplifies to standard ELO for 1-vs-1 matchups).
    Does not allow ties.
    """

    def __init__(
        self,
        k_value=config["elo"]["DEFAULT_K_VALUE"],
        d_value=config["elo"]["DEFAULT_D_VALUE"],
        score_function_base=config["elo"]["DEFAULT_SCORING_FUNCTION_BASE"],
        scale_k=config["elo"]["DEFAULT_SCALE_K"],
    ):
        """
        :param k_value: K parameter in ELO algorithm that determines how much ratings increase or decrease
        after each match
        :type k_value: float
        :param d_value: D parameter in ELO algorithm that determines how much ELO difference affects win
        probability
        :type d_value: float
        :param score_function_base: base value to use for scoring function; scores are approximately
        multiplied by this value as you improve from one place to the next (minimum allowed value is 1,
        which results in a linear scoring function)
        :type score_function_base: float
        :param scale_k: if True, use K * (number of players - 1) as the factor to determine change in
        ELO rating; otherwise use K. When scale_k == True, players
        :type scale_k: bool
        """
        self.k = k_value
        self.d = d_value
        self._score_func = score.create_exponential_score_function(
            base=score_function_base
        )
        self.scale_k = scale_k

    def get_new_ratings(self, initial_ratings):
        """
        Update ratings based on results. Takes an array of ratings before the matchup and returns an array with
        the updated ratings. Provided array should be ordered by the actual results (first place finisher's
        initial rating first, second place next, and so on).

        :param initial_ratings: array of ratings (float values) in order of actual results
        :type initial_ratings: ndarray

        :return: array of updated ratings (float values) in same order as input
        :rtype: ndarray
        """
        n = len(initial_ratings)
        actual_scores = self.get_actual_scores(n)
        expected_scores = self.get_expected_scores(initial_ratings)
        scale_factor = self.k * (n - 1) if self.scale_k else self.k
        return initial_ratings + scale_factor * (actual_scores - expected_scores)

    def get_actual_scores(self, n):
        """
        Return the scores to be awarded to the players based on the results.

        :param n: number of players in the matchup
        :type n: int

        :return: array of length n of scores to be assigned to first place, second place, and so on
        :rtype: ndarray
        """
        scores = self._score_func(n)
        if not np.allclose(1, sum(scores)):
            raise e.ScoreError("scoring function does not return scores summing to 1")
        if min(scores) != 0:
            raise e.ScoreError("scoring function does not return minimum value of 0")
        if not np.all(np.diff(scores) < 0):
            raise e.ScoreError("scoring function does not return monotonically decreasing values")
        return scores

    def get_expected_scores(self, ratings):
        """
        Get the expected scores for all players given their ratings before the matchup.

        :param ratings: array of ratings for each player in a matchup
        :type ratings: ndarray

        :return: array of expected scores for all players
        :rtype: ndarray
        """
        if ratings.ndim > 1:
            raise ValueError(f"ratings should be 1-dimensional array (received {ratings.ndim})")

        # get all pairwise differences
        diff_mx = ratings - ratings[:, np.newaxis]

        # get individual contributions to expected score using logistic function
        logistic_mx = 1 / (1 + 10 ** (diff_mx / self.d))
        np.fill_diagonal(logistic_mx, 0)

        # get each expected score (sum individual contributions, then scale)
        expected_scores = logistic_mx.sum(axis=1)
        n = len(ratings)
        denom = (
            n * (n - 1) / 2
        )  # number of individual head-to-head matchups between n players
        expected_scores = expected_scores / denom

        # this should be guaranteed, but check to make sure
        if not np.allclose(1, sum(expected_scores)):
            raise e.ScoreError("expected scores do not sum to 1")
        return expected_scores


class Player:
    def __init__(
        self,
        player_id,
        rating=config["elo"]["INITIAL_RATING"],
        rating_history=None,
        date=None,
    ):
        """
        Instantiate a player.

        :param player_id: player ID (e.g., the player's name)
        :type player_id: str
        :param rating: player's current rating
        :type rating: float
        :param rating_history: history of player's ratings (each entry is a (date, rating) tuple); if None, create
        the first entry in the player's history
        :type rating_history: list
        :param date: date of this rating (e.g., player's first matchup date)
        :type date: str
        """
        self.id = player_id
        self.rating = rating
        if rating_history is None:
            self.rating_history = []
            self._update_rating_history(rating, date)
        else:
            self.rating_history = rating_history

    def update_rating(self, new_rating, date=None):
        """
        Update a player's rating and rating history. (updates the self.rating and self.rating_history attributes)

        :param new_rating: player's new rating
        :type new_rating: float
        :param date: date the new rating was achieved
        :type date: str
        """
        self.rating = new_rating
        self._update_rating_history(new_rating, date)

    def count_games(self):
        return len(self.rating_history) - 1

    def _update_rating_history(self, rating, date):
        self.rating_history.append((date, rating))

    def __str__(self):
        return f"{self.id}: {round(self.rating, 2)} ({self.count_games()} games)"

    def __repr__(self):
        return f"Player(id = {self.id}, rating = {round(self.rating, 2)}, n_games = {self.count_games()})"

    def __lt__(self, other):
        return self.rating < other

    def __le__(self, other):
        return self.rating <= other

    def __gt__(self, other):
        return self.rating > other

    def __ge__(self, other):
        return self.rating >= other


class Tracker:
    def __init__(
        self,
        elo_rater=MultiELO(),
        initial_rating=config["elo"]["INITIAL_RATING"],
        player_df=None,
    ):
        """
        Instantiate a tracker that will track player's ratings over time as matchups occur.

        :param elo_rater:
        :type elo_rater: MultiELO
        :param initial_rating: initial rating value for new players
        :type initial_rating: float
        :param player_df: dataframe of existing players. New players will be added to the dataframe when they
        appear in a matchup for the first time. If None, begin with no players in the dataframe.
        :type player_df: DataFrame
        """
        self.elo = elo_rater
        self._initial_player_rating = initial_rating

        if player_df is None:
            player_df = DataFrame(columns=["player_id", "player"], dtype=object)

        self.player_df = player_df
        self._validate_player_df()

    def process_data(self, matchup_history_df, date_col="date", verbose=False):
        """
        Process the full matchup history of a group of players. Update the ratings and rating history for all
        players in found in the matchup history.

        :param matchup_history_df: dataframe of matchup history with a column for date and one column for each
        possible finishing place (e.g., "date", "1st", "2nd", "3rd", ...). Finishing place columns should be in
        order of first to last.
        :type matchup_history_df: DataFrame
        :param date_col: name of the date column
        :type date_col: str
        :param verbose: if True, print the details of all rating changes after processing each matchup
        :type verbose: bool
        """
        matchup_history_df = matchup_history_df.sort_values(date_col).reset_index(
            drop=True
        )
        place_cols = [x for x in matchup_history_df.columns if x != date_col]
        matchup_history_df = matchup_history_df.dropna(how="all", axis=0, subset=place_cols)  # drop rows if all NaN
        for _, row in matchup_history_df.iterrows():
            date = row[date_col]
            players = [self._get_or_create_player(row[x]) for x in place_cols if not pd.isna(row[x])]
            initial_ratings = np.array([player.rating for player in players])
            new_ratings = self.elo.get_new_ratings(initial_ratings)

            for i, player in enumerate(players):
                player.update_rating(new_ratings[i], date=date)

            # optionally print the details of each iteration
            if verbose:
                out = f"{date}: "
                for i, player in enumerate(players):
                    out += f"{player.id}: {round(initial_ratings[i], 2)} --> {round(player.rating, 2)}; "

                print(out)

    def get_current_ratings(self):
        """
        Retrieve the current ratings of all players in this Tracker.

        :return: dataframe with all players' ratings and number of games played
        :rtype: DataFrame
        """
        df = self.player_df.copy()
        df["rating"] = df["player"].apply(lambda x: x.rating)
        df["n_games"] = df["player"].apply(lambda x: x.count_games())
        df = df.sort_values("player", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, df.shape[0] + 1)
        df = df[["rank", "player_id", "n_games", "rating"]]
        return df

    def get_history_df(self):
        """
        Retrieve the rating history for all players in this Tracker.

        :return: dataframe with all players' ratings on each date that they changed
        :rtype: DataFrame
        """
        history_df = DataFrame(columns=["player_id", "date", "rating"])
        players = [player for player in self.player_df["player"]]
        for player in players:
            # check if there are any missing dates after the first entry (the initial rating)
            if any([x[0] is None for x in player.rating_history[1:]]):
                print(f"WARNING: possible missing dates in history for Player {player.id}")

            player_history_df = DataFrame(player.rating_history, columns=["date", "rating"])
            player_history_df = player_history_df[~player_history_df["date"].isna()]
            player_history_df["player_id"] = player.id
            history_df = pd.concat([history_df, player_history_df], sort=False)

        return history_df.reset_index(drop=True)

    def plot_history(
        self,
        include_points=True,
        include_average=True,
        average_val=config["elo"]["INITIAL_RATING"],
    ):
        """
        Plot the rating history of each player in the Tracker.

        :param include_points: if True, include point markers (if False, only show lines)
        :type include_points: bool
        :param include_average: if True, include a dashed line at the starting rating for new players
        :type include_average: bool
        :param average_val: value to use for the average rating (displayed as a dashed line); by default, use
        the starting value for new players
        :type average_val: float

        :return: a plot generated using plotly.express.line
        """
        history_df = self.get_history_df()
        fig = px.line(history_df, x="date", y="rating", color="player_id")
        mode = "lines+markers" if include_points else "lines"
        fig.update_traces(mode=mode)
        fig.update_layout(
            yaxis_title="ELO rating",
            title="ELO history",
            title_x=0.5,
            legend=dict(title="<b>Player</b>", y=0.5),
        )
        if include_average:
            fig.update_layout(
                shapes=[
                    dict(
                        type="line",
                        yref="y",
                        y0=average_val,
                        y1=average_val,
                        xref="paper",
                        x0=0,
                        x1=1,
                        opacity=0.5,
                        line=dict(dash="dash", width=1.5),
                    )
                ]
            )
        return fig

    def retrieve_existing_player(self, player_id):
        """
        Retrieve a player in the Tracker with a given ID.

        :param player_id: the player's ID
        :type player_id: str

        :return: the Player object associated with the provided ID
        :rtype: Player
        """
        if player_id in self.player_df["player_id"].tolist():
            player = self.player_df.loc[
                self.player_df["player_id"] == player_id, "player"
            ].tolist()[0]
            return player
        else:
            raise e.PlayerRetrievalError(f"no player found with ID {player_id}")

    def _get_or_create_player(self, player_id):
        if player_id in self.player_df["player_id"].tolist():
            return self.retrieve_existing_player(player_id)
        else:
            return self._create_new_player(player_id)

    def _create_new_player(self, player_id):
        # first check if the player already exists
        if player_id in self.player_df["player_id"].tolist():
            raise e.PlayerCreationError(
                f"a player with ID {player_id} already exists in the tracker"
            )

        # create and add the player to the database
        player = Player(player_id, rating=self._initial_player_rating)
        add_df = DataFrame({"player_id": [player_id], "player": [player]})
        self.player_df = pd.concat([self.player_df, add_df])
        self._validate_player_df()
        return player

    def _validate_player_df(self):
        if not self.player_df["player_id"].is_unique:
            raise ValueError("Player IDs must be unique")

        if not all([isinstance(x, Player) for x in self.player_df["player"]]):
            raise ValueError("The player column should contain Player objects")

        self.player_df = self.player_df.sort_values("player_id").reset_index(drop=True)

    def __repr__(self):
        return f"Tracker({self.player_df.shape[0]} total players)"
