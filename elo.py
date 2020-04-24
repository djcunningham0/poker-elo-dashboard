from config import config
import errors as e
import numpy as np
import pandas as pd
from plotnine import *
import plotly.express as px


class MultiELO:
    """
    Generalized ELO for multiplayer matchups (also simplifies to standard ELO for 1-vs-1 matchups).
    Does not allow ties.
    """
    def __init__(self, k_value=config['DEFAULT_K_VALUE'], d_value=config['DEFAULT_D_VALUE'],
                 score_function=config['DEFAULT_SCORING_FUNCTION']):
        self.k = k_value
        self.d = d_value
        self._score_func = score_function

    def get_new_ratings(self, initial_ratings):
        n = len(initial_ratings)
        actual_scores = self.get_actual_scores(n)
        expected_scores = self.get_expected_scores(initial_ratings)
        return initial_ratings + self.k * (actual_scores - expected_scores)

    def get_actual_scores(self, n):
        scores = self._score_func(n)
        if not np.allclose(1, sum(scores)):
            raise e.ScoreError('scoring function does not return scores summing to 1')
        if min(scores) != 0:
            raise e.ScoreError('scoring function does not return minimum value of 0')
        if not np.all(np.diff(scores) < 0):
            raise e.ScoreError('scoring function does not return monotonically decreasing values')
        return scores

    def get_expected_scores(self, ratings):
        if not isinstance(ratings, np.ndarray):
            raise TypeError('ratings should be a numpy array')

        if ratings.ndim > 1:
            raise ValueError(f'ratings should be 1-dimensional array (received {ratings.ndim})')

        # get all pairwise differences
        diff_mx = ratings - ratings[:, np.newaxis]

        # get individual contributions to expected score using logistic function
        logistic_mx = 1 / (1 + 10 ** (diff_mx / self.d))
        np.fill_diagonal(logistic_mx, 0)

        # get each expected score (sum individual contributions, then scale)
        expected_scores = logistic_mx.sum(axis=1)
        n = len(ratings)
        denom = n * (n - 1) / 2  # number of individual head-to-head matchups between n players
        expected_scores = expected_scores / denom

        # this should be guaranteed, but check to make sure
        if not np.allclose(1, sum(expected_scores)):
            raise e.ScoreError('expected scores do not sum to 1')
        return expected_scores


class Player:
    def __init__(self, player_id, rating=config['INITIAL_RATING'], rating_history=None, date=None):
        self.id = player_id
        self.rating = rating
        if rating_history is None:
            self.rating_history = []
            self._update_rating_history(rating, date)
        else:
            self.rating_history = rating_history

    def update_rating(self, new_rating, date=None):
        self.rating = new_rating
        self._update_rating_history(new_rating, date)

    def _update_rating_history(self, rating, date):
        self.rating_history.append((date, rating))

    def __str__(self):
        return f'{self.id}: {round(self.rating, 2)} ({len(self.rating_history)-1} games)'

    def __repr__(self):
        return f'Player(id = {self.id}, rating = {round(self.rating, 2)}, n_games = {len(self.rating_history)-1})'

    def __lt__(self, other):
        return self.rating < other

    def __le__(self, other):
        return self.rating <= other

    def __gt__(self, other):
        return self.rating > other

    def __ge__(self, other):
        return self.rating >= other


class Tracker:
    def __init__(self, player_df=None, elo_rater=MultiELO()):
        if player_df is None:
            player_df = pd.DataFrame(columns=['player_id', 'player'], dtype=object)

        self.player_df = player_df
        self._validate_player_df()
        self.elo = elo_rater

    def process_data(self, df, date_col='date', verbose=False):
        df = df.sort_values(date_col).reset_index(drop=True)
        place_cols = [x for x in df.columns if x != date_col]
        for _, row in df.iterrows():
            date = row[date_col]
            players = [self._get_or_create_player(row[x]) for x in place_cols if not pd.isna(row[x])]
            initial_ratings = np.array([player.rating for player in players])
            new_ratings = self.elo.get_new_ratings(initial_ratings)

            for i, player in enumerate(players):
                player.update_rating(new_ratings[i], date=date)

            # optionally print the details of each iteration
            if verbose:
                out = f'{date}: '
                for i, player in enumerate(players):
                    out += f'{player.id}: {round(initial_ratings[i], 2)} --> {round(player.rating, 2)}; '

                print(out)

    def get_current_ratings(self):
        out = self.player_df.sort_values('player', ascending=False).reset_index(drop=True)
        out['rank'] = range(1, out.shape[0] + 1)
        return out

    def get_history_df(self):
        history_df = pd.DataFrame(columns=['player_id', 'date', 'rating'])
        players = [player for player in self.player_df['player']]
        for player in players:
            # check if there are any missing dates after the first entry (the initial rating)
            if any([x[0] is None for x in player.rating_history[1:]]):
                print(f'WARNING: possible missing dates in history for Player {player.id}')

            player_history_df = pd.DataFrame(player.rating_history, columns=['date', 'rating'])
            player_history_df = player_history_df[~player_history_df['date'].isna()]
            player_history_df['player_id'] = player.id
            history_df = pd.concat([history_df, player_history_df], sort=False)

        return history_df.reset_index(drop=True)

    def plot_history(self, line=True, point=False, interactive=False):
        if interactive:
            return self._plot_history_interactive()

        history_df = self.get_history_df()
        p = (
            ggplot(history_df, aes(x='date', y='rating', color='player_id'))
            + labs(title='ELO history', y='ELO rating')
        )
        if line:
            p += geom_line()
        if point:
            p += geom_point()

        # rotate labels if they are dates because it looks better
        if pd.api.types.is_datetime64_any_dtype(history_df['date']):
            p += theme(axis_text_x=element_text(angle=90, hjust=1))

        return p

    def _plot_history_interactive(self):
        history_df = self.get_history_df()
        fig = px.line(history_df, x='date', y='rating', color='player_id')
        fig.update_layout(
            title='ELO history',
            yaxis_title='ELO rating'
        )
        fig.show()

    def retrieve_existing_player(self, player_id):
        if player_id in self.player_df['player_id'].tolist():
            return self.player_df.loc[self.player_df['player_id'] == player_id, 'player'].tolist()[0]
        else:
            raise e.PlayerRetrievalError(f'no player found with ID {player_id}')

    def _get_or_create_player(self, player_id):
        if player_id in self.player_df['player_id'].tolist():
            return self.retrieve_existing_player(player_id)
        else:
            return self._create_new_player(player_id)

    def _create_new_player(self, player_id, rating=config['INITIAL_RATING'], rating_history=None, date=None):
        # first check if the player already exists
        if player_id in self.player_df['player_id'].tolist():
            raise e.PlayerCreationError(f'a player with ID {player_id} already exists in the tracker')

        # create and add the player to the database
        player = Player(player_id, rating, rating_history=rating_history, date=date)
        add_df = pd.DataFrame({'player_id': [player_id], 'player': [player]})
        self.player_df = pd.concat([self.player_df, add_df])
        self._validate_player_df()
        return player

    def _validate_player_df(self):
        if not self.player_df['player_id'].is_unique:
            raise ValueError('Player IDs must be unique')

        if not all([isinstance(x, Player) for x in self.player_df['player']]):
            raise ValueError('The player column should contain Player objects')

        self.player_df = self.player_df.sort_values('player_id').reset_index(drop=True)

    def __repr__(self):
        return f'Tracker({self.player_df.shape[0]} total players)'
