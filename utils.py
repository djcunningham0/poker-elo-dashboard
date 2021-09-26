from google.oauth2.service_account import Credentials
import gspread
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import plotly.express as px
from multielo import MultiElo, Tracker
from gspread.client import Client
from gspread.models import Spreadsheet, Worksheet
from plotly.graph_objs import Figure
from typing import List, Union

import config


def load_data_from_gsheet() -> pd.DataFrame:
    gc = set_up_gsheets_client(config.GSHEETS_CREDENTIALS_FILE)
    spreadsheet = gc.open_by_key(config.SPREADSHEET_ID)
    data_sheet = get_worksheet_by_id(spreadsheet, config.DATA_SHEET_ID)
    df = worksheet_to_dataframe(data_sheet)
    return df


def set_up_gsheets_client(credentials_file: str) -> Client:
    scopes = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]

    credentials = Credentials.from_service_account_file(
        filename=credentials_file, scopes=scopes
    )

    client = gspread.authorize(credentials)

    return client


def get_worksheet_by_id(spreadsheet: Spreadsheet, worksheet_id: str) -> Worksheet:
    try:
        return [w for w in spreadsheet.worksheets() if w.id == worksheet_id][0]
    except IndexError:
        raise gspread.WorksheetNotFound(f"worksheet ID {worksheet_id} does not exist")


def get_worksheet_by_name(spreadsheet: Spreadsheet, worksheet_name: str) -> Worksheet:
    return spreadsheet.worksheet(worksheet_name)


def worksheet_to_dataframe(worksheet: Worksheet, headers: bool = True) -> pd.DataFrame:
    data = worksheet.get_all_values()
    if headers:
        columns = data[0]
        data = data[1:]
    else:
        columns = [f"col{i}" for i in range(len(data[0]))]

    df = pd.DataFrame(data, columns=columns)
    df = replace_null_string_with_nan(df)
    return df


def replace_null_string_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace("", np.nan)


def get_dash_theme(style: str) -> List[str]:
    try:
        return [getattr(dbc.themes, style)]
    except AttributeError:
        raise AttributeError(f"could not find theme named '{style}'")


def prep_results_history_for_dash(
        data: pd.DataFrame,
) -> pd.DataFrame:
    results_history = data.copy()
    results_history = results_history.dropna(how="all", axis=1)  # drop columns if all NaN
    results_history = results_history.rename(columns={"date": "Date"})
    return results_history


def prep_current_ratings_for_dash(
        tracker: Tracker,
        results_history: pd.DataFrame,
        min_games: int = 0,
) -> pd.DataFrame:
    current_ratings = tracker.get_current_ratings()
    current_ratings["rating"] = current_ratings["rating"].round(2)
    win_df = get_wins_from_history(results_history)
    current_ratings = (
        remove_dummy_player(df=current_ratings)
        .merge(win_df, on="player_id", how="left")
        .fillna({"n_wins": 0})
        .rename(columns={
            "rank": "Rank",
            "player_id": "Name",
            "n_games": "Games Played",
            "n_wins": "Wins",
            "rating": "Elo Rating",
        })
    )

    # only include players who have played min_games, then re-rank
    current_ratings = current_ratings[current_ratings["Games Played"] >= min_games]
    current_ratings["Rank"] = range(1, current_ratings.shape[0] + 1)

    col_order = ["Rank", "Name", "Games Played", "Wins", "Elo Rating"]
    return current_ratings[col_order]


def get_wins_from_history(results_history: pd.DataFrame) -> pd.DataFrame():
    return (
        pd.DataFrame(results_history["1st"].value_counts())
        .reset_index()
        .rename(columns={
            "index": "player_id",
            "1st": "n_wins",
        })
    )


def plot_tracker_history(
        tracker: Tracker,
        title: str = None,
        equal_time_steps: bool = False,
        min_games: int = 0,
) -> Figure:
    """
    Create an interactive plot with the rating history of each player in the Tracker.

    :param tracker: tracker with Elo history for all players
    :param title: title for the plot
    :param equal_time_steps: if True, space the x-axis equally; otherwise use the
    provided timestamps
    :param min_games: minimum number of games player must have played to be included

    :return: a plot generated using plotly.express.line
    """
    history_df = tracker.get_history_df()
    history_df = remove_dummy_player(df=history_df)

    # filter out players who haven't played min_games
    include_players = [player.id for player in tracker.player_df["player"]
                       if player.count_games() >= min_games]
    history_df = history_df[history_df["player_id"].isin(include_players)]

    if equal_time_steps:
        date_df = history_df[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
        date_df["game number"] = date_df.index + 1
        history_df = history_df.merge(date_df, on="date", how="inner")
        x_col = "game number"
    else:
        x_col = "date"

    history_df = history_df.sort_values(["player_id", x_col]).reset_index(drop=True)

    fig = px.line(
        history_df,
        x=x_col,
        y="rating",
        color="player_id",
        color_discrete_sequence=px.colors.qualitative.Plotly + px.colors.qualitative.Set2,
    )
    fig.update_traces(mode="lines+markers")
    fig.update_layout(
        yaxis_title="Elo rating",
        title=title,
        title_x=0.5,
        legend=dict(title="<b>Player</b>", y=0.5),
        # dashed line at average rating
        shapes=[dict(
            type="line",
            yref="y",
            y0=tracker.initial_player_rating,
            y1=tracker.initial_player_rating,
            xref="paper",
            x0=0,
            x1=1,
            opacity=0.5,
            line=dict(dash="dash", width=1.5),
        )]
    )
    return fig


def display_current_ratings_table(
    current_ratings: pd.DataFrame,
    striped: bool = True,
    bordered: bool = True,
    hover: bool = False,
    **kwargs
) -> dbc.Table:
    table = dbc.Table.from_dataframe(
        current_ratings,
        striped=striped,
        bordered=bordered,
        hover=hover,
        **kwargs
    )
    return table


def display_game_results_table(
        results_history: pd.DataFrame,
        hover: bool = True,
        **kwargs
) -> dbc.Table:
    return dbc.Table.from_dataframe(results_history, hover=hover, **kwargs)


def get_tracker(
    k_value: float,
    d_value: float,
    score_function_base: float,
    initial_rating: float,
    data_to_process: pd.DataFrame = None,
) -> Tracker:
    elo_rater = MultiElo(
        k_value=k_value,
        d_value=d_value,
        score_function_base=score_function_base,
    )
    tracker = Tracker(elo_rater=elo_rater, initial_rating=initial_rating)
    if data_to_process is not None:
        tracker.process_data(data_to_process)
    return tracker


def load_json_data(json_data) -> pd.DataFrame:
    return pd.read_json(json_data, convert_dates=False)


def remove_dummy_player(
    df: pd.DataFrame,
) -> pd.DataFrame:
    dummy_player_id = config.DUMMY_PLAYER_NAME
    if dummy_player_id is None:
        return df
    df = df[df["player_id"] != dummy_player_id]
    return df


def make_ordinal(n: Union[int, str]) -> str:
    """
    Convert an integer into its ordinal representation.

    Example:
        make_ordinal(0)   => '0th'
        make_ordinal(3)   => '3rd'
        make_ordinal(122) => '122nd'
        make_ordinal(213) => '213th'
    """
    n = int(n)
    suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    return f"{n}{suffix}"
