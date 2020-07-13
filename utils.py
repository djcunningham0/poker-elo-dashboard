from google.oauth2.service_account import Credentials
import gspread
import pandas as pd
import numpy as np
import errors as e
import dash_bootstrap_components as dbc


def raise_exception_if_not_type(x, dtype, error_type=TypeError, message=None):
    if isinstance(x, dtype):
        return x
    else:
        if message is None:
            message = f"{x} is not an instance of {dtype}"
        raise error_type(message)


def can_be_type(x, dtype):
    """
    Determine whether an object can be coerced to the specified type.

    :param x: object to change type of
    :param dtype: type to change to (e.g., int)
    :return: True if coercion is allowed, False otherwise
    """
    try:
        dtype(x)
        return True
    except ValueError:
        return False
    except TypeError:
        return False


def load_data_from_gsheet(config):
    gc = set_up_gsheets_client(config["google_sheets"]["credentials_file"])
    spreadsheet = gc.open_by_key(config["google_sheets"]["spreadsheet_id"])
    data_sheet = get_worksheet_by_id(spreadsheet, config["google_sheets"]["data_sheet_id"])
    df = worksheet_to_dataframe(data_sheet)
    return df


def set_up_gsheets_client(credentials_file):
    scopes = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]

    credentials = Credentials.from_service_account_file(
        filename=credentials_file, scopes=scopes
    )

    client = gspread.authorize(credentials)

    return client


def get_worksheet_by_id(spreadsheet, worksheet_id):
    try:
        return [w for w in spreadsheet.worksheets() if w.id == worksheet_id][0]
    except IndexError:
        raise gspread.WorksheetNotFound(f"worksheet ID {worksheet_id} does not exist")


def get_worksheet_by_name(spreadsheet, worksheet_name):
    return spreadsheet.worksheet(worksheet_name)


def worksheet_to_dataframe(worksheet, headers=True):
    data = worksheet.get_all_values()
    if headers:
        columns = data[0]
        data = data[1:]
    else:
        columns = [f"col{i}" for i in range(len(data[0]))]

    df = pd.DataFrame(data, columns=columns)
    df = replace_null_string_with_nan(df)

    return df


def replace_null_string_with_nan(df):
    return df.replace("", np.nan)


def get_dash_theme(style):
    try:
        return [getattr(dbc.themes, style)]
    except AttributeError:
        raise e.DashStyleError(f"could not find theme named '{style}'")


def prep_results_history_for_dash(data):
    results_history = data.copy()
    results_history = results_history.dropna(how="all", axis=1)  # drop columns if all NaN
    results_history = results_history.rename(columns={"date": "Date"})
    return results_history


def prep_current_ratings_for_dash(tracker):
    current_ratings = tracker.get_current_ratings()
    current_ratings["rating"] = current_ratings["rating"].round(2)
    current_ratings = current_ratings.rename(
        columns={
            "rank": "Rank",
            "player_id": "Name",
            "n_games": "Games Played",
            "rating": "Elo Rating",
        }
    )
    return current_ratings


def prep_history_plot_for_dash(tracker, title=None):
    return tracker.plot_history().update_layout(title=title, title_x=0.5)


def display_current_ratings_table(
    current_ratings, striped=True, bordered=True, hover=False, **kwargs
):
    table = dbc.Table.from_dataframe(current_ratings, striped=striped, bordered=bordered, hover=hover, **kwargs)
    return table


def display_game_results_table(results_history, hover=True, **kwargs):
    return dbc.Table.from_dataframe(results_history, hover=hover, **kwargs)
