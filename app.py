import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash_table import DataTable
from dash.dependencies import Input, Output, State
import plotly.io as pio
import pandas as pd
from multielo import MultiElo

import argparse
from typing import List

import utils
import config


# determine if the script was run with any arguments
parser = argparse.ArgumentParser()
parser.add_argument("--debug", "-d", "--DEBUG", "-D", action="store_const", const=True,
                    help="Run in debug mode")
args, _ = parser.parse_known_args()
DEBUG = True if args.debug else False


# set up styles to be used in UI
pio.templates.default = config.PLOTLY_THEME  # do this before creating any plots
center_style = {"textAlign": "center"}
external_stylesheets = utils.get_dash_theme(config.DBC_THEME)


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = config.TITLE
server = app.server
app.config["suppress_callback_exceptions"] = True


def header():
    logo = dbc.CardImg(src=config.LOGO_PATH)
    # TODO: make GitHub link open in a new tab
    github_link = dbc.Row(align="center", form=True, justify="end", children=[
        dbc.Col(html.Img(src=config.GITHUB_LOGO_PATH), width="auto"),
        dbc.Col(dbc.Button(children=["View on GitHub"], href=config.GITHUB_URL, color="primary", outline=True))
    ])

    return html.Div([
        dbc.Row(align="center", children=[
            dbc.Col(logo, width=2),
            dbc.Col(children=[
                html.H1(children=config.TITLE, className="text-primary", style=center_style),
                html.H4(children=config.SUBTITLE, className="text-secondary", style=center_style),
            ]),
            dbc.Col(github_link, width="auto")
        ]),
        html.Hr(),
    ])


def footer():
    return html.Div([
        html.Hr(),
        html.Small("*We claim no affiliation with Red Dog, Miller Brewing Company, or "
                   "Plank Road Brewing, but we would happily promote for you if you in "
                   "exchange for free beer :)",
                   className="text-muted font-italic"),
        html.Hr(),
    ])


def section_header(text: str):
    return html.H4(children=text, style=center_style)


def current_elo_tab():
    return html.Div(children=[
        dbc.Row(children=[
            dbc.Col(width=5, children=[
                section_header("Current Elo Ratings"),
                dbc.Spinner(id="current-ratings-table")
            ]),
            dbc.Col(width=7, children=[
                section_header("Elo History"),
                dbc.Spinner(children=dcc.Graph(id="main-chart")),
                dbc.Col(width=6, children=[
                    dbc.Row(children=[
                        daq.BooleanSwitch(id="time-step-toggle-input", on=True,
                                          style={"margin-left": "20px", "margin-right": "10px"}),
                        dcc.Markdown(className="text-muted",
                                     children="use equally spaced time steps"),
                        html.Div(id="time-step-null-output", hidden=True),
                    ]),

                    dbc.InputGroup(children=[
                        dbc.InputGroupAddon("Minimum games played", addon_type="prepend"),
                        dbc.Input(
                            id="min-games-input",
                            value=1,
                            type="number",
                            min=1,
                            step=1,
                        )
                    ])
                ]),
            ])
        ]),

        html.Hr(),

        dbc.Row(children=[
            dbc.Col(children=[
                section_header("Game Results"),
                dbc.Spinner(id="game-results-table")
            ])
        ])
    ])


def scenario_generator_tab():
    return html.Div([
        dcc.Markdown(children="""
                     Edit Elo parameters and the game history to see how the Elo ratings would be affected.
                     """),

        dcc.Markdown(className="text-muted",
                     children=f"""
                     *K* controls how many Elo rating points are gained or lost in a single game. Larger
                     *K* will result in larger changes after each game. This is a standard Elo parameter.
                     (default = {config.DEFAULT_K_VALUE})
                     """),

        dcc.Markdown(className="text-muted",
                     children=f"""
                     *D* controls the estimated win probability of each player. *D* value of 400 means
                     that a player with a 200-point Elo advantage wins ~75% of the time in a head-to-head
                     matchup. *D* value of 200 means that player wins ~90% of the time. This is a standard
                     Elo parameter. (default = {config.DEFAULT_D_VALUE})
                     """),

        dcc.Markdown(className="text-muted",
                     children=f"""
                     The score function base value controls how much more valuable it is to finish in a
                     high place. Larger value means greater reward for finishing near the top. A value of
                     *p* means that 1st place is worth approximately *p* times as much as 2nd, which is
                     worth *p* times 3rd, and so on. This is a parameter I made up to generalize Elo to
                     multiplayer games. (default = {config.DEFAULT_SCORING_FUNCTION_BASE})
                     """),

        dbc.Row(justify="center", align="center", children=[
            dbc.Col(width=4, children=dbc.InputGroup(children=[
                dbc.InputGroupAddon("K =", addon_type="prepend"),
                dbc.Input(id="k-value", value=config.DEFAULT_K_VALUE,
                          type="number", min=0, step=16)
            ])),
            dbc.Col(width=4, children=dbc.InputGroup(children=[
                dbc.InputGroupAddon("D =", addon_type="prepend"),
                dbc.Input(id="d-value", value=config.DEFAULT_D_VALUE,
                          type="number", min=100, step=100)
            ])),
            dbc.Col(width=4, children=dbc.InputGroup(children=[
                dbc.InputGroupAddon("score function base =", addon_type="prepend"),
                dbc.Input(id="score-function-base", value=config.DEFAULT_SCORING_FUNCTION_BASE,
                          type="number", min=1, max=5, step=0.05)
            ])),
        ]),

        html.Br(),

        dcc.Loading(dbc.Row(justify="center", align="start", children=[
            dbc.Col(id="elo-scenario-table", width=5),
            dbc.Col(width=7, children=[
                dcc.Graph(id="elo-scenario-chart"),
                dbc.Col(width=6, children=[
                    dbc.Row(children=[
                        daq.BooleanSwitch(id="time-step-toggle-input", on=True,
                                          style={"margin-left": "20px", "margin-right": "10px"}),
                        dcc.Markdown(className="text-muted",
                                     children="use equally spaced time steps"),
                        html.Div(id="time-step-null-output", hidden=True),
                    ]),

                    dbc.InputGroup(children=[
                        dbc.InputGroupAddon("Minimum games played", addon_type="prepend"),
                        dbc.Input(
                            id="min-games-input",
                            value=1,
                            type="number",
                            min=1,
                            step=1,
                        )
                    ])
                ]),
            ]),
        ])),

        html.Br(),
        html.Hr(),

        section_header("Editable Game Result History"),

        dbc.Row(justify="center", children=[
            dbc.Col(children=[
                DataTable(id="editable-table", editable=True, row_deletable=True)
            ])
        ]),

        dbc.Row(justify="center", children=[
            dbc.Button(
                "add row",
                id="editing-rows-button",
                color="secondary",
                n_clicks=0
            )
        ])
    ])


def win_probability_tab():
    return html.Div(children=[
        dbc.Row(children=[
            dbc.Col(width=3, children=[
                section_header("Player List"),
                dcc.Markdown(className="text-muted",
                             children="Select which players will participate."),
                dbc.Checklist(id="player-options", value=[]),
                html.Br(),
                dbc.Button(id="clear-button", children="Clear selections", color="primary")
            ]),

            dbc.Col(width=9, children=[
                section_header("Probability of Results"),
                html.Div(id="win-probability-table")
            ])
        ]),
    ])


@app.callback(
    Output(component_id="tab-content", component_property="children"),
    Input(component_id="tab-name", component_property="value")
)
def render_content(tab_name: str):
    if tab_name == "tab-1":
        return current_elo_tab()
    elif tab_name == "tab-2":
        return scenario_generator_tab()
    elif tab_name == "tab-3":
        return win_probability_tab()


app.layout = dbc.Container(children=[

    header(),

    # load the data immediately after opening (this way the app loads a bit quicker)
    html.Div(children=[
        html.Div(id="hidden-trigger", hidden=True),
        html.Div(id="original-data", hidden=True),
    ]),

    dcc.Tabs(id="tab-name", value="tab-1", children=[
        dcc.Tab(label="Current Elo Ratings", value="tab-1"),
        dcc.Tab(label="Scenario Generator", value="tab-2"),
        dcc.Tab(label="Win Probabilities", value="tab-3"),
    ]),

    html.Br(),
    html.Div(id="tab-content"),

    footer(),

])


@app.callback(
    Output("original-data", "children"),
    [Input("hidden-trigger", "n_clicks")]
)
def load_original_data(_):
    data = utils.load_data_from_gsheet()
    return data.to_json()


@app.callback(
    [Output(component_id="current-ratings-table", component_property="children"),
     Output(component_id="game-results-table", component_property="children")],
    [Input(component_id="original-data", component_property="children"),
     Input(component_id="min-games-input", component_property="value")]
)
def load_current_elo_tables(json_data: List[dict], min_games: int):
    data = utils.load_json_data(json_data)
    tracker = utils.get_tracker(
        k_value=config.DEFAULT_K_VALUE,
        d_value=config.DEFAULT_D_VALUE,
        score_function_base=config.DEFAULT_SCORING_FUNCTION_BASE,
        initial_rating=config.INITIAL_RATING,
        data_to_process=data,
    )
    results_history = utils.prep_results_history_for_dash(data)
    current_ratings = utils.prep_current_ratings_for_dash(
        tracker=tracker,
        results_history=results_history,
        min_games=min_games
    )
    return (
        utils.display_current_ratings_table(current_ratings),
        utils.display_game_results_table(results_history)
    )


@app.callback(
    Output(component_id="main-chart", component_property="figure"),
    [Input(component_id="original-data", component_property="children"),
     Input(component_id="time-step-toggle-input", component_property="on"),
     Input(component_id="min-games-input", component_property="value")]
)
def load_current_elo_chart(
        json_data: List[dict],
        equal_time_steps: bool,
        min_games: int,
):
    data = utils.load_json_data(json_data)
    tracker = utils.get_tracker(
        k_value=config.DEFAULT_K_VALUE,
        d_value=config.DEFAULT_D_VALUE,
        score_function_base=config.DEFAULT_SCORING_FUNCTION_BASE,
        initial_rating=config.INITIAL_RATING,
        data_to_process=data,
    )
    history_plot = utils.plot_tracker_history(
        tracker=tracker,
        equal_time_steps=equal_time_steps,
        min_games=min_games,
    )
    return history_plot


@app.callback(
    Output("time-step-null-output", "children"),
    [Input("time-step-toggle-input", "on")]
)
def toggle_time_steps(value: bool) -> bool:
    return value


@app.callback(
    [Output(component_id="elo-scenario-table", component_property="children"),
     Output(component_id="elo-scenario-chart", component_property="figure")],
    [Input(component_id="editable-table", component_property="data"),
     Input(component_id="k-value", component_property="value"),
     Input(component_id="d-value", component_property="value"),
     Input(component_id="score-function-base", component_property="value"),
     Input(component_id="time-step-toggle-input", component_property="on"),
     Input(component_id="min-games-input", component_property="value")]
)
def update_scenario_generator_chart_and_figure(
        tmp_data: List[dict],
        k: float,
        d: float,
        base: float,
        equal_time_steps: bool,
        min_games: int,
):
    # get data from editable table
    tmp_data = pd.DataFrame(tmp_data)
    tmp_data = utils.replace_null_string_with_nan(tmp_data)

    # set up Elo tracker from editable parameters and process data
    tmp_tracker = utils.get_tracker(
        k_value=k,
        d_value=d,
        score_function_base=base,
        initial_rating=config.INITIAL_RATING,
        data_to_process=tmp_data,
    )

    # get current ratings for table
    results_history = utils.prep_results_history_for_dash(tmp_data)
    tmp_ratings = utils.prep_current_ratings_for_dash(
        tracker=tmp_tracker,
        results_history=results_history,
        min_games=min_games
    )

    # get plot of Elo history
    title = f"Elo history -- K={k}, D={d}, base={base}"
    tmp_fig = utils.plot_tracker_history(
        tracker=tmp_tracker,
        title=title,
        equal_time_steps=equal_time_steps,
        min_games=min_games,
    )

    return utils.display_current_ratings_table(tmp_ratings), tmp_fig


@app.callback(
    [Output("editable-table", "data"),
     Output("editable-table", "columns")],
    [Input("editing-rows-button", "n_clicks"),
     Input("original-data", "children")],
    [State("editable-table", "data"),
     State("editable-table", "columns")])
def build_editable_table(
        n_clicks: int,
        orig_data: List[dict],
        current_data: List[dict],
        current_columns: List[dict],
):
    # when the app loads, use the original data
    if n_clicks == 0:
        df = pd.read_json(orig_data, convert_dates=False)
        data = df.to_dict("records")
        columns = [{"id": col, "name": col} for col in df.columns]

    # after that, add a new row whenever the button is clicked
    else:
        data = current_data + [{c["id"]: "" for c in current_columns}]
        columns = current_columns

    return data, columns


@app.callback(
    Output(component_id="player-options", component_property="options"),
    Input(component_id="original-data", component_property="children")
)
def get_player_options(json_data: List[dict]):
    data = utils.load_json_data(json_data)
    tracker = utils.get_tracker(
        k_value=config.DEFAULT_K_VALUE,
        d_value=config.DEFAULT_D_VALUE,
        score_function_base=config.DEFAULT_SCORING_FUNCTION_BASE,
        initial_rating=config.INITIAL_RATING,
        data_to_process=data,
    )
    player_df = tracker.player_df
    player_df = utils.remove_dummy_player(player_df)
    return [{"label": f"{player.id} (rating = {player.rating:.2f})", "value": (player.id, player.rating)}
            for player in player_df["player"]]


@app.callback(
    Output(component_id="player-options", component_property="value"),
    Input(component_id="clear-button", component_property="n_clicks")
)
def clear_selection(_):
    return []


@app.callback(
    Output(component_id="win-probability-table", component_property="children"),
    Input(component_id="player-options", component_property="value")
)
def create_win_probability_table(players: List[tuple]):
    if not players:
        return None

    n_players = len(players)
    player_ids = [x[0] for x in players]
    ratings = [x[1] for x in players]

    elo = MultiElo(
        k_value=config.DEFAULT_K_VALUE,
        d_value=config.DEFAULT_D_VALUE,
        score_function_base=config.DEFAULT_SCORING_FUNCTION_BASE,
    )

    result_mx = elo.simulate_win_probabilities(ratings, n_sim=int(1e5), seed=0)
    df = pd.DataFrame(result_mx)

    df["Player"] = player_ids

    place_cols = [utils.make_ordinal(x + 1) for x in range(n_players)]
    rename_dict = dict(zip(range(n_players), place_cols))
    df = df.rename(columns=rename_dict)
    df = df[["Player", *[x for x in df.columns if x != "Player"]]]
    df = df.apply(lambda row: [f"{x:.1%}" if isinstance(x, float) else x for x in row])

    return dbc.Table.from_dataframe(
        df,
        bordered=True,
        striped=True,
    )


if __name__ == "__main__":
    app.run_server(debug=DEBUG)
