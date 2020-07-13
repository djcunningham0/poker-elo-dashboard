import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash_table import DataTable
from dash.dependencies import Input, Output, State
import plotly.io as pio
import pandas as pd
import argparse

import utils
from config import config
import elo


# determine if the script was run with any arguments
parser = argparse.ArgumentParser()
parser.add_argument("--debug", "-d", "--DEBUG", "-D", action="store_const", const=True,
                    help="Run in debug mode")
args, _ = parser.parse_known_args()
DEBUG = True if args.debug else False


# set up styles to be used in UI
pio.templates.default = config["dash"]["plotly_theme"]  # do this before creating any plots
center_style = {"textAlign": "center"}
external_stylesheets = utils.get_dash_theme(config["dash"]["dbc_theme"])
logo = dbc.CardImg(src=config["dash"]["logo_path"])
github_link = dbc.Row(align="center", form=True, justify="end", children=[
    dbc.Col(html.Img(src=config["dash"]["github_logo_path"]), width="auto"),
    dbc.Col(dbc.Button(children=["View on GitHub"], href=config["dash"]["github_url"], color="primary", outline=True))
])


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = config["dash"]["title"]
server = app.server

app.layout = dbc.Container(children=[

    dbc.Row(align="center", children=[
        dbc.Col(logo, width=2),
        dbc.Col(children=[
            html.H1(children=config["dash"]["title"], className="text-primary", style=center_style),
            html.H4(children=config["dash"]["subtitle"], className="text-secondary", style=center_style),
        ]),
        dbc.Col(github_link, width="auto")
    ]),

    html.Hr(),

    # load the data immediately after opening (this way the app loads a bit quicker)
    dbc.Spinner(children=[
        html.Div(id="hidden-trigger", hidden=True),
        html.Div(id="original-data", hidden=True),
        html.Div(id="main-content"),
    ]),

    html.Hr(),

    dbc.Row(justify="center", children=[
        dbc.Button("Show/hide experimental content", id="collapse-button", color="primary")
    ]),

    html.Br(),

    # I"d like to use dbc.Collapse instead of dbc.Fade but the chart doesn"t render correctly
    dbc.Fade(id="collapse", is_in=False, children=[
        dcc.Markdown(children="""
                     Edit Elo parameters and the game history to see how the Elo ratings would be affected.
                     """),

        dcc.Markdown(className="text-muted",
                     children=f"""
                     *K* controls how many Elo rating points are gained or lost in a single game. Larger
                     *K* will result in larger changes after each game. This is a standard Elo parameter.
                     (default = {config["elo"]["DEFAULT_K_VALUE"]})
                     """),

        dcc.Markdown(className="text-muted",
                     children=f"""
                     *D* controls the estimated win probability of each player. *D* value of 400 means
                     that a player with a 200-point Elo advantage wins ~75% of the time in a head-to-head
                     matchup. *D* value of 200 means that player wins ~90% of the time. This is a standard
                     Elo parameter. (default = {config["elo"]["DEFAULT_D_VALUE"]})
                     """),

        dcc.Markdown(className="text-muted",
                     children=f"""
                     The score function base value controls how much more valuable it is to finish in a
                     high place. Larger value means greater reward for finishing near the top. A value of
                     *p* means that 1st place is worth approximately *p* times as much as 2nd, which is
                     worth *p* times 3rd, and so on. This is a parameter I made up to generalize Elo to
                     multiplayer games. (default = {config["elo"]["DEFAULT_D_VALUE"]})
                     """),

        dcc.Markdown(className="text-muted",
                     children=f"""
                     If "Scale *K* with # of players" is true, Elo ratings will change more after games
                     with more players. The *K* value will be multiplied by the number of opposing players
                     before calculating change in Elo. (default = {config["elo"]["DEFAULT_SCALE_K"]})
                     """),

        dcc.Markdown(className="text-muted",
                     children=f"""
                     The regress-to-mean factor is used to regress scores toward the mean value after each
                     matchup. The idea is that inactive players should not be able to stay at the top of the
                     rankings. Larger value means more aggressive regression toward the mean, and a value of zero
                     means no regression toward the mean. (default = {config["elo"]["REGRESS_TO_MEAN_FACTOR"]})
                     """),

        dbc.Row(justify="center", align="center", children=[
            dbc.Col(width=2, children=dbc.InputGroup(children=[
                dbc.InputGroupAddon("K =", addon_type="prepend"),
                dbc.Input(id="k-value", value=config["elo"]["DEFAULT_K_VALUE"],
                          type="number", min=0, step=16)
            ])),
            dbc.Col(width=2, children=dbc.InputGroup(children=[
                dbc.InputGroupAddon("D =", addon_type="prepend"),
                dbc.Input(id="d-value", value=config["elo"]["DEFAULT_D_VALUE"],
                          type="number", min=100, step=100)
            ])),
            dbc.Col(width=3, children=dbc.InputGroup(children=[
                dbc.InputGroupAddon("score function base =", addon_type="prepend"),
                dbc.Input(id="score-function-base", value=config["elo"]["DEFAULT_SCORING_FUNCTION_BASE"],
                          type="number", min=1, max=5, step=0.05)
            ])),
            dbc.Col(width=3, children=dbc.InputGroup(children=[
                dbc.InputGroupAddon("regress-to-mean factor =", addon_type="prepend"),
                dbc.Input(id="regress-to-mean-factor", value=config["elo"]["REGRESS_TO_MEAN_FACTOR"],
                          type="number", min=0, max=1, step=0.01)
            ])),
            dbc.Col(width=2, children=dbc.InputGroup(children=[
                dbc.Checklist(id="scale-k",
                              options=[{"label": "scale K with # of players", "value": 1}],
                              value=[1] if config["elo"]["DEFAULT_SCALE_K"] else [])
            ]))
        ]),
        dbc.Row(justify="left", align="left", children=[

        ]),

        html.Br(),

        dcc.Loading(dbc.Row(justify="center", align="center", children=[
            dbc.Col(id="elo-history-table", width=config["dash"]["current_elo_table_width"]),
            dbc.Col(dcc.Graph(id="elo-history-chart"), width=config["dash"]["elo_history_chart_width"])
        ])),

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
])


@app.callback(
    Output("original-data", "children"),
    [Input("hidden-trigger", "n_clicks")]
)
def load_original_data(n):
    data = utils.load_data_from_gsheet(config)
    return data.to_json()


@app.callback(
    Output("main-content", "children"),
    [Input("original-data", "children")]
)
def load_main_content(json_data):
    data = pd.read_json(json_data, convert_dates=False)
    tracker = elo.Tracker()
    tracker.process_data(data)
    results_history = utils.prep_results_history_for_dash(data)
    current_ratings = utils.prep_current_ratings_for_dash(tracker)
    history_plot = utils.prep_history_plot_for_dash(tracker)

    out = dbc.Container(children=[

        dbc.Row(children=[
            dbc.Col(width=config["dash"]["current_elo_table_width"], children=[
                html.H4(children="Current Elo Ratings", style=center_style),
                utils.display_current_ratings_table(current_ratings)
            ]),
            dbc.Col(width=config["dash"]["elo_history_chart_width"], children=[
                html.H4(children="Elo History", style=center_style),
                dcc.Graph(id="elo-history", figure=history_plot)
            ])
        ]),

        html.Hr(),

        dbc.Row(children=[
            dbc.Col(children=[
                html.H4(children="Game Results", style=center_style),
                utils.display_game_results_table(results_history)
            ])
        ])
    ])

    return out


@app.callback(
    Output("collapse", "is_in"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_in")],
)
def toggle_collapse(n, is_in):
    if n:
        return not is_in
    return is_in


@app.callback(
    [Output(component_id="elo-history-table", component_property="children"),
     Output(component_id="elo-history-chart", component_property="figure")],
    [Input(component_id="editable-table", component_property="data"),
     Input(component_id="k-value", component_property="value"),
     Input(component_id="d-value", component_property="value"),
     Input(component_id="score-function-base", component_property="value"),
     Input(component_id="scale-k", component_property="value"),
     Input(component_id="regress-to-mean-factor", component_property="value")]
)
def update_chart_and_figure(tmp_data, k, d, base, scale_k, regress_to_mean_factor):
    # get data from editable table
    tmp_data = pd.DataFrame(tmp_data)
    tmp_data = utils.replace_null_string_with_nan(tmp_data)

    # set up Elo tracker from editable parameters and process data
    scale_k = True if len(scale_k) > 0 else False  # will either be [1] or []
    elo_rater = elo.MultiElo(k_value=k, d_value=d, score_function_base=base, scale_k=scale_k)
    tmp_tracker = elo.Tracker(elo_rater=elo_rater, regress_to_mean_factor=regress_to_mean_factor)
    tmp_tracker.process_data(tmp_data)

    # get current ratings for table
    tmp_ratings = utils.prep_current_ratings_for_dash(tmp_tracker)

    # get plot of Elo history
    title = f"Elo history -- K={k}, D={d}, base={base}, scale_k={scale_k}"
    tmp_fig = utils.prep_history_plot_for_dash(tmp_tracker, title=title)

    return utils.display_current_ratings_table(tmp_ratings), tmp_fig


@app.callback(
    [Output("editable-table", "data"),
     Output("editable-table", "columns")],
    [Input("editing-rows-button", "n_clicks"),
     Input("original-data", "children")],
    [State("editable-table", "data"),
     State("editable-table", "columns")])
def build_editable_table(n_clicks, orig_data, current_data, current_columns):
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

if __name__ == "__main__":
    app.run_server(debug=DEBUG)
