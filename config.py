config = {
    # parameters for the Elo algorithm -- setting kind of arbitrarily for now, should tune once we have more data
    "elo": {
        "DEFAULT_K_VALUE": 32,
        "DEFAULT_D_VALUE": 400,
        "DEFAULT_SCORING_FUNCTION_BASE": 1.25,
        "DEFAULT_SCALE_K": True,
        "INITIAL_RATING": 1000,
        "REGRESS_TO_MEAN_FACTOR": 0,
    },
    "google_sheets": {
        "credentials_file": "./google-credentials.json",
        "spreadsheet_id": "1ZuEwqaBx4N_VX7nKGZ0nPcBilRLpvrQA00ddnrbEIOU",
        "data_sheet_id": 626646746,
    },
    "dash": {
        "dbc_theme": "FLATLY",  # others I liked: DARKLY, SIMPLEX, LUMEN (https://bootswatch.com/flatly/)
        "plotly_theme": "plotly_white",
        "logo_path": "/assets/Red-Dog-Txt-&-Logo.jpg",
        "github_logo_path": "assets/GitHub-Mark-32px.png",
        "github_url": "https://github.com/djcunningham0/poker-elo",
        "title": "Red Dog Poker Network",
        "subtitle": "Home of Uncommonly Smooth Poker",
        "current_elo_table_width": 5,
        "elo_history_chart_width": 7,
    },
}
