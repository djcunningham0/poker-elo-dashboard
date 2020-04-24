import score_functions as score


# parameters for the ELO algorithm -- setting kind of arbitrarily for now
config = {
    'DEFAULT_K_VALUE': 32,
    'DEFAULT_D_VALUE': 400,
    'INITIAL_RATING': 1000,
    'DEFAULT_SCORING_FUNCTION': score.create_exponential_score_function(base=1.5)
}
