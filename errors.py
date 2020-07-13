class Error(Exception):
    """Base class for other exceptions"""
    pass


class PlayerRetrievalError(Error):
    """Raised when there is an issue retrieving a Player"""
    pass


class PlayerCreationError(Error):
    """Raised when there is an issue creating a Player"""
    pass


class ScoreError(Error):
    """Raised when there is an issue with an expected or actual Elo score"""
    pass


class PlottingError(Error):
    """Raised when there is an error related to plotting"""
    pass


class DashStyleError(Error):
    """Raised when there is an error relating to the style or layout of the dashboard"""
    pass
