import pathlib

__datapath__ = pathlib.Path(__file__).parent.resolve() / "data"
"""Standard path to the data directory."""

__logpath__ = pathlib.Path(__file__).parent.parent.resolve() / "logs"
"""Standard path to the log directory."""