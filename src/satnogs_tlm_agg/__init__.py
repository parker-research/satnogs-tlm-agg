"""Aggregate SatNOGS telemetry (DB + Network) into DuckDB and Parquet."""


def main() -> None:
    # Print a traceback if a compiled dependency (polars, duckdb, satkit, ...) crashes the
    # process (e.g. "illegal instruction" on CPUs without the instructions its wheel targets);
    # without this the process dies silently with a non-zero exit code.
    import faulthandler

    faulthandler.enable()

    from .cli import main as cli_main

    cli_main()
