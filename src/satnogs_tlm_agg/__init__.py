"""Aggregate SatNOGS telemetry (DB + Network) into DuckDB and Parquet."""


def main() -> None:
    from .cli import main as cli_main

    cli_main()
