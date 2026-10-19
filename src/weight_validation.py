"""Compare estimated broiler weights with measured weights from a CSV file."""

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple


def load_weight_pairs(
    csv_path: Path, actual_column: str, estimated_column: str
) -> List[Tuple[float, float]]:
    """Load (actual, estimated) weight pairs and validate the input rows."""
    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("CSV file is empty or has no header.")

        missing_columns = {
            actual_column,
            estimated_column,
        } - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                "Missing CSV column(s): " + ", ".join(sorted(missing_columns))
            )

        pairs = []
        for line_number, row in enumerate(reader, start=2):
            try:
                actual = float(row[actual_column])
                estimated = float(row[estimated_column])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid or missing weight value on CSV line {line_number}."
                ) from None

            if not math.isfinite(actual) or not math.isfinite(estimated):
                raise ValueError(
                    f"Weight values must be finite on CSV line {line_number}."
                )
            pairs.append((actual, estimated))

    if not pairs:
        raise ValueError("CSV contains no weight rows.")
    return pairs


def calculate_metrics(pairs: List[Tuple[float, float]]) -> dict:
    """Calculate common error metrics; weights are in kilograms."""
    errors = [estimated - actual for actual, estimated in pairs]
    absolute_errors = [abs(error) for error in errors]
    nonzero_actual_errors = [
        abs(error) / abs(actual) * 100
        for (actual, _), error in zip(pairs, errors)
        if actual != 0
    ]

    return {
        "count": len(pairs),
        "mae": sum(absolute_errors) / len(pairs),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(pairs)),
        "bias": sum(errors) / len(pairs),
        "mape": (
            sum(nonzero_actual_errors) / len(nonzero_actual_errors)
            if nonzero_actual_errors
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare estimated broiler weights with measured weights in a CSV."
    )
    parser.add_argument("csv_file", type=Path, help="CSV containing paired weights")
    parser.add_argument(
        "--actual-column",
        default="actual_weight_kg",
        help="Measured-weight column (default: actual_weight_kg)",
    )
    parser.add_argument(
        "--estimated-column",
        default="volume",
        help="Estimated-weight column (default: volume, as written by this project)",
    )
    args = parser.parse_args()

    try:
        pairs = load_weight_pairs(
            args.csv_file, args.actual_column, args.estimated_column
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    metrics = calculate_metrics(pairs)
    print(f"Compared weights: {metrics['count']}")
    print(f"MAE:  {metrics['mae']:.4f} kg")
    print(f"RMSE: {metrics['rmse']:.4f} kg")
    print(f"Bias: {metrics['bias']:.4f} kg (estimate - actual)")
    if metrics["mape"] is None:
        print("MAPE: unavailable (no nonzero actual weights)")
    else:
        print(f"MAPE: {metrics['mape']:.2f}% (zero actual weights excluded)")


if __name__ == "__main__":
    main()
