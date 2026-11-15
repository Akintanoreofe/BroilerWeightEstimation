"""Interfaces and implementations for writing pipeline results."""

from typing import Optional
from abc import ABC, abstractmethod
import csv
import json


FEATURE_CSV_FIELDS = (
    "video_name", "recording_date", "frame_idx", "timestamp_s", "segmenter",
    "track_id", "measurable", "feeder_contact", "occluder_contact", "solidity",
    "ellipse_aspect", "sharpness", "dist_feeder_radii", "area_cm2", "length_cm",
    "max_width_cm", "perimeter_cm", "volume_cm3", "n_slices", "cross_section",
    "k", "px_per_cm", "feeder_major_px", "feeder_minor_px",
    "calib_uncertainty_pct", "slice_widths_cm", "weight_kg_est",
)


class OutputWriter(ABC):
    """Abstract interface for writing and closing result outputs."""

    @abstractmethod
    def write(self, record: dict):
        """Write one result record.

        Parameters
        ----------
        record : dict
            Result values to write.
        """
        pass

    @abstractmethod
    def close(self):
        """Release resources associated with the output."""
        pass


class CSVOutputWriter(OutputWriter):
    """Write result records to a CSV file."""

    def __init__(self, csv_path: str):
        """Open the destination CSV file for writing.

        Parameters
        ----------
        csv_path : str
            Destination file path. Existing contents are overwritten.
        """
        self.csv_file = open(csv_path, "w", newline="")
        self.writer: Optional[csv.DictWriter] = None
        self.csv_path = csv_path

    def write(self, record: dict):
        """Write a record, creating the header from its keys on first use.

        Parameters
        ----------
        record : dict
            Mapping of CSV column names to values.
        """
        if self.writer is None:
            self.writer = csv.DictWriter(self.csv_file, fieldnames=list(record.keys()))
            self.writer.writeheader()
        self.writer.writerow(record)

    def close(self):
        """Close the CSV file and report its destination."""
        self.csv_file.close()
        print(f"[CSVOutputWriter] Saved results to {self.csv_path}")


class FeatureCSVWriter(OutputWriter):
    """Write one validation feature row per selected bird and frame."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=FEATURE_CSV_FIELDS)
        self.writer.writeheader()

    def write(self, record: dict):
        row = dict(record)
        widths = row.get("slice_widths_cm")
        if widths is not None and not isinstance(widths, str):
            row["slice_widths_cm"] = json.dumps(widths)
        self.writer.writerow(row)

    def close(self):
        self.csv_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()