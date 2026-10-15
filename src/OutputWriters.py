"""Interfaces and implementations for writing pipeline results."""

from typing import Optional
from abc import ABC, abstractmethod
import csv


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