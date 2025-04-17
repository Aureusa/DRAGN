import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

from data_pipeline.getter import TELESCOPES_DB
from utils import print_box


# Regular expression pattern for the AGN fraction
# This pattern is used to extract the AGN fraction from the data.
AGN_FRACTION_PATTERN = rf"{TELESCOPES_DB['AGN FRACTION PATTERN']}"
REDSHIFT_PATTERN = rf"{TELESCOPES_DB['REDSHIFT PATTERN']}"


class DataAnalysisEngine:
    def __init__(self, data: list[str]):
        print_box("Data Analysis Engine")
        print_box("Analysing data...")

        agn_pattern = re.compile(AGN_FRACTION_PATTERN)

        agn_matches = (float(f"0.{match.group(1)}") for item in data for match in [agn_pattern.search(item)] if match)
        
        # Count occurrences of each match
        agn_match_counts = Counter(agn_matches)

        # Sort matches by their counts in descending order
        self.agn_sorted_matches = sorted(agn_match_counts.items(), key=lambda x: x[1], reverse=True)

        # Redshift pattern
        redshift_pattern = re.compile(REDSHIFT_PATTERN)

        # "_sn(\\d{3})_.*?_(\\d+)\\.fits"
        # Extract redshift values from the data
        redshift_matches = (TELESCOPES_DB["SNAP-REDSHIFT MAP"][match.group(1)] for item in data for match in [redshift_pattern.search(item)] if match)

        # Count occurrences of each redshift match
        redshift_match_counts = Counter(redshift_matches)

        # Sort redshift matches by their counts in descending order
        self.redshift_sorted_matches = sorted(redshift_match_counts.items(), key=lambda x: x[1], reverse=True)

        print_box("Data analysis completed.")

    def get_agn_sorted_matches(self) -> list[tuple[str, int]]:
        """
        Get the sorted matches from the analysis.

        :return: A list of tuples containing the match and its count, sorted by count.
        :rtype: list[tuple[str, int]]
        """
        return self.agn_sorted_matches
    
    def get_top_agn_matches(self, n: int) -> list[tuple[str, int]]:
        """
        Get the top N matches from the analysis.

        :param n: The number of top matches to retrieve.
        :type n: int
        :return: A list of tuples containing the match and its count, sorted by count.
        :rtype: list[tuple[str, int]]
        """
        return self.agn_sorted_matches[:n]
    
    def get_bottom_agn_matches(self, n: int) -> list[tuple[str, int]]:
        """
        Get the bottom N matches from the analysis.

        :param n: The number of bottom matches to retrieve.
        :type n: int
        :return: A list of tuples containing the match and its count, sorted by count.
        :rtype: list[tuple[str, int]]
        """
        return self.agn_sorted_matches[-n:]
    
    def plot_agn_histogram(self, num_bins: int = 10, hist_name: str = "AGN_fraction_histogram") -> None:
        all_matches = self.get_agn_sorted_matches()
        matches, counts = zip(*all_matches)

        # Convert matches to floats for binning
        matches = np.array(matches, dtype=float)

        # Create bins for the histogram
        bins = np.linspace(matches.min(), matches.max(), num_bins + 1)

        # Plot the histogram
        plt.hist(matches, bins=bins, weights=counts, edgecolor="black", alpha=0.7)
        plt.xlabel("AGN Fraction")
        plt.ylabel("Counts")
        plt.title("AGN Fraction Histogram")
        plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
        plt.tight_layout()
        plt.savefig(f"{hist_name}.png")
        plt.close()

    def plot_redshift_histogram(self, num_bins: int = 10, hist_name: str = "Redshift_histogram") -> None:
        all_matches = self.redshift_sorted_matches
        matches, counts = zip(*all_matches)

        # Convert matches to floats for binning
        matches = np.array(matches, dtype=float)

        # Create bins for the histogram
        bins = np.linspace(matches.min(), matches.max(), num_bins + 1)

        # Plot the histogram
        plt.hist(matches, bins=bins, weights=counts, edgecolor="black", alpha=0.7)
        plt.xlabel("Redshift")
        plt.ylabel("Counts")
        plt.title("Redshift Histogram")
        plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
        plt.tight_layout()
        plt.savefig(f"{hist_name}.png")
        plt.close()

    def make_pi_chart(self) -> None:
        all_matches = self.get_agn_sorted_matches()
        matches, counts = zip(*all_matches)

        # Highlight the top 5 contributors
        top_5_labels = [f"{match} (Top {i+1})" if i < 10 else None for i, match in enumerate(matches)]

        plt.pie(counts, labels=top_5_labels, startangle=180)
        plt.axis('equal')
        plt.title("AGN Fraction Distribution")
        plt.show()
        