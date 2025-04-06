import re
from collections import Counter
import matplotlib.pyplot as plt

from data_pipeline.getter import TELESCOPES_DB
from utils import print_box


# Regular expression pattern for the AGN fraction
# This pattern is used to extract the AGN fraction from the data.
AGN_FRACTION_PATTERN = rf"{TELESCOPES_DB['AGN FRACTION PATTERN']}"


class DataAnalysisEngine:
    def __init__(self, data: list[str]):
        print_box("Data Analysis Engine")
        print_box("Analysing data...")

        pattern = re.compile(AGN_FRACTION_PATTERN)

        matches = (f"0.{match.group(1)}" for item in data for match in [pattern.search(item)] if match)
        
        # Count occurrences of each match
        match_counts = Counter(matches)

        # Sort matches by their counts in descending order
        self.sorted_matches = sorted(match_counts.items(), key=lambda x: x[1], reverse=True)

        print_box("Data analysis completed.")

    def get_sorted_matches(self) -> list[tuple[str, int]]:
        """
        Get the sorted matches from the analysis.

        :return: A list of tuples containing the match and its count, sorted by count.
        :rtype: list[tuple[str, int]]
        """
        return self.sorted_matches
    
    def get_top_matches(self, n: int) -> list[tuple[str, int]]:
        """
        Get the top N matches from the analysis.

        :param n: The number of top matches to retrieve.
        :type n: int
        :return: A list of tuples containing the match and its count, sorted by count.
        :rtype: list[tuple[str, int]]
        """
        return self.sorted_matches[:n]
    
    def get_bottom_matches(self, n: int) -> list[tuple[str, int]]:
        """
        Get the bottom N matches from the analysis.

        :param n: The number of bottom matches to retrieve.
        :type n: int
        :return: A list of tuples containing the match and its count, sorted by count.
        :rtype: list[tuple[str, int]]
        """
        return self.sorted_matches[-n:]
    
    def plot_histogram(self) -> None:
        all_matches = self.get_sorted_matches()
        matches, counts = zip(*all_matches)

        plt.bar(matches, counts)
        plt.xlabel("AGN Fraction")
        plt.ylabel("Counts")
        plt.title("All AGN Fractions")
        plt.xticks(rotation=0, ha='right')  # Rotate and align x-axis labels for better readability
        plt.gcf().autofmt_xdate()  # Automatically adjust x-axis spacing
        plt.tight_layout()
        plt.show()

    def make_pi_chart(self) -> None:
        all_matches = self.get_sorted_matches()
        matches, counts = zip(*all_matches)

        # Highlight the top 5 contributors
        top_5_labels = [f"{match} (Top {i+1})" if i < 10 else None for i, match in enumerate(matches)]

        plt.pie(counts, labels=top_5_labels, startangle=180)
        plt.axis('equal')
        plt.title("AGN Fraction Distribution")
        plt.show()