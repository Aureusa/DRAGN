import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize
from astropy.io import fits

from data_pipeline._telescopes_db import TELESCOPES_DB
from utils import print_box
import random


# Regular expression pattern for the AGN fraction
# This pattern is used to extract the AGN fraction from the data.
AGN_FRACTION_PATTERN = rf"{TELESCOPES_DB['AGN FRACTION PATTERN']}"
REDSHIFT_PATTERN = rf"{TELESCOPES_DB['REDSHIFT PATTERN']}"


class DataAnalysisEngine:
    def __init__(self, data: list[str]|None = None):
        """
        Initialize the DataAnalysisEngine class.
        This class is responsible for analyzing the data and extracting
        useful information from it.

        :param data: The data to analyze. If None, the analysis will be skipped.
        :type data: list[str]|None
        """
        if data is not None:
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
    
    def plot_agn_histogram(
            self,
            num_bins: int = 10,
            hist_name: str = "AGN_fraction_histogram"
        ) -> None:
        """
        Plot a histogram of the AGN fraction values.

        :param num_bins: The number of bins for the histogram.
        :type num_bins: int
        :param hist_name: The name of the histogram file.
        :type hist_name: str
        """
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

    def plot_redshift_histogram(
            self,
            num_bins: int = 10,
            hist_name: str = "Redshift_histogram"
        ) -> None:
        """
        Plot a histogram of the redshift values.

        :param num_bins: The number of bins for the histogram.
        :type num_bins: int
        :param hist_name: The name of the histogram file.
        :type hist_name: str
        """
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
        """
        Create a pie chart of the AGN fraction distribution.
        """
        all_matches = self.get_agn_sorted_matches()
        matches, counts = zip(*all_matches)

        # Highlight the top 5 contributors
        top_5_labels = [f"{match} (Top {i+1})" if i < 10 else None for i, match in enumerate(matches)]

        plt.pie(counts, labels=top_5_labels, startangle=180)
        plt.axis('equal')
        plt.title("AGN Fraction Distribution")
        plt.show()
        

    # TODO: INSPECT BEFORE USE. FIX THE FUNCTION
    def plot_galaxy_grid(self, file_groups: dict, n: int = 5) -> None:
        """
        Depricated: INSPECT BEFORE USE

        Plot a grid of AGN and AGN-free images.

        :param file_groups: A dictionary containing the file groups.
        :type file_groups: dict
        :param n: The number of rows in the grid.
        :type n: int
        """
        pattern_agn_free = TELESCOPES_DB["AGN_FREE_PATERN"]
        agn_pattern = re.compile(AGN_FRACTION_PATTERN)
        redshift_pattern = re.compile(REDSHIFT_PATTERN)

        # Define the number of columns and rows
        first_key = next(iter(file_groups))
        first_item = file_groups[first_key]
        n_cols = len(first_item)
        n_rows = n

        info = "Plotting AGN and AGN-free images in a grid..."
        info += f"\nRecieved {len(file_groups)} files."
        info += f"\nPlotting {n} rows and {n_cols} columns."
        info += "\nThe first column will be AGN-free images and the rest will be AGN images."

        print_box(info)

        # Plot the images in a grid
        _, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        
        nice_keys_euclid = [('060', '31048'), ('062', '220201'), ('065', '24762'), ('064', '57713'), ('055', '274159')]
        keys_used = []
        row = 0
        while row < n_rows:
            # Select a random key from the file_groups dictionary
            random_key = random.choice(list(file_groups.keys()))
            random_key = random.choice(nice_keys_euclid)
            if random_key in keys_used:
                continue
            keys_used.append(random_key)

            # Get the corresponding files for the random key
            files = file_groups[random_key]

            agn_free = []
            agn_contam = []
            if any(re.search(pattern_agn_free, f) for f in files):
                for file in files:
                    if re.search(pattern_agn_free, file):
                        for i in range(len(files)-1):
                            agn_free.append(file)
                    else:
                        agn_contam.append(file)

            # Load the AGN free image
            with fits.open(agn_free[0]) as hdul:
                agn_free_data = hdul[0].data

            # Convert to 2D arrays if the AGN free image is 3D
            if len(agn_free_data.shape) == 3:
                agn_free_data = agn_free_data[0]

            # Skip the images if the galaxy is very feint
            if agn_free_data.std() < 0.25 or agn_free_data.std() > 1.25:
                    continue
            
            info = f"Key: {random_key}"

            norm = ImageNormalize(agn_free_data/agn_free_data.max(), stretch=AsinhStretch(), clip=True)#a=np.median(agn_free_data)

            # Find the redshift value
            redshift_match = re.search(redshift_pattern, agn_free[0])
            redshift_value = TELESCOPES_DB["SNAP-REDSHIFT MAP"][redshift_match.group(1)]

            # Add redshift value to the left of the row
            axes[row, 0].text(
                -0.25, 0.5, rf"$z = {redshift_value:.2f}$", 
                color="black", fontsize=12, transform=axes[row, 0].transAxes, 
                verticalalignment="center", horizontalalignment="right", rotation=90
            )

            # Plot the AGN-free images
            axes[row, 0].imshow(agn_free_data, cmap='gray_r', norm=norm)
            axes[row, 0].text(
                0.05, 0.95, r"$f_{AGN} = 0$", 
                color="black", fontsize=12, transform=axes[row, 0].transAxes, 
                verticalalignment="top", horizontalalignment="left"
            )

            # Sort the AGN-contaminated files by AGN fraction strength
            agn_contam = sorted(
                agn_contam,
                key=lambda agn: float(f"0.{re.search(agn_pattern, agn).group(1)}") if re.search(agn_pattern, agn) else 0
            )

            # Plot the AGN images
            for column, agn in enumerate(agn_contam):
                with fits.open(agn) as hdul:
                    agn_data = hdul[0].data

                agn_frac = re.search(agn_pattern, agn)
                if agn_frac:
                    agn_frac = float(f"0.{agn_frac.group(1)}")

                    axes[row, column+1].imshow(agn_data, cmap='gray_r', norm=norm)
                    axes[row, column+1].text(
                        0.05, 0.95, rf"$f_{{AGN}} = {agn_frac:.2f}$", 
                        color="black", fontsize=12, transform=axes[row, column+1].transAxes, 
                        verticalalignment="top", horizontalalignment="left"
                    )

            info += f"ANG Free image standart deviation: {agn_free_data.std():.2f}"
            print_box(info)
            row += 1

        # Remove x and y axis value for all images
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        info = f"Plotted {n} rows and {n_cols} columns."
        print_box(info)

        plt.tight_layout()
        plt.savefig(f"example_galaxies_grid_{n}.png")
