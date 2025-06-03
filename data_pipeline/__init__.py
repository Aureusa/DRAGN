"""
Example: End-to-End Data Pipeline Usage for the DRAGN Project

This example demonstrates how to use the main components of the data_pipeline package
to prepare and load data for training or analysis.

------------------------------------------------------------
Step 1: Import the main classes and functions
------------------------------------------------------------
from data_pipeline import (
    FilepathGetter,
    ForgeData,
    GalaxyDataset,
    FitsLoader,
    DataAnalysisEngine
)

------------------------------------------------------------
Step 2: Retrieve file paths using FilepathGetter
------------------------------------------------------------
# Initialize the FilepathGetter with the path to your data directory
getter = FilepathGetter(path="/path/to/your/data")

# Get grouped file paths and unique keys
file_groups, unique_keys = getter.get_data()

------------------------------------------------------------
Step 3: Split data into train/val/test using ForgeData
------------------------------------------------------------
forge = ForgeData()
X_train, y_train, X_val, y_val, X_test, y_test = forge.forge_training_data(file_groups)

------------------------------------------------------------
Step 4: Create GalaxyDataset objects for each split
------------------------------------------------------------
train_dataset = GalaxyDataset(X_train, y_train)
val_dataset = GalaxyDataset(X_val, y_val)
test_dataset = GalaxyDataset(X_test, y_test)

------------------------------------------------------------
Step 5: Create DataLoaders for each split
------------------------------------------------------------
train_loader = FitsLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = FitsLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = FitsLoader(test_dataset, batch_size=32, shuffle=False)

------------------------------------------------------------
Step 6: (Optional) Analyze your data
------------------------------------------------------------
# Analyze AGN fraction and redshift distributions
analysis = DataAnalysisEngine(X_train + X_val + X_test)
analysis.plot_agn_histogram()
analysis.plot_redshift_histogram()

------------------------------------------------------------
Notes:
- All datasets must inherit from _BaseDataset and follow the FITS filename conventions described in the DRAGN project.
- The FilepathGetter, ForgeData, and GalaxyDataset classes are tailored for the DRAGN data structure and may not work with arbitrary data.
- For more details, see the documentation for each class or contact the maintainers.
"""
from data_pipeline._telescopes_db import TELESCOPES_DB
from data_pipeline.data_analysis import DataAnalysisEngine
from data_pipeline.data_split import (
    create_source_target_pairs,
    test_train_val_split,
    ForgeData
)
from data_pipeline.galaxy_dataset import (
    _BaseDataset,
    GalaxyDataset
)
from data_pipeline.getter import FilepathGetter
from data_pipeline.loaders import FitsLoader
from data_pipeline.transforms import (
    _BaseTransform,
    PerImageMinMax,
    PerImageNormalize,
    NormalizationParams
)
