"""
NOTE FOR USERS:

This module serves only for the DRAGN project to specify filepaths and data availability for smoother data retrieval.
**Deprecated:** Do not use this if your data is stored elsewhere or follows a different structure.
"""
import os
import json

# Load the telescope database from the JSON file
db_path = os.path.join(os.path.dirname(__file__), "data_db.json")
with open(db_path, "r") as f:
    TELESCOPES_DB = json.load(f)
