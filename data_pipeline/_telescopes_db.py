import os
import json

# Load the telescope database from the JSON file
db_path = os.path.join(os.path.join(os.getcwd(), "data_pipeline"), "data_db.json")
with open(db_path, "r") as f:
    TELESCOPES_DB = json.load(f)
