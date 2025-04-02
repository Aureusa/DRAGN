def print_box(message: str):
    """
    Print a message in a box format.
    The box is created using Unicode box-drawing characters.

    :param message: The message to be printed in the box.
    :type message: str
    """
    lines = message.split('\n')
    max_length = max(len(line) for line in lines)
    border_up = '┌' + '─' * (max_length + 2) + '┐'
    border_down = '└' + '─' * (max_length + 2) + '┘'
    print(border_up)
    for line in lines:
        print(f'│ {line.ljust(max_length)} │')
    print(border_down)

'''
import pickle
from collections import defaultdict
from data_pipeline.getter import TELESCOPES_DB

from tqdm import tqdm





def finding_the_testing_data():
    with open(os.path.join("data", "train_data.pkl"), "rb") as f:
        x_t, y_t = pickle.load(f)

    with open(os.path.join("data", "val_data.pkl"), "rb") as f:
        x_v, y_v = pickle.load(f)

    print_box(f"Train len {len(x_t)}-{len(y_t)} (x-y). Should be 181385-181385.")
    print_box(f"Val len {len(x_v)}-{len(y_v)} (x-y). Should be 25913-25913.")

    # Get unique elements from all 4 lists
    unique_elements = list(set(x_t) | set(y_t) | set(x_v) | set(y_v))

    print(f"Total unique elements collected: {len(unique_elements)}")

    # Get the data folder from the database
    SCRATCH_DIR = "/scratch"
    data_folder = "data_Euclid"

    os.chdir("..")
    os.chdir("..")

    telecope_source_path = os.path.join(SCRATCH_DIR, "s4683099", data_folder)

    fits_files = glob.glob(f"{telecope_source_path}/**/*.fits", recursive=True)

    os.chdir(os.path.expanduser("~"))
    os.chdir(os.path.join(os.getcwd(), "Deep-AGN-Clean"))

    print_box(f"Found {len(fits_files)} .fits files in {telecope_source_path}\nSupposedly {len(fits_files)-len(unique_elements)} is the test set, should be bigger than 51825.")

    # Dictionary to group files by (snXXX, unique_id)
    file_groups = defaultdict(list)

    # Regular expression to extract identifiers
    pattern_agn = re.compile(rf"{TELESCOPES_DB['AGN_CONTAMINATION_PATTERN']}")
    pattern_agn_free = re.compile(rf"{TELESCOPES_DB['AGN_FREE_PATERN']}")

    unique_elements_set = set(unique_elements)
    count_agn = 0 
    count_agn_free = 0
    # Collect all file paths in the directory and its subdirectories
    for file in tqdm(fits_files, desc="Searching..."):
        if file not in unique_elements_set:
            match_agn = pattern_agn.search(file)
            match_agn_free = pattern_agn_free.search(file)
            if match_agn:
                sn_number = match_agn.group(1)  # e.g., snXXX
                unique_id = match_agn.group(2)  # e.g., unique_id
                key = (sn_number, unique_id)  # Create a tuple key
                file_groups[key].append(file)
                count_agn += 1
            elif match_agn_free:
                sn_number = match_agn_free.group(1)
                unique_id = match_agn_free.group(2)
                key = (sn_number, unique_id)
                file_groups[key].append(file)
                count_agn_free += 1

    print_box(f"Found {count_agn_free} AGN free images.\nFound {count_agn} AGN corrupted images.\nRatio {count_agn/count_agn_free}")
    print_box(f"Found {len(file_groups)} unique (snXXX, unique_id) pairs.")

    X_test_final, y_test_final = create_source_target_pairs(file_groups)

    print_box(f"Test len is now {len(X_test_final)}-{len(y_test_final)} (x-y). Should be 51825-51825.")

    with open(os.path.join("data", "test_data.pkl"), "wb") as test_file:
        pickle.dump((X_test_final, y_test_final), test_file)

    msg = "Successfully saved test_data"
    border = "=" * len(msg)
    print(border)
    print(msg)
    print(border)



def finding_the_testing_data_mse():
    with open(os.path.join("data", "train_data_mse.pkl"), "rb") as f:
        x_t, y_t = pickle.load(f)

    with open(os.path.join("data", "val_data_mse.pkl"), "rb") as f:
        x_v, y_v = pickle.load(f)

    print_box(f"Train len {len(x_t)}-{len(y_t)} (x-y). Should be 181385-181385.")
    print_box(f"Val len {len(x_v)}-{len(y_v)} (x-y). Should be 25913-25913.")

    # Val
    val_set = set(x_v)
    for y in tqdm(y_v, desc="Appending y_v into x_v..."):
        if y not in val_set:
            x_v.append(y)

    print_box(f"Appended all y_v to x_v. Now its {len(x_v)}.")

    # Train
    train_set = set(x_t)
    for y in tqdm(y_t, desc="Appending y_t into x_t..."):
        if y not in train_set:
            x_t.append(y)

    print_box(f"Appended all y_t to x_t. Now its {len(x_t)}.")

    # Concat both lists
    for x in tqdm(x_v, desc="Appending x_v into x_t (FINAL)..."):
        x_t.append(x)
    
    print_box(f"Appended all x_v to x_t. Now its {len(x_t)}. This should be rougly ~ {65694*4*0.7}")
    # x_t is everything

    # Get the data folder from the database
    SCRATCH_DIR = "/scratch"
    data_folder = "data_Euclid"

    os.chdir("..")
    os.chdir("..")

    telecope_source_path = os.path.join(SCRATCH_DIR, "s4683099", data_folder)

    fits_files = glob.glob(f"{telecope_source_path}/**/*.fits", recursive=True)

    os.chdir(os.path.expanduser("~"))
    os.chdir(os.path.join(os.getcwd(), "Deep-AGN-Clean"))

    print_box(f"Found {len(fits_files)} .fits files in {telecope_source_path}")

    # Dictionary to group files by (snXXX, unique_id)
    file_groups = defaultdict(list)

    # Regular expression to extract identifiers
    pattern_agn = re.compile(rf"{TELESCOPES_DB['AGN_CONTAMINATION_PATTERN']}")
    pattern_agn_free = re.compile(rf"{TELESCOPES_DB['AGN_FREE_PATERN']}")

    x_t = set(x_t)
    # Collect all file paths in the directory and its subdirectories
    for file in tqdm(fits_files, desc="Searching..."):
        if file not in x_t:
            match_agn = pattern_agn.search(file)
            match_agn_free = pattern_agn_free.search(file)
            if match_agn:
                sn_number = match_agn.group(1)  # e.g., snXXX
                unique_id = match_agn.group(2)  # e.g., unique_id
                key = (sn_number, unique_id)  # Create a tuple key
                file_groups[key].append(file)
            if match_agn_free:
                sn_number = match_agn_free.group(1)
                unique_id = match_agn_free.group(2)
                key = (sn_number, unique_id)
                file_groups[key].append(file)

    print_box(f"Found {len(file_groups)} unique (snXXX, unique_id) pairs.")

    X_test_final, y_test_final = create_source_target_pairs(file_groups)

    print_box(f"Test len is now {len(X_test_final)}-{len(y_test_final)} (x-y). Should be 51825-51825.")

    with open(os.path.join("data", "test_data_mse.pkl"), "wb") as test_file:
        pickle.dump((X_test_final, y_test_final), test_file)

    msg = "Successfully saved test_data"
    border = "=" * len(msg)
    print(border)
    print(msg)
    print(border)
'''