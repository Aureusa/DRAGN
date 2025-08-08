import glob


def get_real_data_filepaths(real_images_path: str|None = "/scratch/s4683099/real_JWST/COSMOS-Web_cutouts_Zhuang2024"):
    fits_files = glob.glob(f"{real_images_path}/*.fits", recursive=True)
    return fits_files
