import os
import glob
import numpy as np
from tqdm import tqdm

from model_testing import PAdict, ResultInterpreter, Plotter, Tester
from loggers_utils import TrainingLogger
from data_pipeline import _BaseDataset, _BaseTransform, _BaseLoader
from utils import load_pkl_file, print_box

def plot_loss_pipeline(data_folder: str, filename: str, discriminator: bool = False, from_epoch: int = 0, to_epoch: int = -1) -> None:
    logger = TrainingLogger(
        save_dir=data_folder,
        adverserial_logger=discriminator
    )

    history = logger.history

    train_loss = history["train_loss"][from_epoch:to_epoch]
    val_loss = history["val_loss"][from_epoch:to_epoch]
    best_val_loss = min(val_loss)

    if discriminator:
        train_loss = history["train_loss_D"][from_epoch:to_epoch]
        val_loss = history["val_loss_D"][from_epoch:to_epoch]

    plotter = Plotter()

    plotter.plot_loss(
        train_loss=train_loss,
        val_loss=val_loss,
        best_val_loss=best_val_loss,
        filename=filename,
        data_folder=data_folder
    )

def image_cleaner_pipeline(
        model_names: list[str],
        model_types: list[str],
        model_filenames: list[str],
        data_folders: list[str],
        datasets: list[_BaseDataset],
        loaders: list[_BaseLoader],
        transforms: list[_BaseTransform|None],
        n: int,
        test_data_path: str|None = "/home4/s4683099/Deep-AGN-Clean/data/jwst_full_data/test_data.pkl",
        plots_filename: str = "test_image",
        show_real_min_infered: bool = False,
        f_agn: int|None = None,
        report: bool = False,
    ) -> None:
    _check_list_lengths(
        model_names=model_names,
        model_types=model_types,
        model_filenames=model_filenames,
        data_folder=data_folders,
        dataset=datasets,
        transform=transforms,
        loader=loaders,
    )
    # Load the test data
    X_test, y_test = load_pkl_file(test_data_path)

    source_list = []
    target_list = []
    cleaned_images_list = []
    psf_predicted_list = []
    psfs_list = []

    for i in range(len(model_names)):
        model_type = model_types[i]
        model_filename = model_filenames[i]
        data_folder = data_folders[i]

        transform = transforms[i] if transforms else None
        dataset = datasets[i](source=X_test, target=y_test, transform=transform, training=False)
        loader = loaders[i](dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
        print(data_folder)
        tester = Tester(
            model_type=model_type,
            model_filename=model_filename,
            data_folder=data_folder,
            test_loader=loader,
            transform=transform,
        )

        (
            source_arr,
            target_arr,
            cleaned_image_arr,
            psf_predicted_arr,
            psf_arr
        ) = tester.clean_images(n=n, f_agn=f_agn)

        source_list.append(source_arr)
        target_list.append(target_arr)
        cleaned_images_list.append(cleaned_image_arr)
        psf_predicted_list.append(psf_predicted_arr)
        psfs_list.append(psf_arr)

    plotter = Plotter()
    
    if report:
        plotter.grid_plot(
            sources=source_list,
            targets=target_list,
            outputs=cleaned_images_list,
            titles=model_names,
            filename=plots_filename,
            data_folder=os.getcwd(),
            f_agn=f_agn,
            save=True,
        )
    else:
        plotter.diagnostic_plot(
            sources=source_list,
            targets=target_list,
            outputs=cleaned_images_list,
            predicted_psfs=psf_predicted_list,
            psfs=psfs_list,
            titles=model_names,
            filename=plots_filename,
            data_folder=os.getcwd(),
            show_real_min_infered=show_real_min_infered,
            save=True,
        )

# DEPRICATED: This function is not used anymore.
def plot_gal_3dgal_psf():
    from data_pipeline.galaxy_dataset import MockRealGalaxyDataset

    real_images_path: str|None = "/scratch/s4683099/real_JWST/COSMOS-Web_cutouts_Zhuang2024"
    fits_files = glob.glob(f"{real_images_path}/*.fits", recursive=True)
    print_box(f"Found {len(fits_files)} .fits files in {real_images_path}")

    X_test, y_test = load_pkl_file("/home4/s4683099/Deep-AGN-Clean/data/jwst_full_data/test_data.pkl")

    dataset = MockRealGalaxyDataset(
        real_images=fits_files,
        source=X_test,
        target=y_test,
        training=False
    )

    psfs_list = []
    real_gal_list = []
    for i in range(len(fits_files)):
        real_image_tensor, psf_tensor = dataset[i]
        psfs_list.append(psf_tensor.cpu().numpy())
        real_gal_list.append(real_image_tensor.cpu().numpy())

    plotter = Plotter()

    plotter.make_3d_galaxy_plot(real_gal=real_gal_list, psfs=psfs_list)

def real_image_cleaner_pipeline(
        model_names: list[str],
        model_types: list[str],
        model_filenames: list[str],
        data_folders: list[str],
        datasets: list[_BaseDataset],
        loaders: list[_BaseLoader],
        transforms: list[_BaseTransform|None],
        n: int,
        real_images_path: str|None = "/scratch/s4683099/real_JWST/COSMOS-Web_cutouts_Zhuang2024",
        plots_filename: str = "real_images",
    ) -> None:
    _check_list_lengths(
        model_names=model_names,
        model_types=model_types,
        model_filenames=model_filenames,
        data_folder=data_folders,
        dataset=datasets,
        transform=transforms,
        loader=loaders,
    )
    fits_files = glob.glob(f"{real_images_path}/*.fits", recursive=True)
    print_box(f"Found {len(fits_files)} .fits files in {real_images_path}")

    source_list = []
    target_list = []
    cleaned_images_list = []
    frf_list = []

    print("Real Data Analysis:")
    import matplotlib.pyplot as plt
    for i in range(len(model_names)):
        model_type = model_types[i]
        model_filename = model_filenames[i]
        data_folder = data_folders[i]

        transform = transforms[i] if transforms else None
        dataset = datasets[i](source=fits_files, target=fits_files, transform=transform, training=False)
        loader = loaders[i](dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

        tester = Tester(
            model_type=model_type,
            model_filename=model_filename,
            data_folder=data_folder,
            test_loader=loader,
            transform=transform,
        )

        (
            source_arr,
            target_arr,
            cleaned_image_arr,
            _,
            _
        ) = tester.clean_images(f_agn=None, n=n)

        source_list.append(source_arr)
        cleaned_images_list.append(cleaned_image_arr)
        target_list.append(target_arr)

        predicted_gal_fluxes = np.sum(cleaned_image_arr, axis=(1, 2, 3))  # (B,)
        real_gal_fluxes = np.sum(source_arr, axis=(1, 2, 3))  # (B,)
        frf = predicted_gal_fluxes / real_gal_fluxes - 1

        # if i == 0:
        #     plt.hist(real_gal_fluxes, bins=10, alpha=0.5, label="Real Galaxy Fluxes", density=True)
        #     plt.legend()
        #     plt.xlabel("Fluxes")
        #     plt.ylabel("PDF")
        #     plt.savefig(os.path.join(os.getcwd(), "real_vs_mock_flux_histogram.png"))
        #     plt.close()

        print_box(f"Model: {model_names[i]}\nMean FRF: {np.mean(frf):.4f}\nMedian FRF: {np.median(frf):.4f}\nStd FRF: {np.std(frf):.4f}")

        plt.hist(frf, bins=10, alpha=0.5, label=model_names[i], density=True)

    plt.legend()
    plt.xlabel("FCM")
    plt.ylabel("PDF")
    plt.savefig(os.path.join(os.getcwd(), "real_images_frf_histogram.png"))
    plt.close()

    plotter = Plotter()
    
    plotter.grid_plot(
        sources=source_list,
        targets=None,
        outputs=cleaned_images_list,
        titles=model_names,
        filename=plots_filename,
        data_folder=os.getcwd(),
        f_agn=None,
        save=True,
    )

# DEPRICATED: This function is not used anymore.
def mockreal_image_cleaner_pipeline(
        model_names: list[str],
        model_types: list[str],
        model_filenames: list[str],
        data_folders: list[str],
        datasets: list[_BaseDataset],
        loaders: list[_BaseLoader],
        transforms: list[_BaseTransform|None],
        n: int,
        real_images_path: str|None = "/scratch/s4683099/real_JWST/COSMOS-Web_cutouts_Zhuang2024",
        plots_filename: str = "real_images",
    ) -> None:
    _check_list_lengths(
        model_names=model_names,
        model_types=model_types,
        model_filenames=model_filenames,
        data_folder=data_folders,
        dataset=datasets,
        transform=transforms,
        loader=loaders,
    )
    from data_pipeline.galaxy_dataset import MockRealGalaxyDataset
    fits_files = glob.glob(f"{real_images_path}/*.fits", recursive=True)
    print_box(f"Found {len(fits_files)} .fits files in {real_images_path}")

    # Load the test data
    X_test, y_test = load_pkl_file("/home4/s4683099/Deep-AGN-Clean/data/jwst_full_data/test_data.pkl")

    source_list = []
    target_list = []
    cleaned_images_list = []

    for i in range(len(model_names)):
        model_type = model_types[i]
        model_filename = model_filenames[i]
        data_folder = data_folders[i]

        transform = transforms[i] if transforms else None
        #dataset = datasets[i](source=fits_files, target=fits_files, transform=transform, training=False)
        dataset = MockRealGalaxyDataset(real_images=fits_files, source=X_test, target=y_test, transform=transform, training=False)
        loader = loaders[i](dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

        tester = Tester(
            model_type=model_type,
            model_filename=model_filename,
            data_folder=data_folder,
            test_loader=loader,
            transform=transform,
        )

        (
            source_arr,
            target_arr,
            cleaned_image_arr,
            _,
            _
        ) = tester.clean_images(f_agn=[70], n=n)

        source_list.append(source_arr)
        cleaned_images_list.append(cleaned_image_arr)
        target_list.append(target_arr)

    plotter = Plotter()
    
    plotter.grid_plot(
        sources=source_list,
        targets=target_list,
        outputs=cleaned_images_list,
        titles=model_names,
        filename=plots_filename,
        data_folder=os.getcwd(),
        f_agn=None,
        save=True,
    )

def plot_pixel_hist(
        real_images_path: str|None = "/scratch/s4683099/real_JWST/COSMOS-Web_cutouts_Zhuang2024",
        test_data_path_pkl: str|None = "/home4/s4683099/Deep-AGN-Clean/data/jwst_full_data/test_data.pkl",
        plots_filename: str = "plot_dist_hist",
    ) -> None:
    from data_pipeline.galaxy_container import GalaxyContainer
    import torch

    # Get the list of .fits files in the specified directory
    # must be real images path
    fits_files = glob.glob(f"{real_images_path}/*.fits", recursive=True)
    print_box(f"Found {len(fits_files)} .fits files in {real_images_path}")

    # Load the test data
    X_test, y_test = load_pkl_file(test_data_path_pkl)

    # Create a dataset and loader for the real images
    dataset_real = GalaxyContainer(filepaths=fits_files)
    dataset_mock = GalaxyContainer(filepaths=X_test)
    dataset_mock.filter_by_f_agn_list(f_agn_list=[10, 30, 44, 65, 70, 90], n=1360)  # Filter to match the real images
    
    # Extract the real and mock data
    real_data = []
    mock_data = []
    for i in tqdm(range(len(dataset_real))):
        real_data.append(dataset_real[i])

    for i in tqdm(range(len(dataset_mock))):
        mock_data.append(dataset_mock[i])

    real_data = torch.stack(real_data, dim=0)
    mock_data = torch.stack(mock_data, dim=0)

    # Flatten the data for histogram plotting
    real_data_flat = real_data.flatten().numpy()
    mock_data_flat = mock_data.flatten().numpy()

    # Create a Plotter instance
    plotter = Plotter()

    # Plot the histograms for real and mock data
    plotter.plot_two_histograms(
        arr1=real_data_flat,
        arr2=mock_data_flat,
        label1="Real Images",
        label2="Mock Images",
        title="Pixel Value Distribution",
        filename=plots_filename,
        data_folder=os.getcwd(),
        save=True,
    )

def performance_analysis_pipeline(
        data_folder: str,
        model_name: str,
        pa_filename: str,
        verbose: bool = True,
        latex: bool = False,
    ):
    filepath = os.path.join(data_folder, pa_filename)

    padict: PAdict = load_pkl_file(filepath)

    plotter = Plotter()

    real_psf_fluxes, predicted_psf_fluxes, real_gal_fluxes, predicted_gal_fluxes = padict.get_all_fluxes_np()

    interpeter = ResultInterpreter()

    interpeter.interpret_performance_analyis(
        data=padict,
        model_name=model_name,
        verbose=verbose,
        latex=latex,
    )

    df_psf, df_gal = interpeter.frf_flux_correlation(
        data=padict,
        model_name=model_name,
        verbose=verbose,
        latex=latex,
    )

    interpeter.compare_model_performance(
        data_list=[padict],
        verbose=verbose,
        latex=latex,
    )

    plotter.make_2d_histogram(
        real_fluxes=real_psf_fluxes,
        predicted_fluxes=predicted_psf_fluxes,
        histogram_filename="psf_flux_hist",
        data_folder=data_folder,
        x_label="Real AGN Flux (Jy)",
        y_label="Predicted AGN Flux (Jy)",
        title=f"Real vs Predicted PSF Flux for {model_name}",
    )

    plotter.make_2d_histogram(
        real_fluxes=real_gal_fluxes,
        predicted_fluxes=predicted_gal_fluxes,
        histogram_filename="gal_flux_hist",
        data_folder=data_folder,
        x_label="Real Galaxy Flux (Jy)",
        x_y_lim=40000,
        y_label="Predicted Galaxy Flux (Jy)",
        title=f"Real vs Predicted Galaxy Flux for {model_name}",
    )

    plotter.make_frf_flux_correlation_plot(
        df_psf_list=[df_psf],
        df_gal_list=[df_gal],
        data_folder=data_folder,
        model_names=[model_name],
        filename=model_name.lower().replace(" ", "_"),
    )

    plotter.make_trend_plots_fagn(pa_list=[padict], model_names=[model_name], data_folder=data_folder)

    plotter.flux_histogram(
        fluxes=real_psf_fluxes,
        title="Real AGN Flux Histogram",
        data_folder=os.path.join(data_folder, "histograms"),
        filename="agn_flux_histogram",
    )

    plotter.flux_histogram(
        fluxes=real_gal_fluxes,
        title="Real Galaxy Flux Histogram",
        data_folder=os.path.join(data_folder, "histograms"),
        filename="gal_flux_histogram",
    )

def _check_list_lengths(
        model_names: list[str],
        model_types: list[str],
        model_filenames: list[str],
        data_folder: list[str],
        dataset: list[_BaseDataset],
        transform: list[_BaseTransform|None],
        loader: list[_BaseLoader]
    ) -> None:
    if not all(len(lst) == len(model_names) for lst in [model_types, model_filenames, data_folder, dataset, transform, loader]):
        raise ValueError("All lists must have the same length.")
    

def compare_performance_analysis_pipeline(
        data_folders: list[str],
        model_names: list[str],
        pa_filenames: list[str],
        data_folder: str = os.getcwd(),
        verbose: bool = True,
        latex: bool = False,
    ) -> None:
    if len(data_folders) != len(model_names) or len(data_folders) != len(pa_filenames):
        raise ValueError("data_folders, model_names, and filenames must have the same length.")
    
    interpeter = ResultInterpreter()
    
    df_psf_list = []
    df_gal_list = []
    pa_list = []
    for i in range(len(data_folders)):
        filepath = os.path.join(data_folders[i], pa_filenames[i])
        pa = load_pkl_file(filepath)

        df_psf, df_gal = interpeter.frf_flux_correlation(
            data=pa,
            model_name=model_names[i],
            verbose=False,
            latex=False,
        )

        pa_list.append(pa)
        df_psf_list.append(df_psf)
        df_gal_list.append(df_gal)

    interpeter.compare_model_performance(
        data_list=pa_list,
        verbose=verbose,
        latex=latex,
    )

    plotter = Plotter()

    plotter.make_trend_plots_fagn(pa_list=pa_list, model_names=model_names, data_folder=data_folder)
    
    # DEPRICATED: This function is not used anymore.
    # plotter.make_frf_flux_correlation_plot(
    #     df_psf_list=df_psf_list,
    #     df_gal_list=df_gal_list,
    #     data_folder=data_folder,
    #     model_names=model_names,
    #     filename="comparison",
    # )
