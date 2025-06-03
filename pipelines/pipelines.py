from model_utils.model_testing import ModelTester
from model_utils.model_training import ModelTrainer
from model_utils.loss_functions import _get_avaliable_loss_funcstions
from model_utils.plotter import Plotter
from model_utils.result_interpreter import ResultInterpreter

from data_pipeline import (
    GalaxyDataset,
    FitsLoader,
    _BaseTransform
)

from loggers_utils import TrainingLogger
from utils import load_pkl_file

import os


AVALIABLE_LOSS_FUNCTIONS = list(_get_avaliable_loss_funcstions().keys())


def training_pipeline(
        model_type,
        model_name,
        telescope,
        data_folder,
        loss,
        batch_size,
        prefetch_factor,
        num_workers,
        lr,
        **model_kwargs
    ) -> None:
    trainer = ModelTrainer(
        model_type,
        model_name,
        telescope,
        data_folder,
        **model_kwargs,
    )

    X_train, X_val, _, y_train, y_val, _ = trainer.load_data("jwst_full_data")

    train_loader, val_loader = trainer.forge_loaders(
        X_train,
        X_val,
        y_train,
        y_val,
        dataset=GalaxyDataset,
        loader=FitsLoader,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor
    )

    trainer.train_model(
        train_loader,
        val_loader,
        loss_name=loss,
        lr=lr
    )


def finetunning_pipeline(
        model_type,
        model_name,
        telescope,
        data_folder,
        loss,
        batch_size,
        prefetch_factor,
        num_workers,
        lr,
        loading_params,
        new_folder: str = "",
        **model_kwargs
    ) -> None:
    trainer = ModelTrainer(
        model_type=model_type,
        model_name=model_name,
        telescope=telescope,
        data_folder=data_folder,
        **model_kwargs,
    )

    X_train, X_val, _, y_train, y_val, _ = trainer.load_data("jwst_full_data")

    train_loader, val_loader = trainer.forge_loaders(
        X_train,
        X_val,
        y_train,
        y_val,
        dataset=GalaxyDataset,
        loader=FitsLoader,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor
    )

    if new_folder:
        trainer.model.load_model(dir_ = os.path.join("data", data_folder), **loading_params)
        trainer.data_folder = os.path.join("data", new_folder)

        trainer.train_model(
            train_loader,
            val_loader,
            loss_name=loss,
            lr=lr
        )
    else:
        trainer.fine_tune_model(
            train_loader,
            val_loader,
            loss_name=loss,
            lr=lr,
            **loading_params
        )

def image_cleaner_pipeline(
        model_names: list[str],
        model_types: list[str],
        model_filenames: list[str],
        data_folder: list[str],
        dataset: list[GalaxyDataset],
        transform: list[_BaseTransform|None],
        loader: list[FitsLoader],
        deep: list[bool],
        n: int,
        filename: str = "test_image",
        show_real_min_infered: bool = False,
        f_agn: int|None = None,
        report: bool = False,
    ) -> None:
    _check_list_lengths(
        model_names=model_names,
        model_types=model_types,
        model_filenames=model_filenames,
        data_folder=data_folder,
        dataset=dataset,
        transform=transform,
        loader=loader,
        deep=deep
    )
    sources_list, targets_list, cleaned_images_list, diffs_list, psfs_list = _clean_images(
        model_names=model_names,
        model_types=model_types,
        model_filenames=model_filenames,
        data_folder=data_folder,
        dataset=dataset,
        transform=transform,
        loader=loader,
        deep=deep,
        n=n,
        f_agn=f_agn
    )

    plotter = Plotter()
    
    if report:
        plotter.plot_cleaned_images_report(
            sources=sources_list,
            targets=targets_list,
            outputs=cleaned_images_list,
            titles=model_names,
            filename=filename,
            f_agn=f_agn,
        )
    else:
        plotter.plot_cleaned_images(
            sources=sources_list,
            targets=targets_list,
            outputs=cleaned_images_list,
            diffs=diffs_list,
            psfs=psfs_list,
            titles=model_names,
            filename=filename,
            show_real_min_infered=show_real_min_infered
        )

def psf_histogram_pipeline(
        model_names: list[str],
        model_types: list[str],
        model_filenames: list[str],
        data_folder: str,
        dataset: list[GalaxyDataset],
        transform: list[_BaseTransform|None],
        loader: list[FitsLoader],
        deep: list[bool],
        n: int,
        filename: str = "test_image",
        test_set: bool = True
    ) -> None:
    _check_list_lengths(
        model_names=model_names,
        model_types=model_types,
        model_filenames=model_filenames,
        data_folder=data_folder,
        dataset=dataset,
        transform=transform,
        loader=loader,
        deep=deep
    )
    _, _, _, diffs_list, psfs_list = _clean_images(
        model_names=model_names,
        model_types=model_types,
        model_filenames=model_filenames,
        data_folder=data_folder,
        dataset=dataset,
        transform=transform,
        loader=loader,
        deep=deep,
        n=n,
        test_set=test_set
    )

    plotter = Plotter()

    plotter.plot_psf_hist(
        psfs=psfs_list,
        infered_psfs=diffs_list,
        titles=model_names,
        filename=filename
    )
    
def psf_plot_3d_pipeline(
        model_names: list[str],
        model_types: list[str],
        model_filenames: list[str],
        data_folder: str,
        dataset: list[GalaxyDataset],
        transform: list[_BaseTransform|None],
        loader: list[FitsLoader],
        deep: list[bool],
        n: int,
        filename: str = "test_image",
        show_real_min_infered: bool = False,
        test_set: bool = True
    ) -> None:
    _check_list_lengths(
        model_names=model_names,
        model_types=model_types,
        model_filenames=model_filenames,
        data_folder=data_folder,
        dataset=dataset,
        transform=transform,
        loader=loader,
        deep=deep
    )
    _, _, _, diffs_list, psfs_list = _clean_images(
        model_names=model_names,
        model_types=model_types,
        model_filenames=model_filenames,
        data_folder=data_folder,
        dataset=dataset,
        transform=transform,
        loader=loader,
        deep=deep,
        n=n,
        test_set=test_set
    )

    plotter = Plotter()

    plotter.plot_psf_3d(
        psfs=psfs_list,
        infered_psfs=diffs_list,
        titles=model_names,
        filename=filename,
        show_real_min_infered=show_real_min_infered
    )

def plot_loss_pipeline(path: str, model_name: str, discriminator: bool = False) -> None:
    logger = TrainingLogger(
        save_dir=path,
        adverserial_logger=discriminator
    )

    history = logger.history

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    best_val_loss = history["best_val_loss"]

    if discriminator:
        train_loss = history["train_loss_D"]
        val_loss = history["val_loss_D"]

    plotter = Plotter()

    plotter.plot_loss(
        train_loss=train_loss,
        val_loss=val_loss,
        best_val_loss=best_val_loss,
        model_name=model_name,
        filepath=path
    )


def compare_performance_analysis_pipeline(
        data_folder: list[str],
        model_names: list[str],
        filenames: list[str],
        latex: bool = False,
    ) -> None:
    if len(data_folder) != len(model_names) or len(data_folder) != len(filenames):
        raise ValueError("data_folder, model_names, and filenames must have the same length.")
    
    pa_list = []
    for i in range(len(data_folder)):
        filepath = os.path.join(data_folder[i], filenames[i])

        pa = load_pkl_file(filepath)
        pa_list.append(pa)
        
    interpeter = ResultInterpreter()

    interpeter.compare_model_performance(
        data_list=pa_list,
        verbose=True,
        latex=latex,
    )

    plotter = Plotter()

    plotter.make_trend_plots(data_list=pa_list, filepath=os.getcwd())


def performance_analysis_pipeline(
        data_folder: str,
        model_filename: str,
        model_name: str,
        filename: str,
        latex: bool = False
    ):
    filepath = os.path.join(data_folder, filename)

    pa = load_pkl_file(filepath)

    plotter = Plotter()

    real_psf_fluxes, predicted_psf_fluxes, real_gal_fluxes, predicted_gal_fluxes = pa.get_all_fluxes_np()

    flux_data = pa.get_flux_data()

    interpeter = ResultInterpreter()

    interpeter.interpret_performance_analyis(
        data=pa,
        model_name=model_filename,
        verbose=True,
        latex=latex
    )

    plotter.make_2d_histogram(
        real_fluxes=real_psf_fluxes,
        predicted_fluxes=predicted_psf_fluxes,
        histogram_filename="psf_flux_hist",
        histogram_datafolder=data_folder,
        title=f"Real vs Predicted PSF Flux for {model_name}",
    )

    plotter.make_2d_histogram(
        real_fluxes=real_gal_fluxes,
        predicted_fluxes=predicted_gal_fluxes,
        histogram_filename="gal_flux_hist",
        histogram_datafolder=data_folder,
        x_label="Real Galaxy Flux",
        y_label="Predicted Galaxy Flux",
        title=f"Real vs Predicted Galaxy Flux for {model_name}",
        gal=True
    )

    plotter.make_trend_plots(data_list=[pa], filepath=data_folder)

def _clean_images(
        model_names: list[str],
        model_types: list[str],
        model_filenames: list[str],
        data_folder: list[str],
        dataset: list[GalaxyDataset],
        transform: list[_BaseTransform|None],
        loader: list[FitsLoader],
        deep: list[bool],
        n: int,
        test_set: bool = True,
        f_agn: int|None = None
    ):
    sources_list = []
    targets_list = []
    cleaned_images_list = []
    diffs_list = []
    psfs_list = []
    for i in range(len(model_names)):
        if deep[i]:
            tester = ModelTester(
                model_name=model_names[i],
                model_type=model_types[i],
                model_filename=model_filenames[i],
                data_folder=data_folder[i],
                dataset=dataset[i],
                transform=transform[i],
                loader=loader[i],
                channels=(32, 64, 128, 256),
            )
        else:
            tester = ModelTester(
                model_name=model_names[i],
                model_type=model_types[i],
                model_filename=model_filenames[i],
                data_folder=data_folder[i],
                dataset=dataset[i],
                transform=transform[i],
                loader=loader[i],
            )

        source_arr, target_arr, cleaned_image_arr, diff_predicted_arr, psf_arr = tester.clean_n_images(n=n, test_set=test_set, f_agn=f_agn)

        sources_list.append(source_arr)
        targets_list.append(target_arr)
        cleaned_images_list.append(cleaned_image_arr)
        diffs_list.append(diff_predicted_arr)
        psfs_list.append(psf_arr)

    return sources_list, targets_list, cleaned_images_list, diffs_list, psfs_list

def _check_list_lengths(
        model_names: list[str],
        model_types: list[str],
        model_filenames: list[str],
        data_folder: list[str],
        dataset: list[GalaxyDataset],
        transform: list[_BaseTransform|None],
        loader: list[FitsLoader],
        deep: list[bool]
    ) -> None:
    if not all(len(lst) == len(model_names) for lst in [model_types, model_filenames, data_folder, dataset, transform, loader, deep]):
        raise ValueError("All lists must have the same length.")
    