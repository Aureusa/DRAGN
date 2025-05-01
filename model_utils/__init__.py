from model_utils.model_testing import ModelTester
from model_utils.model_training import ModelTrainer
from model_utils.loss_functions import _get_avaliable_loss_funcstions
from model_utils.plotter import Plotter


AVALIABLE_LOSS_FUNCTIONS = list(_get_avaliable_loss_funcstions().keys())


def clean_n_image_with_multiple_models(
        model_names: list[str],
        model_types: list[str],
        model_filenames: list[str],
        data_folder: str,
        n: int,
        filename: str = "test_image"
    ) -> None:
    sources_list = []
    targets_list = []
    cleaned_images_list = []
    diffs_list = []
    psfs_list = []
    for i in range(len(model_names)):
        tester = ModelTester(
            model_name=model_names[i],
            model_type=model_types[i],
            model_filename=model_filenames[i],
            data_folder=data_folder
        )

        source_arr, target_arr, cleaned_image_arr, diff_predicted_arr, psf_arr, norm_list = tester.clean_n_images(n=n)

        sources_list.append(source_arr)
        targets_list.append(target_arr)
        cleaned_images_list.append(cleaned_image_arr)
        diffs_list.append(diff_predicted_arr)
        psfs_list.append(psf_arr)

    plotter = Plotter()

    plotter.plot_cleaned_images(
        sources=sources_list,
        targets=targets_list,
        cleaned_images=cleaned_images_list,
        diffs=diffs_list,
        psfs=psfs_list,
        titles=model_names,
        norms=norm_list,
        filename=filename
    )
    