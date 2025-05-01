from model_utils.performance_analysis import PAdict
import os
from utils import load_pkl_file


if __name__ == "__main__":
    # Load the performance analysis data
    pa_data = load_pkl_file(os.path.join(os.getcwd(), "data", "performance_analysis", "generator_tripple_cgan_best_model_performance_analysis.pkl"))
    
    # Create a PAdict instance
    pa_dict = PAdict(model_name="Model1")
    
    # Set the data
    pa_dict.data = pa_data
    
    # Print the PAdict instance
    df = pa_dict.get_df(True)
    print(df)
    
    # Check if two PAdict instances are equal
    pa_dict2 = PAdict(model_name="Model2")
    pa_dict2.data = pa_data
    print(pa_dict == pa_dict2)

    print(pa_dict.model_name)
    print(pa_dict2.model_name)

    pa_dict3 = PAdict(model_name="Model3")

    eval_dic = {
        "PSNR": 1,
        "SSIM": 2,
        "LPIPS": 3
    }
    count = 10
    agn_fraction = "f_agn = 0.5"

    pa_dict3.add(agn_fraction, count, eval_dic)
    pa_dict3.add(agn_fraction, count+10, eval_dic)

    print(pa_dict3)
