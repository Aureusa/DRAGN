# import torch
# import torch.nn as nn
# from typing import Any
# from abc import ABC, abstractmethod

from model_utils.loss_functions import get_loss_function
from utils import print_box


# TODO: Implement metrics for evaluation
def get_metrics(metrics: list[str]) -> None:
    metrics_dict = {
            # Add other metrics here if needed
        }
    
    metrics_list = []

    info = "Retrieved metrics:\n"
    for m in metrics:
        try:
            metric = get_loss_function(m)
            metrics_list.append(metric)
            info += f" - {m}\n"
        except Exception as e:
            if m not in metrics_dict:
                continue
            else:
                metric = metrics_dict[m]
                metrics_list.append(metric)
                info += f" - {m}\n"

    print_box(info)

    return metrics_list
