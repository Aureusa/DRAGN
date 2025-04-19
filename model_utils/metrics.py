import torch
import torch.nn as nn
# from typing import Any
# from abc import ABC, abstractmethod

from model_utils.loss_functions import get_loss_function, Loss
from utils import print_box
#from ignite.metrics import PSNR


# TODO: Implement metrics for evaluation
# 1: PSNR metric
def get_metrics(metrics: list[str]) -> None:
    metrics_dict = {
            #"PSNR": PSNRmetric()# Add other metrics here if needed
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

# class PSNRmetric(nn.Module, Loss):
#     def __init__(self):
#         super(PSNRmetric, self).__init__()

#     def __str__(self):
#         return "PSNR Metric"

#     def forward(self, x, y_pred, y_true, psf):
#         min_, max_ = y_true.min(), y_true.max()

#         metric = PSNR(data_range=max_ - min_)
#         metric.update(y_pred, y_true)
#         psnr_value = metric.compute()
#         metric.reset()
#         return psnr_value
    