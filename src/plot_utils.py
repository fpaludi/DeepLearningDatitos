from typing import Iterator, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch import Tensor
from torch.nn.parameter import Parameter
import pytorch_lightning as pl


def plot_confusion_matrix(
    preds: Tensor,
    targets: Tensor,
    n_classes: int
):
    confusion_matrix = pl.metrics.functional.confusion_matrix(
        preds,
        targets,
        num_classes=n_classes,
        normalize="true",
    )
    df_cm = pd.DataFrame(
        confusion_matrix.numpy(),
        index=range(n_classes),
        columns=range(n_classes)
    )
    plt.figure(figsize = (10,7))
    fig = sns.heatmap(
        df_cm,
        annot=True,
        cmap='Blues',
    )
    fig.set(
        xlabel="Predictions",
        ylabel="Targets"
    )
    fig = fig.get_figure()
    plt.close(fig)
    return fig

def plot_grad_flow(named_parameters: Iterator[Tuple[str, Parameter]]):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    fig = plt.figure()
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k", label="Zero Gradient")
    plt.plot(
        np.arange(len(max_grads)),
        max_grads,
        lw=1,
        color="r",
        marker="o",
        label="Max Gradient"
    )
    plt.plot(
        np.arange(len(max_grads)),
        ave_grads,
        lw=1,
        color="b",
        marker="x",
        label="Average Gradient"
    )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation=30)
    plt.xlim(left=0, right=len(ave_grads))
    #plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend()
    plt.close(fig)
    return fig