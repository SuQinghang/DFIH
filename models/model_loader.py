import torch.nn as nn

import models.alexnet as alexnet


def load_model(arch, code_length):
    """
    Load CNN model.

    Args
        arch(str): CNN model name.
        code_length(int): Hash code length.

    Returns
        model(torch.nn.Module): CNN model.
    """
    if arch == 'alexnet':
        model = alexnet.load_model( code_length)
    else:
        raise ValueError('Invalid cnn model name!')

    return model

