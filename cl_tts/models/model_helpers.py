from .tacotron2_nv.tacotron2_nv import get_tacotron2_nv
from .tacotron2_nv.tacotron2_loss import Tacotron2Loss


def get_model(params, n_symbols, n_speakers, device):
    # -----> Tacotron2NV
    if params["model_name"] == "Tacotron2NV":
        model, forward_func, criterion_func = get_tacotron2_nv(
            params=params,
            n_symbols=n_symbols,
            n_speakers=n_speakers,
            device=device
        )

    else:
        raise NotImplementedError()

    return model, forward_func, criterion_func
