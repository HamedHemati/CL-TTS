from .tacotron2_nv.tacotron2_nv import get_tacotron2_nv
from .tacotron2_nv.tacotron2_loss import Tacotron2Loss


def get_model(params, n_symbols, n_speakers, device):
    # -----> Tacotron2NV
    if params["model"] == "Tacotron2NV":
        model = get_tacotron2_nv(params=params, n_symbols=n_symbols,
                                 n_speakers=n_speakers)
        criterion = Tacotron2Loss(params["n_frames_per_step"],
                                  params["reduction"],
                                  params["pos_weight"],
                                  device)

    else:
        raise NotImplementedError()

    return model, criterion
