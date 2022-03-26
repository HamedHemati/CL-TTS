import os

from TTS.tts.configs.tacotron2_config import Tacotron2Config


def get_model_config(params, ds_path):
    if params["model_name"] == "Tacotron2":
        model_config = Tacotron2Config(
            text_cleaner="phoneme_cleaners",
            use_phonemes=params["use_phonemes"],
            phoneme_language=params["phoneme_language"],
            phoneme_cache_path=os.path.join(ds_path, "phonemes"),
            use_d_vector_file=params["model"]["use_d_vector_file"],
            d_vector_dim=params["model"]["d_vector_dim"],
        )

    else:
        raise NotImplementedError()

    return model_config
