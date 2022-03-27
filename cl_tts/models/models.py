from TTS.tts.models.tacotron2 import Tacotron2


def get_model(
        params,
        config,
        ap,
        tokenizer,
        speaker_manager,
):
    if params["model"] == "tacotron2":
        model = Tacotron2(
            config,
            ap,
            tokenizer,
            speaker_manager=speaker_manager
        )
    else:
        raise NotImplementedError()

    return model
