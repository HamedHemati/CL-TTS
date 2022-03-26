from .naive import Naive


def get_strategy(
        params,
        model,
        optimizer,
        forward_func,
        criterion_func,
        evaluation_plugin,
        device
):
    """
    Initializes strategy for a given model.

    :param params:
    :param model:
    :param optimizer:
    :param forward_func:
    :param criterion_func:
    :param collator:
    :param evaluation_plugin:
    :param device:
    :return:
    """

    strategy = None
    if params["strategy"] == "naive":
        strategy = Naive(
            model=model,
            optimizer=optimizer,
            params=params,
            forward_func=forward_func,
            criterion_func=criterion_func,
            num_workers=params["num_workers"],
            device=device,
            evaluator=evaluation_plugin,
        )
    else:
        raise NotImplementedError

    return strategy
