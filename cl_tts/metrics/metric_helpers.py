from avalanche.evaluation.metrics import forgetting_metrics, \
    accuracy_metrics, loss_metrics


def get_metrics(metric_names):
    metric_list = []
    for metric_name in metric_names:
        if metric_name == "accuracy":
            metric_list.append(accuracy_metrics(minibatch=True, epoch=True,
                                                experience=True, stream=False))
        elif metric_name == "loss":
            metric_list.append(loss_metrics(minibatch=True, epoch=True,
                                            experience=True, stream=False))
        elif metric_name == "forgetting":
            metric_list.append(forgetting_metrics(experience=True,
                                                  stream=True))
        else:
            raise NotImplementedError()
    return metric_list
