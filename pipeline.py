from train.train_mlp import train_mlp


def evaluate_pipeline(trial_id=None, **config):
    return train_mlp(
        trial_id=trial_id,
        **config
    )