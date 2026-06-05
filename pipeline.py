

from pathlib import Path

from train.predict import predict
from train.train_mlp import train_mlp


def neps_wrapper(df, device, global_min, global_max):
    
    def evaluate_pipeline(trial_id=None, **config):
        return predict(
            df=df,
            device=device,
            global_min=float(global_min),
            global_max=float(global_max),
            **config
        )

    return evaluate_pipeline
    