

from pathlib import Path

from train.predict import predict
from train.train_mlp import train_mlp


def neps_wrapper(neps_path=Path):
    
    def evaluate_pipleine(trial_id=None, **config):
        return predict(trial_id=trial_id, **config)
    
    return evaluate_pipleine
    