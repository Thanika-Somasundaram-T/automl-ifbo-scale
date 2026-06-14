

from pathlib import Path

from train.predict import predict


def neps_wrapper(df, device):
    
    def evaluate_pipeline(trial_id=None, **config):
        return predict(
            df=df,
            device=device,
            **config
        )

    return evaluate_pipeline
    