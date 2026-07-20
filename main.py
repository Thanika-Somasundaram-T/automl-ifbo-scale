import yaml
import neps
from train.train_mlp import train_mlp


def evaluate_pipeline(trial_id=None, **config):
    return train_mlp(trial_id=trial_id, **config)


def main():

    pipeline_space = {
        "lr": neps.HPOCategorical(choices=[3e-3, 1e-3, 3e-4, 1e-4]),
        "weight_decay": neps.HPOCategorical(choices=[0.0, 1e-3]),
        "hidden_dim": neps.HPOCategorical(choices=[24, 32, 64, 128, 256]),
        "epochs": neps.HPOConstant(value=100),
        "num_layers": neps.HPOConstant(value=4),
        "lr_schedule": neps.HPOConstant(value="none"),
        "batch_size": neps.HPOConstant(value=64),
    }

    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=pipeline_space,
        optimizer='grid_search',
    )


if __name__ == "__main__":
    main()