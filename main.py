import torch
import argparse
import yaml
from pipeline import neps_wrapper
import neps

def parsey():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--num_layers", type=int,
        default=2,
        choices=[2, 4],
        help="Number of layers for the MLP (2 or 4)"
    )
    return parser.parse_args()

def main(num_layers):
    with open("./neps_config_4.yaml", "r") as f:
        neps_config = yaml.safe_load(f)
        
    neps_config["evaluate_pipeline"] = neps_wrapper()
    # neps_config["num_layers"] = num_layers
    
    neps.run(**neps_config)
    

if __name__ == "__main__":
    args = parsey()
    main(num_layers=args.num_layers)
