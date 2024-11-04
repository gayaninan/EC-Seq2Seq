import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from fine_tuning.training_pipeline import run_training_pipeline

def main():
    parser = argparse.ArgumentParser(description='Run fine-tuning pipeline with specified model and objective.')
    parser.add_argument('--model_type', type=str, required=True, choices=['bart', 't5'], help='Type of model to use (bart or t5).')
    parser.add_argument('--model_size', type=str, required=True, choices=['small', 'base', 'large'], help='Size of the model (small, base, or large).')
    parser.add_argument('--objective', type=str, default='summarisation', help='Training objective.')

    args = parser.parse_args()

    run_training_pipeline(model_type=args.model_type, model_size=args.model_size, objective=args.objective)

if __name__ == "__main__":
    main()
