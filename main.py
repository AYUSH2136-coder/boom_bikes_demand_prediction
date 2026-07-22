import argparse
import sys
import os
import pandas as pd
import numpy as np
from src.training.train import train_pipeline
from src.inference.predict import predict
from src.utils.logger import get_logger
from src.utils.helpers import load_yaml

logger = get_logger("main")

def main():
    parser = argparse.ArgumentParser(
        description="BoomBikes Demand Prediction Pipeline Orchestration CLI",
        usage="python main.py <command> [<args>]"
    )
    parser.add_argument("command", choices=["train", "predict"], help="Pipeline stage to execute: 'train' or 'predict'.")
    
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
        
    command = sys.argv[1]
    
    if command == "train":
        train_parser = argparse.ArgumentParser(description="Run the model training and tuning pipeline.")
        train_parser.add_argument("--config-dir", type=str, default="configs", help="Path to config folder.")
        train_parser.add_argument("--no-comparison", action="store_true", help="Skip baseline model comparison.")
        args = train_parser.parse_args(sys.argv[2:])
        
        logger.info("Executing training subcommand...")
        train_pipeline(config_dir=args.config_dir, run_comparison=not args.no_comparison)
        
    elif command == "predict":
        predict_parser = argparse.ArgumentParser(description="Run batch or single-row inference.")
        predict_parser.add_argument("--input", type=str, required=True, help="Path to input CSV file.")
        predict_parser.add_argument("--output", type=str, required=True, help="Path to save predictions CSV.")
        predict_parser.add_argument("--model", type=str, default="models/tuned_model.pkl", help="Path to model file.")
        predict_parser.add_argument("--scaler", type=str, default="models/scaler.pkl", help="Path to scaler file.")
        predict_parser.add_argument("--config-dir", type=str, default="configs", help="Path to config folder.")
        args = predict_parser.parse_args(sys.argv[2:])
        
        logger.info("Executing prediction subcommand...")
        df_input = pd.read_csv(args.input)
        preds = predict(
            input_df=df_input,
            model_path=args.model,
            scaler_path=args.scaler,
            config_dir=args.config_dir
        )
        
        df_output = df_input.copy()
        cfg = load_yaml(os.path.join(args.config_dir, "config.yaml"))
        target = cfg["project"]["target"]
        df_output[f"predicted_{target}"] = np.round(preds).astype(int)
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df_output.to_csv(args.output, index=False)
        logger.info(f"Predictions successfully saved to: {args.output}")

if __name__ == "__main__":
    main()
