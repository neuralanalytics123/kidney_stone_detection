import argparse
from loguru import logger
from kidney_ai.training.trainer import train_pipeline
from kidney_ai.testing.evaluator import run_clinical_test_suite


def main():
    parser = argparse.ArgumentParser(description="Run End-to-End Kidney Stone AI Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    args = parser.parse_args()

    logger.info("=== Phase 1-3: Preprocessing, Feature Engineering & Training ===")
    train_pipeline(config_path=args.config, epochs_override=args.epochs)

    logger.info("=== Phase 4: Clinical Hold-Out Test Suite ===")
    run_clinical_test_suite(config_path=args.config)


if __name__ == "__main__":
    main()
