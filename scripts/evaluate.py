import os
import sys
import argparse
PROJECT_DIR = r"c:\Users\Nisarga N\OneDrive\Documents\My Projects\Healthcare-AI-Capstone"
sys.path.append(PROJECT_DIR)

from src.evaluation.evaluator import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Medical Image Classifier Evaluation Script")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=os.path.join(PROJECT_DIR, "models", "resnet50_run", "best_model.pth"),
        help="Path to saved .pth model checkpoint"
    )
    args = parser.parse_args()
    
    # Define splits and results directories
    dataset_dir = os.path.join(PROJECT_DIR, "dataset")
    val_dir = os.path.join(dataset_dir, "validation")
    test_dir = os.path.join(dataset_dir, "test")
    results_dir = os.path.join(PROJECT_DIR, "results")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    evaluate_model(
        checkpoint_path=args.checkpoint,
        val_dir=val_dir,
        test_dir=test_dir,
        results_dir=results_dir,
        device=device
    )

if __name__ == "__main__":
    import torch
    main()
