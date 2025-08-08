import os
import argparse
import json
from eval_framework import CompetitionKit

def main():
    parser = argparse.ArgumentParser(description="Bio-Medical AI Competition Framework")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--dataset", help="Dataset name to evaluate")
    parser.add_argument("--model", help="Model name/path")
    parser.add_argument("--model-type", default="auto", help="Model type (auto, chatgpt, local, vllm, custom)")
    parser.add_argument("--max-examples", type=int, help="Maximum examples to evaluate")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🏥 CURE-Bench Competition - Evaluation")
    print("="*60)
    
    # Initialize the competition kit
    kit = CompetitionKit(config_path=args.config)
    
    # Load model
    model_name = args.model or kit.config.get("metadata", {}).get("model_name")
    model_type = args.model_type
    
    if kit.config.get("metadata", {}).get("model_type"):
        model_type = kit.config["metadata"]["model_type"]
    
    if not model_name:
        raise ValueError("Model name must be specified in config or command line")
    
    kit.load_model(model_name, model_type.lower())
    
    # Show available datasets
    print("Available datasets:")
    for ds_name in kit.datasets.keys():
        print(f"- {ds_name}")
    
    # Get dataset name
    dataset_name = args.dataset or list(kit.datasets.keys())[0]
    
    # Run evaluation
    print(f"Running evaluation on dataset: {dataset_name}")
    
    results = kit.evaluate(
        dataset_name=dataset_name,
        output_file=args.output,
        max_examples=args.max_examples
    )
    
    # Save submission
    print("Generating submission with metadata...")
    submission_path = kit.save_submission_with_metadata(results)
    
    print(f"\n✅ Evaluation completed successfully!")
    print(f"📊 Accuracy: {results.accuracy:.2%} ({results.correct_predictions}/{results.total_examples})")
    print(f"📄 Submission saved to: {submission_path}")
    
    # Show metadata summary if verbose
    final_metadata = kit.config.get("metadata", {})
    print("\n📋 Final metadata:")
    for key, value in final_metadata.items():
        print(f"  {key}: {value}")
            


if __name__ == "__main__":
    main()