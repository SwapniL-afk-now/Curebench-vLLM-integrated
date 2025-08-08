# !git clone https://github.com/thisisAranya/CUREBench.git
# %cd /kaggle/working/CUREBench
# !pip install -r requirements.txt
# !pip install bitsandbytes

# import json

# config_data = {
#     "metadata": {
#         "model_name": "Qwen/Qwen3-4B-Thinking-2507",  # Change this to your vLLM model name
#         "model_type": "vllm",  # Changed from "LocalModel" to "vllm"
#         "track": "internal_reasoning",
#         "base_model_type": "Open weight model",
#         "base_model_name": "microsoft/MediPhi-Instruct",
#         "dataset": "cure_bench_phase_1",
#         "additional_info": "Submission using configuration file with vLLM and resume functionality"
#     },
#     "dataset": {
#         "dataset_name": "cure_bench_phase_1",
#         "dataset_path": "/kaggle/input/cure-bench-splitted/curebench_first_half.jsonl",
#         "description": "CureBench 2025 test questions"
#     },
#     "output_dir": "results"  # Added output directory
# }

# with open('/kaggle/working/CUREBench/metadata_config_test.json', 'w') as f:
#    json.dump(config_data, f, indent=4)

# print("Configuration saved to metadata_config_test.json")

# # Run with configuration file (recommended)
# # The modified framework will automatically handle incremental CSV writing and resume functionality
# !python run.py --config '/kaggle/working/CUREBench/metadata_config_test.json'