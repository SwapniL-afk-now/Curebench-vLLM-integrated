// filepath: README.md
# CUREBench

CUREBench is a framework for evaluating AI models on clinical and pharmaceutical knowledge.

## Features

- Support for multiple model types (ChatGPT, Local, vLLM)
- Incremental evaluation with resume capability
- Automatic answer parsing with robust \boxed{} format handling
- CSV output with detailed predictions and reasoning traces
- Submission generation with metadata

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/CUREBench.git
cd CUREBench
pip install -r requirements.txt
```

## Usage

```bash
python eval_framework.py --config config.json --model your_model_name --dataset dataset_name
```

## Configuration

Create a `config.json` file with your dataset and model configurations:

```json
{
    "output_dir": "results",
    "dataset": {
        "dataset_name": "treatment",
        "dataset_path": "path/to/dataset.jsonl"
    },
    "metadata": {
        "model_name": "your_model",
        "model_type": "chatgpt"
    }
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.