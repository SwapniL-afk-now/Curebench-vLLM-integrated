import json
import os
import sys
import logging
import argparse
import csv
import zipfile
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from abc import ABC, abstractmethod

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Simple container for evaluation results"""
    dataset_name: str
    model_name: str
    accuracy: float
    correct_predictions: int
    total_examples: int
    predictions: List[Dict]
    reasoning_traces: List[str] = None
    details: Optional[Dict] = None


# --- SYSTEM PROMPT (Unified for all models) ---
# This prompt instructs the model to use the <thinking> and <answer> tags.
UNIFIED_SYSTEM_PROMPT = (
    "You are an expert clinical pharmacologist and medical decision-making AI assistant specializing in drug therapy, treatment planning, and pharmaceutical safety. "
    "Your role is to provide precise, evidence-based answers about medications, their uses, contraindications, dosing, interactions, and clinical applications.\n\n"
    "--------------------------------------------------------------------------------\n\n"
    "The reasoning process and answer MUST be enclosed within <thinking> </thinking> and <answer> </answer> tags.\n"
    "- Always start your response with the <thinking> tag and end with the </answer> tag.\n"
    "- Do not include any text or commentary before the opening <thinking> tag or after the closing </answer> tag.\n"
    "- Do not include any text or commentary between the closing </thinking> tag and the opening <answer> tag.\n\n"
    "## Core Competencies:\n"
    "- Drug mechanisms of action, pharmacokinetics, and pharmacodynamics\n"
    "- Clinical contraindications and safety profiles\n"
    "- Drug interactions and combination therapies\n"
    "- Dosing regimens and administration guidelines\n"
    "- Treatment protocols for various medical conditions\n"
    "- Drug repurposing and alternative therapeutic applications\n"
    "- Adverse effects monitoring and management\n"
    "- Special population considerations (pediatric, geriatric, pregnancy, renal/hepatic impairment)\n"
    "- Brand name and generic drug identification and differentiation\n\n"
    "## Answer Guidelines:\n"
    "1. **Precision First**: Provide exact, specific answers based on established clinical evidence and drug labeling information\n"
    "2. **Safety Priority**: Always prioritize patient safety when discussing contraindications, warnings, and precautions\n"
    "3. **Evidence-Based**: Ground responses in pharmaceutical literature, clinical guidelines, and regulatory information\n"
    "4. **Multiple Choice Strategy**: For multiple choice questions, eliminate incorrect options systematically and select the most clinically appropriate answer and STRICTLY return the OPTION ONLY in between <answer> </answer> tags.\n"
    "5. **Open-Ended Responses**: For open-ended questions, provide comprehensive but focused answers that directly address the clinical scenario\n"
    "6. **Clinical Context**: Always consider patient-specific factors, comorbidities, and real-world clinical scenarios\n\n"
    "## Key Focus Areas:\n"
    "- Brand name and generic drug identification with specific attention to proprietary formulations\n"
    "- Indication-specific treatment recommendations and first-line vs. alternative therapies\n"
    "- Contraindication assessment and risk stratification (absolute vs. relative contraindications)\n"
    "- Dosage calculations and adjustments for different populations and clinical conditions\n"
    "- Drug administration techniques, timing, and route-specific considerations\n"
    "- Monitoring parameters and safety protocols for therapeutic drug management\n"
    "- Drug storage and stability requirements under various conditions\n"
    "- Pregnancy/lactation safety categories and reproductive health considerations\n"
    "- Drug-disease and drug-drug interactions with clinical significance assessment\n"
    "- Adverse effect profiles and management strategies\n"
    "- Therapeutic equivalence and bioequivalence considerations\n"
    "- Drug-food interactions and lifestyle modifications\n"
    "- Account for genetic polymorphisms affecting drug metabolism and response\n"
    "- Evaluate risk-benefit ratios in complex clinical scenarios\n\n"
    "## Special Population Considerations:\n"
    "- **Pediatric patients**: Age-appropriate dosing, safety profiles, and developmental considerations\n"
    "- **Geriatric patients**: Polypharmacy concerns, altered pharmacokinetics, and fall risk\n"
    "- **Pregnant/lactating women**: Teratogenicity risk, pregnancy categories, and breastfeeding safety\n"
    "- **Patients with renal impairment**: Dose adjustments and alternative therapies\n"
    "- **Patients with hepatic impairment**: Metabolic considerations and contraindications\n"
    "- **Immunocompromised patients**: Infection risk and drug interactions with immunosuppressants\n\n"
    "You must provide accurate, clinically sound answers that would be appropriate for healthcare decision-making contexts. Focus on therapeutic efficacy, safety profiles, and evidence-based clinical practice. Your responses should reflect the depth of knowledge expected from a clinical pharmacology expert."
)


# Model Classes
class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
    
    @abstractmethod
    def load(self, **kwargs):
        """Load the model"""
        pass
    
    @abstractmethod
    def inference(self, prompt: str, max_tokens: int = 4096) -> Tuple[str, List[Dict]]:
        """Run inference on the model
        
        Returns:
            Tuple of (response, messages) where messages is the complete conversation history
        """
        pass


class ChatGPTModel(BaseModel):
    """ChatGPT/OpenAI model wrapper"""
    
    def load(self, **kwargs):
        """Load ChatGPT model"""
        api_key = os.getenv("AZURE_OPENAI_API_KEY_O1")
        api_version = "2024-12-01-preview"

        if not api_key:
            raise ValueError("API key not found in environment. Please set the appropriate environment variable.")
        
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        from openai import AzureOpenAI
        print("Initializing AzureOpenAI client with endpoint:", azure_endpoint)
        print("Using API version:", api_version)
        self.model_client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    
    def inference(self, prompt: str, max_tokens: int = 4096) -> Tuple[str, List[Dict]]:
        """ChatGPT inference with specified sampling parameters."""
        messages = [{"role": "system", "content": UNIFIED_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        
        responses = self.model_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.6,
                top_p=0.95,
                presence_penalty=1.0,
            )
        response_text = responses.choices[0].message.content
        
        # Create complete conversation history
        complete_messages = messages + [{"role": "assistant", "content": response_text}]
        
        return response_text, complete_messages


class LocalModel(BaseModel):
    """Local HuggingFace model wrapper"""
    
    def load(self, **kwargs):
        """Load local HuggingFace model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                **kwargs
            )
            logger.info(f"Loaded local model: {self.model_name}")
        except ImportError as e:
            logger.error(f"Failed to import local model dependencies: {e}")
            raise
    
    def inference(self, prompt: str, max_tokens: int = 4096) -> Tuple[str, List[Dict]]:
        """Local model inference with specified sampling parameters."""
        messages = [
            {"role": "system", "content": UNIFIED_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        print("messages:", messages)
        
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors='pt'
        ).to(self.model.device)
        
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            presence_penalty=1.0,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True
        )
        
        response = outputs[0][input_ids.shape[-1]:]
        response_text = self.tokenizer.decode(response, skip_special_tokens=True)
        print("response_text:", response_text)
        
        # Create complete conversation history
        complete_messages = messages + [{"role": "assistant", "content": response_text}]
        
        return response_text, complete_messages


class vLLMModel(BaseModel):
    """vLLM model wrapper for efficient inference"""
    
    def load(self, **kwargs):
        """Load vLLM model"""
        try:
            from vllm import LLM, SamplingParams
            
            # Default vLLM parameters
            vllm_kwargs = {
                'tensor_parallel_size': kwargs.get('tensor_parallel_size', 1),
                'gpu_memory_utilization': kwargs.get('gpu_memory_utilization', 0.9),
                'max_model_len': kwargs.get('max_model_len', 8192),
                'trust_remote_code': kwargs.get('trust_remote_code', True),
            }
            
            # Remove vLLM-specific kwargs from general kwargs
            for key in ['tensor_parallel_size', 'gpu_memory_utilization', 'max_model_len', 'trust_remote_code']:
                kwargs.pop(key, None)
            
            # Merge remaining kwargs
            vllm_kwargs.update(kwargs)
            
            self.model = LLM(model=self.model_name, **vllm_kwargs)
            # Updated sampling parameters
            self.sampling_params = SamplingParams(
                max_tokens=4096,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                presence_penalty=2.0,
            )
            logger.info(f"Loaded vLLM model: {self.model_name}")
        except ImportError as e:
            logger.error(f"Failed to import vLLM dependencies: {e}")
            raise
    
    def inference(self, prompt: str, max_tokens: int = 4096) -> Tuple[str, List[Dict]]:
        """vLLM inference with specified sampling parameters."""
        # Format prompt for vLLM
        formatted_prompt = f"<|system|>\n{UNIFIED_SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>\n"
        
        # Update sampling parameters
        self.sampling_params.max_tokens = max_tokens
        
        # Generate response
        outputs = self.model.generate([formatted_prompt], self.sampling_params)
        response_text = outputs[0].outputs[0].text.strip()
        
        print("vLLM response_text:", response_text)
        
        # Create conversation history
        messages = [
            {"role": "system", "content": UNIFIED_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response_text}
        ]
        
        return response_text, messages


class CustomModel(BaseModel):
    """Custom model wrapper for user-defined models"""
    
    def __init__(self, model_name: str, model_instance, inference_func):
        super().__init__(model_name)
        self.model = model_instance
        self._inference_func = inference_func
    
    def load(self, **kwargs):
        """Custom models are already loaded"""
        logger.info(f"Using custom model: {self.model_name}")
    
    def inference(self, prompt: str, max_tokens: int = 4096) -> Tuple[str, List[Dict]]:
        """Custom model inference"""
        try:
            # For custom models, we'll create a simple message structure
            messages = [{"role": "user", "content": prompt}]
            
            response = self._inference_func(self.model, prompt, max_tokens)
            
            # Create complete conversation history
            complete_messages = messages + [{"role": "assistant", "content": response}]
            
            return response, complete_messages
        except Exception as e:
            logger.error(f"Custom model inference error: {e}")
            error_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "Error occurred"}
            ]
            return "Error occurred", error_messages


class CompetitionKit:
    """
    Simple competition framework - everything you need in one class!
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the competition kit
        
        Args:
            config_path: Path to configuration file containing dataset configs
        """
        self.model = None
        self.model_name = None
        
        self.config = json.load(open(config_path, 'r')) if config_path else {}
        
        self.output_dir = self.config.get('output_dir', 'results')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load dataset configurations from config file or use defaults
        self.datasets = self._load_dataset_configs(self.config)
    
    def load_model(self, model_name: str, model_type: str = "auto", **kwargs):
        """
        Load a model for evaluation
        
        Args:
            model_name: Name/path of the model (e.g., "gpt-4o-mini", "meta-llama/Llama-2-7b-chat-hf")
            model_type: Type of model ("chatgpt", "local", "vllm", "custom", "auto" for auto-detection)
            **kwargs: Additional model configuration
        """
        self.model_name = model_name
        
        # Auto-detect model type if not specified
        if model_type == "auto":
            model_type = self._detect_model_type(model_name)
        
        logger.info(f"Loading model: {model_name} (type: {model_type})")
        
        if model_type == "chatgpt":
            self.model = ChatGPTModel(model_name)
        elif model_type == "local":
            self.model = LocalModel(model_name)
        elif model_type == "vllm":
            self.model = vLLMModel(model_name)
        elif model_type == "custom":
            # For custom models, user should provide model_instance and inference_func
            model_instance = kwargs.get("model_instance")
            inference_func = kwargs.get("inference_func")
            if not model_instance or not inference_func:
                raise ValueError("Custom model requires 'model_instance' and 'inference_func' parameters")
            self.model = CustomModel(model_name, model_instance, inference_func)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load the model
        self.model.load(**kwargs)
    
    def _load_dataset_configs(self, config) -> Dict:
        """
        Load dataset configurations from config file or return defaults
        
        Args:
            config: Configuration dictionary

        Returns:
            Dictionary of dataset configurations
        """
        if not config:
            print("No config provided, exiting.")
            sys.exit(1)

        # Check if config has a single dataset configuration
        if 'dataset' in config:
            dataset_config = config['dataset']
            dataset_name = dataset_config.get('dataset_name', 'treatment')
            # Create a dictionary with the dataset name as key
            return {dataset_name: dataset_config}
        else:
            # If no dataset in config, return defaults
            print("No dataset config found, exiting.")
            sys.exit(1)

    def _detect_model_type(self, model_name: str) -> str:
        """Auto-detect model type based on model name"""
        if "gpt" in model_name.lower() or "o1" in model_name.lower():
            return "chatgpt"
        elif "/" in model_name:  # HuggingFace format
            return "local"
        else:
            return "local"  # Default to local
    
    def load_dataset(self, dataset_name: str) -> List[Dict]:
        """
        Load a dataset by name
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            List of dataset examples
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(self.datasets.keys())}")
        
        dataset_config = self.datasets[dataset_name]
        dataset_path = dataset_config.get('dataset_path')
        
        if not dataset_path or not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        logger.info(f"Loading dataset: {dataset_name} from {dataset_path}")
        
        # Load JSONL file
        examples = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line.strip()))
        
        logger.info(f"Loaded {len(examples)} examples from {dataset_name}")
        return examples
    
    def _check_existing_progress(self, output_file: str) -> int:
        """Check how many examples have already been processed"""
        if not os.path.exists(output_file):
            return 0
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                count = sum(1 for row in reader)
                logger.info(f"Found {count} existing predictions in {output_file}")
                return count
        except Exception as e:
            logger.warning(f"Error reading existing file {output_file}: {e}")
            return 0
    
    def _write_csv_header(self, output_file: str):
        """Write CSV header if file doesn't exist"""
        if not os.path.exists(output_file):
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'prediction', 'choice', 'reasoning'])
    
    def _append_prediction_to_csv(self, output_file: str, prediction: Dict):
        """Append a single prediction to CSV file"""
        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                prediction.get('id', ''),
                prediction.get('prediction', ''),
                prediction.get('choice', ''),
                prediction.get('reasoning', '') # Write the new string-based reasoning
            ])
    
    def evaluate(self, dataset_name: str, output_file: str = None, max_examples: int = None) -> EvaluationResult:
        """
        Evaluate model on a dataset with incremental CSV writing and resume functionality
        
        Args:
            dataset_name: Name of the dataset to evaluate on
            output_file: Path to output CSV file for incremental writing
            max_examples: Maximum number of examples to evaluate (for testing)
            
        Returns:
            EvaluationResult object
        """
        if not self.model:
            raise ValueError("No model loaded. Call load_model() first.")
        
        # Load dataset
        examples = self.load_dataset(dataset_name)
        
        if max_examples:
            examples = examples[:max_examples]
        
        # Set up output file for incremental writing
        if output_file is None:
            output_file = os.path.join(self.output_dir, f"{dataset_name}_predictions.csv")
        
        # Check existing progress
        start_idx = self._check_existing_progress(output_file)
        
        # Write header if starting fresh
        if start_idx == 0:
            self._write_csv_header(output_file)
        
        # Skip already processed examples
        remaining_examples = examples[start_idx:]
        
        if not remaining_examples:
            logger.info("All examples already processed!")
            # Load existing predictions for result calculation
            predictions = []
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    predictions.append(row)
            
            return EvaluationResult(
                dataset_name=dataset_name,
                model_name=self.model_name,
                accuracy=0.0,
                correct_predictions=0,
                total_examples=len(predictions),
                predictions=predictions
            )
        
        logger.info(f"Starting evaluation from example {start_idx + 1}/{len(examples)}")
        
        predictions = []
        reasoning_traces = []
        correct_predictions = 0
        
        # Process remaining examples
        for i, example in enumerate(tqdm(remaining_examples, desc="Evaluating")):
            try:
                # Create prompt with specific output format instructions
                prompt = self._create_prompt(example)
                
                # Get model response
                response, messages = self.model.inference(prompt)
                
                # Parse the raw response to extract the final answer
                prediction = self._parse_response(example, response)
                
                # Add prediction and the full conversation history (as reasoning) to lists
                predictions.append(prediction)
                reasoning_traces.append(messages)
                
                # Immediately write to CSV
                self._append_prediction_to_csv(output_file, prediction)
                
                # Print the response in the desired format
                print(f"\n--- Prediction for ID: {prediction.get('id', '')} ---")
                print(f"Choice: {prediction.get('choice', '')}")
                print(f"Prediction: {prediction.get('prediction', '')}")
                print(f"Reasoning: {prediction.get('reasoning', '')}")
                print("----------------------------------------")

                # Check correctness if ground truth available
                if self._has_ground_truth(example):
                    if self._is_correct(example, prediction):
                        correct_predictions += 1
                
                # Log progress every 10 examples
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {start_idx + i + 1}/{len(examples)} examples")
                    
            except Exception as e:
                logger.error(f"Error processing example {example.get('id', start_idx + i + 1)}: {e}")
                # Write error prediction to maintain consistency
                error_prediction = {
                    'id': example.get('id', f'example_{start_idx + i + 1}'),
                    'prediction': f'Error: {str(e)}',
                    'choice': 'ERROR',
                    'reasoning': f'Error: {str(e)}'
                }
                predictions.append(error_prediction)
                self._append_prediction_to_csv(output_file, error_prediction)
                continue
        
        # Calculate final accuracy based on newly processed examples
        total_newly_processed = len(remaining_examples)
        accuracy = (correct_predictions / total_newly_processed) if total_newly_processed > 0 else 0.0
        
        logger.info(f"Evaluation complete! Processed {len(examples)} total examples")
        logger.info(f"Results saved to: {output_file}")
        
        # Note: The returned result is for the current run only.
        return EvaluationResult(
            dataset_name=dataset_name,
            model_name=self.model_name,
            accuracy=accuracy,
            correct_predictions=correct_predictions,
            total_examples=total_newly_processed,
            predictions=predictions,
            reasoning_traces=reasoning_traces
        )
    
    def _create_prompt(self, example: Dict) -> str:
        """Create prompt from example, with instructions for <thinking> and <answer> format."""
        question = example.get('question', '')
        question_type = example.get('question_type', '')
        
        # Instruction for the model's output format
        instruction = "After your reasoning, conclude with the final answer."

        if question_type in ["multi_choice", "open_ended_multi_choice"]:
            # For multi-choice questions
            options = example.get('options', {})
            options_text = "\n".join([f"{key}. {value}" for key, value in options.items()])
            prompt = f"{question}\n\n{options_text}\n\n{instruction}\n\nAnswer:"
        else:
            # For open-ended questions
            prompt = f"{question}\n\n{instruction}\n\nAnswer:"
        
        return prompt
    
    def _parse_response(self, example: Dict, response: str) -> Dict:
        """
        Parses a model's response into a structured dictionary with 'prediction', 
        'choice', and 'reasoning' fields based on the question_type.
        """
        question_type = example.get('question_type')
        q_id = example.get('id', 'N/A')

        # Initialize the output dictionary with default error messages
        parsed_result = {
            'id': q_id,
            'prediction': 'Error: Could not extract <thinking> block.',
            'choice': 'Error: Could not determine choice.',
            'reasoning': 'Error: Could not determine reasoning.',
        }

        # Regex to find content within <thinking>...</thinking>
        thinking_match = re.search(
            r"<\s*thinking[^>]*>(.*?)<\s*/\s*thinking\s*>", 
            response, 
            re.DOTALL | re.IGNORECASE
        )
        
        # Regex to find content within <answer>...</answer>
        answer_match = re.search(
            r"<\s*answer[^>]*>(.*?)<\s*/\s*answer\s*>", 
            response, 
            re.DOTALL | re.IGNORECASE
        )

        # Handle cases where tags are missing
        if not thinking_match:
            logger.error(f"FATAL: Could not find '<thinking>...</thinking>' block for ID {q_id}.")
            return parsed_result # Return with thinking error
        
        if not answer_match:
            logger.error(f"FATAL: Could not find '<answer>...</answer>' block for ID {q_id}.")
            parsed_result['prediction'] = thinking_match.group(1).strip() if thinking_match else parsed_result['prediction']
            parsed_result['choice'] = 'Error: Could not extract <answer> block.'
            parsed_result['reasoning'] = 'Error: Could not extract <answer> block.'
            return parsed_result

        # Extract the clean text content
        thinking_content = thinking_match.group(1).strip()
        answer_content = answer_match.group(1).strip()

        # Rule 1: The PREDICTION column is always the content from the <thinking> block.
        parsed_result['prediction'] = thinking_content
        
        # Rule 2 & 3: Determine CHOICE and REASONING based on question_type.
        if question_type == "multi_choice":
            # CHOICE is the content from the <answer> block.
            parsed_result['choice'] = answer_content
            # REASONING is also the content from the <answer> block.
            parsed_result['reasoning'] = answer_content
        
        elif question_type == "open_ended":
            # CHOICE is the special string 'NOTAVALUE'.
            parsed_result['choice'] = 'NOTAVALUE'
            # REASONING is the content from the <thinking> block.
            parsed_result['reasoning'] = thinking_content
        
        elif question_type == "open_ended_multi_choice":
            # CHOICE is the content from the <answer> block.
            parsed_result['choice'] = answer_content
            # REASONING is also the content from the <answer> block.
            parsed_result['reasoning'] = thinking_content
            
        else:
            # Handle unknown question types as an error
            error_msg = f"Unknown question_type: '{question_type}'"
            logger.warning(f"{error_msg} for ID {q_id}")
            parsed_result['choice'] = error_msg
            parsed_result['reasoning'] = error_msg

        return parsed_result
    
    def _has_ground_truth(self, example: Dict) -> bool:
        """Check if example has ground truth for evaluation"""
        return 'correct_answer' in example or 'answer' in example
    
    def _is_correct(self, example: Dict, prediction: Dict) -> bool:
        """Check if prediction is correct"""
        ground_truth = example.get('correct_answer') or example.get('answer')
        if not ground_truth:
            return False
        
        question_type = example.get('question_type', 'multiple_choice')
        
        if question_type in ["multi_choice", "open_ended_multi_choice"]:
            return prediction['choice'] == ground_truth
        else: # open_ended
            # For open-ended, perform a case-insensitive comparison
            return prediction['reasoning'].lower().strip() == ground_truth.lower().strip()
    
    def save_submission_with_metadata(self, results: EvaluationResult, output_path: str = None):
        """
        Save submission with metadata in the required format
        
        Args:
            results: EvaluationResult object from the current run
            output_path: Path to save the submission file
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"submission_{results.dataset_name}.zip")
        
        # The submission should contain all predictions, not just from the last run.
        full_csv_path = os.path.join(self.output_dir, f"{results.dataset_name}_predictions.csv")

        # Create temporary directory for submission files
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # The metadata should reflect the entire dataset evaluation.
            # For simplicity, this implementation uses the stats from the final run as an approximation.
            metadata = self.config.get('metadata', {})
            metadata.update({
                'accuracy': results.accuracy,
                'total_examples_in_run': results.total_examples,
                'correct_predictions_in_run': results.correct_predictions
            })
            
            metadata_path = os.path.join(temp_dir, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            # Create ZIP file with the complete predictions CSV
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(full_csv_path, "predictions.csv")
                zipf.write(metadata_path, "metadata.json")
        
        logger.info(f"Submission saved to: {output_path}")
        return output_path


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Bio-Medical AI Competition Framework")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--dataset", help="Dataset name to evaluate")
    parser.add_argument("--model", help="Model name/path")
    parser.add_argument("--model-type", default="auto", help="Model type (auto, chatgpt, local, vllm, custom)")
    parser.add_argument("--max-examples", type=int, help="Maximum examples to evaluate")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    # Initialize framework
    framework = CompetitionKit(config_path=args.config)
    
    # Load model
    model_name = args.model or framework.config.get('metadata', {}).get('model_name')
    model_type = args.model_type
    
    if framework.config.get('metadata', {}).get('model_type'):
        model_type = framework.config['metadata']['model_type']
    
    if not model_name:
        raise ValueError("Model name must be specified in config or command line")
    
    framework.load_model(model_name, model_type.lower())
    
    # Get dataset name
    dataset_name = args.dataset or list(framework.datasets.keys())[0]
    
    # Run evaluation
    results = framework.evaluate(
        dataset_name=dataset_name,
        output_file=args.output,
        max_examples=args.max_examples
    )
    
    # Save submission
    submission_path = framework.save_submission_with_metadata(results)
    
    print(f"\n✅ Evaluation completed successfully!")
    print(f"📊 Accuracy for this run: {results.accuracy:.2%} ({results.correct_predictions}/{results.total_examples})")
    print(f"📄 Submission saved to: {submission_path}")
    
    # Show metadata summary if verbose
    final_metadata = framework.config.get('metadata', {})
    print("\n📋 Final metadata:")
    for key, value in final_metadata.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()