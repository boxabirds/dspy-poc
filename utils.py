"""
Utility functions for the DSPy sentiment analysis project.
"""
import json
import os
import dspy
import logging
import litellm
from dspy.utils.callback import BaseCallback

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG to capture all logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dspy_prompts.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('dspy-poc')

# Configure LiteLLM to log prompts and responses
litellm.verbose = True  # Enable verbose logging for LiteLLM

def load_json_data(file_path):
    """
    Load data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        list: The loaded JSON data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return json.load(f)

def load_examples(data_path, input_field="review"):
    """
    Load data from a JSON file and convert to DSPy examples.
    
    Args:
        data_path (str): Path to the JSON file
        input_field (str): The field to use as input
        
    Returns:
        list: List of DSPy Example objects
    """
    data = load_json_data(data_path)
    examples = [
        dspy.Example(**item).with_inputs(input_field)
        for item in data
    ]
    return examples

def sentiment_match(example, pred, trace=None):
    """
    Evaluate if the predicted sentiment matches the ground truth.
    
    Args:
        example: The ground truth example
        pred: The prediction
        trace: Optional trace information
        
    Returns:
        bool: True if the sentiments match, False otherwise
    """
    return example.sentiment == pred.sentiment

def print_evaluation_results(test_examples, predictions):
    """
    Print the evaluation results in a table format.
    
    Args:
        test_examples (list): List of test examples
        predictions (list): List of predictions
        
    Returns:
        float: The accuracy as a value between 0 and 1
    """
    correct = 0
    total = len(test_examples)
    
    print("\nEvaluation Results:")
    print("-" * 80)
    print(f"{'Review':<50} | {'Expected':<10} | {'Predicted':<10}")
    print("-" * 80)
    
    for i, example in enumerate(test_examples):
        prediction = predictions[i]
        is_correct = prediction.sentiment == example.sentiment
        
        if is_correct:
            correct += 1
        
        print(f"{example.review[:47] + '...' if len(example.review) > 47 else example.review:<50} | "
              f"{example.sentiment:<10} | {prediction.sentiment:<10} {'✓' if is_correct else '✗'}")
    
    accuracy = correct / total
    print("-" * 80)
    print(f"Accuracy: {correct}/{total} ({accuracy:.2%})")
    
    return accuracy

def calculate_accuracy(test_examples, predictions):
    """
    Calculate the accuracy of predictions.
    
    Args:
        test_examples: List of test examples with ground truth
        predictions: List of model predictions
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    correct = 0
    for i, (example, pred) in enumerate(zip(test_examples, predictions)):
        if example.sentiment == pred.sentiment:
            correct += 1
    
    accuracy = correct / len(test_examples) if test_examples else 0
    
    return accuracy

def setup_prompt_logging():
    """
    Set up logging to capture prompts sent to the LLM.
    Uses a simple callback approach.
    
    Returns:
        The logger object
    """
    # Enable LiteLLM verbose logging
    litellm.verbose = True
    os.environ["LITELLM_LOG_PROMPTS"] = "True"
    
    # Create a simple callback that logs the inputs
    class SimplePromptLogger(BaseCallback):
        def on_lm_start(self, call_id, inputs, **kwargs):
            logger.info(f"LLM CALL INPUTS: {inputs}")
            
            # Check for different prompt formats
            if 'prompt' in inputs:
                prompt = inputs['prompt']
                logger.info(f"PROMPT: {prompt}")
                print(f"PROMPT: {prompt}")
            elif 'messages' in inputs:
                messages = inputs['messages']
                logger.info(f"MESSAGES: {messages}")
                # Extract the content from the messages
                for msg in messages:
                    if 'content' in msg:
                        logger.info(f"MESSAGE CONTENT: {msg['content']}")
                        print(f"MESSAGE CONTENT: {msg['content']}")
            else:
                logger.warning(f"Could not extract prompt from LM call. Inputs: {inputs}")
            
        def on_lm_end(self, call_id, outputs, exception=None, **kwargs):
            if outputs:
                logger.info(f"RESPONSE: {outputs}")
                print(f"RESPONSE: {outputs}")
            if exception:
                logger.error(f"ERROR: {exception}")
    
    # Register our callback
    dspy.settings.configure(callbacks=[SimplePromptLogger()])
    
    logger.info("Prompt logging enabled")
    print("Prompt logging enabled")
    return logger

def interactive_demo(classifier):
    """
    Run an interactive demo allowing users to input reviews and get predictions.
    
    Args:
        classifier: The trained classifier to use for predictions
    """
    print("\nInteractive Demo Mode")
    print("Enter 'quit' to exit")
    
    while True:
        user_review = input("\nEnter a movie review: ")
        if user_review.lower() == 'quit':
            break
        
        logger.info(f"Processing user input: {user_review}")
        result = classifier(review=user_review)
        logger.info(f"Prediction result: {result.sentiment}")
        print(f"Predicted Sentiment: {result.sentiment}")
