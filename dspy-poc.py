"""
DSPy Sentiment Analysis Proof of Concept

This script demonstrates using DSPy to build a sentiment analysis model
for movie reviews using few-shot learning.
"""
import os
import dspy
import logging
import sys
from dspy.teleprompt import BootstrapFewShot
from utils import (
    load_examples, 
    sentiment_match, 
    print_evaluation_results,
    interactive_demo,
    setup_prompt_logging
)

def main():
    """Main function to run the DSPy sentiment analysis pipeline."""
    try:
        # Step 1: Enable prompt logging - must be done before configuring the LM
        logger = setup_prompt_logging()
        print("Prompt logging has been enabled - check logs for details")
        
        # Step 2: Configure the Language Model
        print("Configuring OpenAI model...")
        openai_model = dspy.LM('openai/gpt-4o-mini', temperature=0)
        dspy.configure(lm=openai_model)
        logger.info("Configured DSPy with OpenAI model")
        
        # Test the logging with a simple direct call to the LM
        print("Testing LM logging with a simple call...")
        test_response = openai_model("What is 2+2?")
        logger.info(f"Test response: {test_response}")
        print(f"Test response: {test_response}")
        
        # Step 3: Define the Signature for sentiment analysis
        class SentimentAnalysis(dspy.Signature):
            """Classify the sentiment of a movie review."""
            review = dspy.InputField(desc="A movie review text")
            sentiment = dspy.OutputField(desc="Positive, Negative, or Mixed")

        # Step 4: Load Training and Test Data
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        train_examples = load_examples(os.path.join(data_dir, 'train_reviews.json'))
        test_examples = load_examples(os.path.join(data_dir, 'test_reviews.json'))
        
        print(f"Loaded {len(train_examples)} training examples and {len(test_examples)} test examples")

        # Step 5: Set Up the Basic Classifier
        sentiment_classifier = dspy.Predict(SentimentAnalysis)

        # Step 6: Run predictions on all test examples
        print("\nRunning predictions on test examples...")
        predictions = []
        for example in test_examples:
            prediction = sentiment_classifier(review=example.review)
            predictions.append(prediction)
        
        # Step 7: Print evaluation results with table
        print("\nEvaluation Results:")
        accuracy = print_evaluation_results(test_examples, predictions)
        print(f"Overall Accuracy: {accuracy:.2%}")
            
        print("\nDSPy sentiment analysis demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()