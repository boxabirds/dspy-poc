# DSPy Demo Project

This project demonstrates how DSPy works with a simple, easy-to-understand example. DSPy is a framework for programming with foundation models (like large language models) in a more structured and reliable way.

## Setup

This project uses Python 3.12 and `uv` for dependency management.

The environment has already been set up with:

```bash
# Create virtual environment
uv venv -p 3.12 .venv

# Initialize project
uv init

# Install dependencies
uv pip install -e .
```

## Running the Demo

1. First, you need to set up your Gemini API key. If you don't have one, you can get it from [Google AI Studio](https://makersuite.google.com/app/apikey).

2. Set the API key as an environment variable:

```bash
export GEMINI_API_KEY='your-api-key-here'
```

3. Activate the virtual environment:

```bash
source .venv/bin/activate
```

4. Run the demo to see a simple question-answering system powered by Google's Gemini model:

```bash
python dspy_demo.py
```

This demo shows:
- How to define a DSPy Signature (input/output contract)
- How to create a basic DSPy Module
- How to use the Module to process inputs and get structured outputs

## How DSPy Works

DSPy helps you build reliable applications with large language models by:

1. **Defining Signatures**: Clear contracts for inputs and outputs
2. **Creating Modules**: Reusable components that implement specific tasks
3. **Composing Pipelines**: Connecting modules together for complex workflows
4. **Optimizing Prompts**: Automatically improving prompts based on examples

## Key DSPy Concepts

- **Signatures**: Define the "contract" for what goes in and what comes out
- **Modules**: Building blocks for LLM tasks (Predict, ChainOfThought, etc.)
- **Teleprompters**: Automatic prompt optimization based on examples
- **Composability**: Building complex applications from simpler pieces

For more information, visit the [DSPy GitHub repository](https://github.com/stanfordnlp/dspy).
