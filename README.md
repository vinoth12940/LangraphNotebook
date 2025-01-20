# LangGraph Chatbot with Claude 3.5

A conversational AI chatbot built with LangGraph, LangChain, and Chainlit, powered by Anthropic's Claude 3.5 model and enhanced with Tavily search capabilities.

## Features

- Interactive chat interface using Chainlit
- Integration with Claude 3.5 Sonnet model
- Web search capabilities using Tavily
- Persistent conversation memory using LangGraph's MemorySaver
- Environment-based configuration

## Prerequisites

- Miniconda or Anaconda (Python 3.8+)
- Anthropic API key
- Tavily API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vinoth12940/LangraphNotebook.git
cd LangraphNotebook
```

2. Create and activate a Conda environment:
```bash
conda create -n langraph-chatbot python=3.12
conda activate langraph-chatbot
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your API keys:
```env
ANTHROPIC_API_KEY=your_anthropic_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Usage

1. Activate the Conda environment (if not already activated):
```bash
conda activate langraph-chatbot
```

2. Start the Chainlit server:
```bash
chainlit run app.py
```

3. Open your browser and navigate to `http://localhost:8000`

4. Start chatting with the AI assistant!

## Project Structure

- `app.py`: Main application file containing the LangGraph setup and Chainlit handlers
- `.env`: Configuration file for API keys
- `requirements.txt`: Python dependencies
- `environment.yml`: Conda environment configuration

## How It Works

The application uses a graph-based architecture with LangGraph:
1. User messages are processed through a state graph
2. The chatbot node processes messages using Claude 3.5
3. When needed, the Tavily search tool provides web search capabilities
4. Conversations are maintained using LangGraph's memory system

## Environment Management

This project uses Conda for environment management. Key details:
- Python version: 3.12
- Platform: osx-arm64 (Apple Silicon)
- Conda version: 24.7.1

To export the current environment:
```bash
conda env export > environment.yml
```

To create environment from exported file:
```bash
conda env create -f environment.yml
```

## License

[Your chosen license]

## Contributing

[Your contribution guidelines] 