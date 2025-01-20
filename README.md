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

MIT License

Copyright (c) 2024 Vinoth Rajalingam

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing

We welcome contributions to improve the LangGraph Chatbot! Here's how you can help:

### Ways to Contribute

1. **Bug Reports**: Open an issue describing the bug and how to reproduce it
2. **Feature Requests**: Submit an issue with a detailed description of your proposed feature
3. **Code Contributions**: Submit pull requests with bug fixes or new features

### Development Process

1. Fork the repository
2. Create a new branch for your feature/fix: `git checkout -b feature-name`
3. Make your changes
4. Write or update tests if necessary
5. Run tests and ensure they pass
6. Commit your changes: `git commit -m "Description of changes"`
7. Push to your fork: `git push origin feature-name`
8. Submit a Pull Request

### Code Style Guidelines

- Follow PEP 8 style guide for Python code
- Write clear, descriptive commit messages
- Include comments and documentation for new features
- Update README if necessary

### Getting Help

If you need help with your contribution:
1. Check existing issues and documentation
2. Open a new issue with your question
3. Tag it with "question" or "help wanted"

Thank you for contributing to make this project better! 