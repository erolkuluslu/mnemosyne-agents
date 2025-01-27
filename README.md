# Mnemosyne Agents 🤖

A powerful multi-modal agentic platform for orchestrating AI agents across text, audio, image, and video processing tasks.

## 🌟 Features

### Core Components
- **Orchestrator**: Central control system for managing agent interactions
- **Multi-Modal Agents**: Support for text, audio, image, and video processing
- **Memory Management**: Vector store integration for long-term memory
- **Advanced Workflows**: Prompt chaining and parallel processing

### Advanced Features
- **Security & Hardening**
  - Fernet-based encryption for sensitive data
  - Configurable encryption key management
  - Exponential backoff retry mechanism
  - Configurable timeouts and retry limits

- **Performance & Monitoring**
  - Prometheus integration for metrics
  - Request latency tracking
  - Error monitoring
  - Queue and worker metrics
  - Memory usage tracking

- **Caching & Rate Limiting**
  - TTL-based caching for embeddings and LLM responses
  - Per-user and global rate limits
  - Type-specific limits (embedding/LLM)
  - Automatic cleanup of old entries

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/mnemosyne-agents.git
cd mnemosyne-agents

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```python
from mnemosyne.orchestrator import Orchestrator, OrchestratorConfig

# Initialize orchestrator
orchestrator = Orchestrator(
    config=OrchestratorConfig(
        debug=True,
        history_length=10
    )
)

# Process input
response = await orchestrator.process_input("Your query here")
```

### Advanced Examples

#### Prompt Chaining
```python
from mnemosyne.chains import PromptChain, ChainNodeType

# Create a chain
chain = PromptChain("text_processor")

# Add nodes
start_id = chain.add_node(
    ChainNodeType.PROMPT,
    "start",
    "Analyze this text: {input_text}"
)

process_id = chain.add_node(
    ChainNodeType.TOOL,
    "process",
    {"name": "text_processor", "params": {"text": "{input_text}"}}
)

# Connect nodes
chain.connect(start_id, process_id)

# Execute chain
results = await chain.execute(
    initial_context={"input_text": "Hello, World!"}
)
```

#### Performance Monitoring
```python
from mnemosyne.monitoring import PerformanceMetrics, MonitoredComponent

# Initialize metrics
metrics = PerformanceMetrics(port=8000)

# Create monitored component
class MyProcessor(MonitoredComponent):
    async def process(self, data):
        async with self.track_operation("process"):
            # Your processing code here
            pass

# Get metrics summary
summary = metrics.get_summary(window=timedelta(minutes=5))
```

## 🛠️ Architecture

### Component Overview
```
Mnemosyne Agents
├── Orchestrator (Core)
│   ├── Input Processing
│   ├── Agent Routing
│   └── Memory Management
├── Agents
│   ├── Text Agent
│   ├── Audio Agent
│   ├── Image Agent
│   └── Video Agent
├── Advanced Features
│   ├── Prompt Chains
│   ├── Worker Pool
│   └── Evaluator
└── Infrastructure
    ├── Security
    ├── Monitoring
    └── Caching
```

### Key Concepts
- **Orchestrator**: Central coordinator for all agent interactions
- **Agents**: Specialized components for different modalities
- **Chains**: Reusable workflow patterns
- **Workers**: Parallel task processing units
- **Evaluator**: Output quality assessment and refinement

## 📊 Monitoring & Performance

### Available Metrics
- Request counts and latencies
- Error rates and types
- Worker pool utilization
- Queue sizes and processing times
- Memory usage by component

### Prometheus Integration
Metrics are exposed on `:8000/metrics` in Prometheus format:
```
# HELP mnemosyne_requests_total Total number of requests
# TYPE mnemosyne_requests_total counter
mnemosyne_requests_total{agent_type="text",operation="process",status="success"} 42

# HELP mnemosyne_request_duration_seconds Request duration in seconds
# TYPE mnemosyne_request_duration_seconds histogram
mnemosyne_request_duration_seconds_bucket{agent_type="text",operation="process",le="0.1"} 12
```

## 🔒 Security

### Encryption
- All sensitive data is encrypted using Fernet
- Configurable key management
- Secure storage of API keys and credentials

### Rate Limiting
- Per-user and global limits
- Type-specific quotas
- Automatic backoff on rate limit hits

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/test_orchestrator.py
pytest tests/test_agents.py
```

## 📝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- Anthropic for Claude models
- Whisper for audio transcription
- EasyOCR for image text extraction
