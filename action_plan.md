# Updated Action Plan for Multi-Modal Agentic Platform

Below is an **updated action plan** that factors in the new Orchestrator code you've added. This roadmap continues to follow the **phased** approach, but now explicitly acknowledges the **Orchestrator** foundation already in place. The steps are broken into **bite-sized, independently codable tasks**, ensuring each one yields a meaningful improvement while maintaining **Separation of Concerns** and a **Layered Architecture**.

---

## Phase 1: Foundational Orchestrator & Basic Text Interaction (Updated)

### 1.1 [X] Minimal Orchestrator
- **What We've Done**  
  - Created `Orchestrator` and `OrchestratorConfig` classes.  
  - Implemented a `process_input` method that logs user input and returns a placeholder response.  
  - Added a basic usage script (`basic_usage.py`) to demonstrate how to instantiate and call the Orchestrator.  
  - Confirmed logging works as intended (via `loguru`).

**Usage & Integration**:
```python
from mnemosyne.orchestrator import Orchestrator, OrchestratorConfig

orchestrator = Orchestrator(
    config=OrchestratorConfig(debug=True)
)
response = orchestrator.process_input("Your query here")
```

**Purpose**: Provides the foundation for all agent interactions and workflow management.
**Related Features**: Core component that all other features build upon.

### 1.2 [X] Integrate a Basic Text LLM
1. **Add LLM Client** [X]
   - Implemented `_call_llm()` with support for multiple providers (OpenAI, Anthropic, Google)
   - Added fallback provider support for reliability
2. **Refine `process_input`** [X]
   - Now uses actual LLM calls with comprehensive logging
   - Handles errors gracefully with fallback providers
3. **Testing & Validation** [X]
   - Added example in `basic_usage.py`
   - Includes error handling for API issues

**Usage**:
```python
orchestrator = Orchestrator(
    config=OrchestratorConfig(
        primary_provider=LLMProviderConfig(
            provider="openai",
            model="gpt-4"
        ),
        fallback_providers=[...]
    )
)
```

**Purpose**: Enables intelligent responses using state-of-the-art LLMs.
**Related Features**: Foundation for all LLM-based operations.

### 1.3 [X] Basic Memory (In-Memory)
1. **Conversation Buffer** [X]
   - Implemented using `deque` for efficient history management
   - Automatically includes context in LLM prompts
2. **Configuration** [X]
   - Added `history_length` to `OrchestratorConfig`
3. **Testing** [X]
   - Demonstrated in `basic_usage.py`

**Usage**:
```python
orchestrator = Orchestrator(
    config=OrchestratorConfig(
        history_length=10  # Store last 10 messages
    )
)
```

**Purpose**: Enables context-aware conversations.
**Related Features**: Foundation for Vector DB integration.

### 1.4 [X] Enhanced Logging & Debugging
1. **Enhanced Debug Mode** [X]
   - Implemented comprehensive logging with loguru
   - Added structured context for better filtering
2. **Structured Logging** [X]
   - Added file rotation and retention policies
   - Implemented context-aware logging

**Usage**: Debug logs are automatically managed based on config:
```python
orchestrator = Orchestrator(
    config=OrchestratorConfig(debug=True)
)
```

**Purpose**: Facilitates debugging and monitoring.
**Related Features**: Supports all components with standardized logging.

---

## Phase 2: Multi-Modal & Workflow Patterns

### 2.1 [X] Audio Agent (STT/TTS)
1. **Speech-to-Text Integration** [X]
   - Created `AudioAgent` with Whisper integration
   - Added language detection and transcription
2. **Optional: TTS** [X]
   - Implemented voice selection and synthesis

**Usage**:
```python
from mnemosyne.agents.audio_agent import AudioAgent, AudioConfig

agent = AudioAgent(
    config=AudioConfig(
        model_type="base",
        device="cpu",
        language="en"
    )
)
text = await agent.transcribe("audio.wav")
```

**Purpose**: Enables audio processing capabilities.
**Related Features**: Integrates with Orchestrator for multi-modal input.

### 2.2 [X] Image Agent (OCR, Generation)
1. **OCR** [X]
   - Implemented `ImageAgent` with Tesseract/EasyOCR
   - Added text extraction and processing
2. **Image Generation** [X]
   - Added support for image generation from text

**Usage**:
```python
from mnemosyne.agents.image_agent import ImageAgent

agent = ImageAgent()
text = await agent.extract_text("image.jpg")
```

**Purpose**: Enables image processing and generation.
**Related Features**: Works with Orchestrator and EvaluatorAgent.

### 2.3 [X] Parallelization & Basic Evaluator
1. **Parallelization Example** [X]
   - Added async support to Orchestrator
   - Implemented `run_parallel_tasks`
2. **Evaluator-Optimizer** [X]
   - Created `EvaluatorAgent` for output validation
   - Added configurable evaluation criteria

**Usage**:
```python
# Parallel execution
results = await orchestrator.run_parallel_tasks([
    task1,
    task2
], timeout=30.0)

# Evaluation
evaluator = EvaluatorAgent()
result = await evaluator.evaluate_output(text)
```

**Purpose**: Enables efficient parallel processing and output validation.
**Related Features**: Enhances all agents with parallel capabilities and quality checks.

---

## Phase 3: Refinement & Memory

### 3.1 Vector Database Integration
1. **Vector Store Setup**  
   - Choose a vector DB (e.g. Pinecone, Weaviate, or local LanceDB).  
   - Implement a small module (`vector_store.py`) for storing and retrieving text embeddings.
2. **Augmented Retrieval**  
   - For each user query, retrieve relevant context from the vector DB to inject into the LLM prompt (RAG approach).
3. **Persisted Memory**  
   - Optionally store conversation turns in the vector DB so that memory can survive restarts.

**Outcome**:  
A more robust memory mechanism and retrieval-augmented generation to reduce hallucinations.

---

### 3.2 [X] Security & Hardening
1. **Auth & Encryption** [X]
   - Implemented Fernet-based encryption for sensitive data
   - Added configurable encryption key management
2. **Error Handling** [X]
   - Added exponential backoff retry mechanism
   - Implemented configurable timeouts and retry limits

**Usage**:
```python
# Encryption
from mnemosyne.security import Security, SecurityConfig

security = Security(
    config=SecurityConfig(
        max_retries=3,
        initial_wait=1.0,
        max_wait=30.0
    )
)

# Encrypt sensitive data
encrypted = security.encrypt_data(sensitive_data)
decrypted = security.decrypt_data(encrypted)

# Retry mechanism
@Security.with_retries(max_tries=3, initial_wait=1.0, max_wait=5.0)
async def flaky_operation():
    # Your code here
    pass
```

**Purpose**: Ensures data security and system reliability.
**Related Features**: 
- Works with all agents to protect sensitive data (OCR results, audio transcripts)
- Enhances reliability of external API calls with retry mechanism

---

### 3.3 [X] Performance & Monitoring
1. **Caching** [X]
   - Implemented TTL-based caching for embeddings and LLM responses
   - Added size limits and automatic cleanup of old entries
2. **Rate Limiting** [X]
   - Added per-user and global rate limits
   - Implemented type-specific limits (embedding/LLM)
   - Added exponential backoff for retries

**Usage**:
```python
from mnemosyne.caching import CacheManager, CacheConfig, RateLimitConfig

# Initialize cache manager
cache_manager = CacheManager(
    cache_config=CacheConfig(
        embedding_ttl=3600,  # 1 hour
        llm_response_ttl=1800,  # 30 minutes
        max_embedding_size=10000,
        max_llm_size=1000
    ),
    rate_limit_config=RateLimitConfig(
        user_limit=100,  # Per minute
        global_limit=1000,  # Per minute
        embedding_limit=500,  # Per minute
        llm_limit=200  # Per minute
    )
)

# Use decorators for automatic caching and rate limiting
@cache_embedding(cache_manager=cache_manager)
async def generate_embedding(text: str):
    # Your embedding code here
    pass

@cache_llm_response(cache_manager=cache_manager)
async def generate_response(prompt: str):
    # Your LLM code here
    pass
```

**Purpose**: Improves performance and reliability while controlling resource usage.
**Related Features**:
- Works with VectorStore for efficient embedding management
- Enhances Orchestrator with response caching
- Protects external APIs with rate limiting

---

### 3.4 Advanced Workflows [X]

#### Prompt Chaining Library [X]
1. **Core Implementation** [X]
   - Created flexible `PromptChain` class with support for:
     - Linear chains (A → B → C)
     - Branching chains (A → [B1, B2] → C)
     - Dynamic chain modification during execution
     - Per-node validation before execution
   - Added comprehensive logging and error handling

2. **Node Types** [X]
   - PROMPT: LLM prompt nodes
   - BRANCH: Conditional branching nodes
   - MERGE: Combine results from multiple paths
   - TOOL: External tool/function calls

3. **Features** [X]
   - Context Management:
     - Variables passed between nodes
     - Results tracking
     - Execution path history
   - Dynamic Modification:
     - Add/remove nodes during execution
     - Modify connections between nodes
     - Conditional path selection

**Usage**:
```python
from mnemosyne.chains import (
    PromptChain,
    ChainNodeType,
    create_linear_chain,
    create_branching_chain
)

# Create a linear chain
chain = create_linear_chain(
    "text_processor",
    prompts=[
        "Analyze this text: {input_text}",
        "Summarize the analysis: {processed_text}"
    ],
    tools=[{
        "name": "text_processor",
        "params": {"text": "{input_text}"}
    }]
)

# Execute with context
results = await chain.execute(
    initial_context={"input_text": "Hello, World!"},
    tools={"text_processor": my_tool}
)

# Create a branching chain
branch_chain = create_branching_chain(
    "classifier",
    condition_prompt="Is this a question: {input_text}",
    true_branch=["Answer: {input_text}"],
    false_branch=["Process: {input_text}"]
)

# Dynamic modification
chain.add_node(...)
chain.connect(from_id, to_id)
```

**Purpose**: Enables creation of complex, dynamic prompt workflows.

**Benefits**:
- Modular and reusable prompt patterns
- Flexible workflow construction
- Dynamic adaptation to results
- Built-in validation and error handling
- Comprehensive logging for debugging

**Related Features**:
- Integrates with caching system for LLM responses
- Works with rate limiting for external API calls
- Supports all agent types (Text, Audio, Image, Video)
- Compatible with security module for sensitive data

---

## Phase 4: Advanced Workflows & Domain-Specific Agents

### 4.1 Prompt Chaining Library
1. **Chain Utility**  
   - Implement a reusable "PromptChain" class or decorator to define multi-step tasks (Outline → Validate → Expand).  
   - Let the Orchestrator load or execute these chains dynamically based on user requests.
2. **Testing**  
   - Provide small chain examples (like a "blog post writer" flow).

---

### 4.2 Orchestrator-Workers Pattern [X]

1. **Worker Pool Implementation** [X]
   - Created flexible `WorkerPool` class with support for:
     - Multiple worker types (TEXT, AUDIO, IMAGE, VIDEO)
     - Task prioritization and dependencies
     - Worker status tracking and error handling
   - Added comprehensive logging for debugging

2. **Task Management** [X]
   - Implemented task queue with priority-based scheduling
   - Added support for task dependencies
   - Created task status tracking and result storage

3. **Worker Orchestration** [X]
   - Created `WorkerOrchestrator` for managing worker pools
   - Added support for parallel and sequential processing
   - Implemented worker group creation with capabilities

**Usage**:
```python
from mnemosyne.workers import WorkerOrchestrator, WorkerType

# Initialize orchestrator
orchestrator = WorkerOrchestrator()

# Create worker groups
orchestrator.create_worker_group(
    type=WorkerType.TEXT,
    count=3,
    capabilities=["summarize", "analyze"]
)

# Process tasks in parallel
results = await orchestrator.process_parallel(
    tasks=[{"text": "Task 1"}, {"text": "Task 2"}],
    worker_type=WorkerType.TEXT,
    processor=text_processor,
    num_workers=3
)

# Process tasks sequentially
results = await orchestrator.process_sequential(
    tasks=[{"path": "image1.jpg"}, {"path": "image2.jpg"}],
    worker_type=WorkerType.IMAGE,
    processor=image_processor
)

# Add tasks with dependencies
task1_id = orchestrator.pool.add_task(
    type=WorkerType.TEXT,
    content={"text": "Process first"}
)
task2_id = orchestrator.pool.add_task(
    type=WorkerType.IMAGE,
    content={"path": "process_after.jpg"},
    dependencies=[task1_id]
)
```

**Purpose**: Enables efficient parallel processing and task orchestration.

**Benefits**:
- Flexible worker pool management
- Priority-based task scheduling
- Support for task dependencies
- Parallel and sequential processing
- Comprehensive logging and error handling

**Related Features**:
- Works with all agent types (Text, Audio, Image, Video)
- Integrates with logging system for debugging
- Compatible with caching and rate limiting
- Supports the security module for sensitive data

---

### 4.3 Enhanced Evaluator-Optimizer [X]

1. **Multi-Criteria Evaluation** [X]
   - Created flexible `ContentEvaluator` with support for:
     - Multiple evaluation criteria (factual accuracy, style, policy, etc.)
     - Weighted scoring system
     - Configurable thresholds per criterion
     - Comprehensive feedback and suggestions
   - Added validation for weights and thresholds

2. **Automatic Refinement** [X]
   - Implemented iterative refinement loop
   - Added smart prompt generation based on failing criteria
   - Created configurable refinement thresholds
   - Added maximum iteration limits

3. **Context-Aware Evaluation** [X]
   - Added support for evaluation context (audience, style guide, domain)
   - Created criterion-specific prompt templates
   - Implemented context-aware refinement suggestions

**Usage**:
```python
from mnemosyne.evaluator import (
    ContentEvaluator,
    EvaluationConfig,
    EvaluationCriteria
)

# Configure evaluator
config = EvaluationConfig(
    criteria=[
        EvaluationCriteria.FACTUAL_ACCURACY,
        EvaluationCriteria.STYLE_CONSISTENCY
    ],
    weights={
        EvaluationCriteria.FACTUAL_ACCURACY: 0.6,
        EvaluationCriteria.STYLE_CONSISTENCY: 0.4
    },
    thresholds={
        EvaluationCriteria.FACTUAL_ACCURACY: 0.8,
        EvaluationCriteria.STYLE_CONSISTENCY: 0.7
    }
)

evaluator = ContentEvaluator(config)

# Single evaluation
result = await evaluator.evaluate(
    content="Your content here",
    context={"audience": "technical"}
)

# Print scores
for score in result.scores:
    print(f"{score.criterion}: {score.score} - {score.feedback}")

# Automatic refinement
refined_content, history = await evaluator.evaluate_and_refine(
    content="Content to improve",
    max_iterations=3
)
```

**Purpose**: Ensures high-quality output through systematic evaluation and refinement.

**Benefits**:
- Comprehensive quality assessment
- Automated content improvement
- Flexible evaluation criteria
- Context-aware evaluation
- Detailed feedback and suggestions

**Related Features**:
- Works with all content types (text, code, etc.)
- Integrates with logging system for debugging
- Compatible with worker pool for parallel evaluation
- Supports the security module for sensitive data

---

## Phase 5: Performance, Monitoring, & CI/CD

### 5.1 Performance Monitoring [X]

1. **Metrics Collection** [X]
   - Created flexible `PerformanceMetrics` class with support for:
     - Request counts and latencies
     - Error tracking
     - Worker and queue metrics
     - Memory usage monitoring
   - Added Prometheus integration for metrics export

2. **Monitoring Tools** [X]
   - Implemented context managers for tracking:
     - Request durations and outcomes
     - Worker activity
     - Queue sizes
   - Added comprehensive error tracking
   - Created metrics aggregation and summaries

3. **Component Integration** [X]
   - Created `MonitoredComponent` base class
   - Added monitoring decorators and utilities
   - Implemented background metric collection

**Usage**:
```python
from mnemosyne.monitoring import (
    PerformanceMetrics,
    MonitoredComponent
)

# Initialize metrics with Prometheus
metrics = PerformanceMetrics(port=8000)

# Create monitored component
class MyComponent(MonitoredComponent):
    def __init__(self, metrics):
        super().__init__(metrics)
    
    async def process(self, data):
        # Track operation metrics
        async with self.track_operation(
            operation="process",
            agent_type="processor"
        ):
            # Your code here
            pass
        
        # Track worker metrics
        async with self.track_worker(
            agent_type="processor"
        ):
            # Worker code here
            pass
        
        # Record errors
        self.record_error(
            error_type="ValidationError",
            agent_type="processor"
        )
        
        # Update queue metrics
        self.update_queue_size(
            size=len(queue),
            agent_type="processor"
        )

# Get metrics summary
summary = metrics.get_summary(
    agent_type="processor",
    window=timedelta(minutes=5)
)
```

**Purpose**: Enables comprehensive performance monitoring and debugging.

**Benefits**:
- Real-time performance tracking
- Prometheus integration for visualization
- Flexible metric collection
- Error tracking and analysis
- Queue and worker monitoring

**Related Features**:
- Works with all components and agents
- Integrates with logging system
- Compatible with worker pool
- Supports the security module

### 5.2 CI/CD Pipeline

1. **Continuous Integration**  
   - Set up tests to run on each pull request (e.g., GitHub Actions).  
   - Automatically build Docker images on merges to `main`.

---

## Phase 6: Extensions & Community Contributions

1. **Domain-Specific Agents**  
   - E.g. a "Code Generation Agent" that can read/write local files in a restricted sandbox.  
   - A "Legal Document Agent" that references known legal data from the vector store.
2. **UI Integrations**  
   - Optionally create a minimal web front-end or connect to no-code platforms (Rivet, Vellum, etc.).

---

# Where We Stand

- **Currently**:  
  - You have a **minimal Orchestrator** (`Orchestrator` & `OrchestratorConfig`) with basic logging and a placeholder response.  
  - A usage demo (`basic_usage.py`) confirms that the Orchestrator can handle text input and log the interaction.

- **Next Immediate Step**:  
  - **Tie in a live LLM** (see Phase 1.2) so that `process_input` produces real AI-driven responses.  
  - Then add **short-term in-memory conversation history** (Phase 1.3) for basic multi-turn interactions.

---

## Tips & Best Practices

- **Keep PRs Small**: Implement each sub-step in its own branch and PR so code reviews remain manageable.  
- **Document as You Go**: Update `README.md` and your docstrings each time you add a new phase or feature.  
- **Test Often**: Each new feature (e.g., `AudioAgent`, `ImageAgent`) should come with at least a small test script or unit test.

With this updated plan, you'll **incrementally build** a powerful multi-modal agentic platform—starting with the **Orchestrator** you've already set up, then layering in specialized agents, memory, retrieval, and advanced workflows.
```