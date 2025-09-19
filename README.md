# Multilingual Voice RAG Agent 🎙️🤖

A comprehensive voice-enabled AI assistant that combines speech-to-text, retrieval-augmented generation (RAG), and text-to-speech capabilities. Built with LangGraph, it supports multilingual conversations and maintains context across sessions while providing access to both web search and local knowledge bases.

## ✨ Features

- **🎤 Voice Input**: Real-time speech recognition using Google Cloud Speech-to-Text
- **📚 RAG Knowledge Base**: Local document search and retrieval using ChromaDB
- **🔊 Text-to-Speech**: High-quality voice synthesis with ElevenLabs
- **🌐 Web Search**: Internet search capabilities via Tavily
- **💬 Memory Management**: Persistent conversation history with LangGraph
- **🌍 Multilingual Support**: Arabic and English language processing
- **🔧 Modular Design**: Separate text-only and voice-enabled modes

## 🏗️ Architecture

The system consists of three main components:

1. **RAG Chatbot** (`memoryrag.py`): Core chatbot with knowledge base integration
2. **Voice Interface** (`voice-rag.py`): Full voice conversation capabilities
3. **Knowledge Base**: ChromaDB vector store for document retrieval

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Cloud Speech-to-Text API credentials
- ElevenLabs API key (for TTS)
- vLLM server running Qwen2.5-32B-Instruct

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/multilingual-speech-agent.git
cd multilingual-speech-agent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
export ELEVENLABS_API_KEY="your-elevenlabs-api-key"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/gcp-credentials.json"
```

4. **Prepare knowledge base**
```bash
mkdir knowledge_base
# Add your PDF and TXT files to this directory
```

### Usage

#### Text-Only Mode
```bash
python memoryrag.py
```

#### Full Voice Mode
```bash
python voice-rag.py --language_code ar-SA --elevenlabs_key YOUR_KEY
```

#### Voice Commands
- **Arabic**: "خروج" or "توقف" to exit, "جديد" for new conversation
- **English**: "exit" or "quit" to stop, "new" for new conversation

## ⚙️ Configuration

### vLLM Server Configuration
```python
BASE_URL = "https://your-vllm-server:8000/v1"
MODEL = "Qwen/Qwen2.5-32B-Instruct"
```

### Speech Recognition Settings
- **Language**: Arabic (ar-SA) by default
- **Model**: Google Cloud Speech latest_long
- **Audio**: 16kHz, mono, LINEAR16

### TTS Configuration
- **Provider**: ElevenLabs
- **Model**: eleven_turbo_v2_5
- **Default Voice**: Adam (multilingual)

## 📁 Project Structure

```
multilingual-speech-agent/
├── memoryrag.py          # Core RAG chatbot
├── voice-rag.py          # Voice interface
├── requirements.txt      # Python dependencies
├── knowledge_base/       # Documents for RAG
├── chroma_db_rag/       # Vector database
├── API-Keys.txt         # API keys (gitignored)
└── old/                 # Archive files
```

## 🛠️ Development

### Adding Documents
Place PDF or TXT files in the `knowledge_base/` directory. The system automatically:
- Loads and chunks documents
- Creates embeddings using multilingual-e5-small
- Stores in persistent ChromaDB vector store

### Customizing the LLM
Modify the system prompt in `memoryrag.py` to adjust the assistant's behavior:

```python
SYSTEM_PROMPT = """You are a professional AI assistant..."""
```

## 🔧 Troubleshooting

**Common Issues:**
- **No audio output**: Check ElevenLabs API key and voice ID
- **Speech not recognized**: Verify Google Cloud credentials
- **Empty knowledge base**: Ensure documents are in `knowledge_base/` folder
- **Connection errors**: Confirm vLLM server is running and accessible

## 📊 Performance

- **Response Time**: ~2-3 seconds for text queries
- **Voice Latency**: ~3-5 seconds end-to-end
- **Supported Languages**: Arabic, English (extensible)
- **Document Capacity**: Unlimited (ChromaDB scales)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain/LangGraph**: Agent framework
- **Google Cloud Speech**: Speech recognition
- **ElevenLabs**: Text-to-speech synthesis
- **ChromaDB**: Vector database
- **Qwen2.5**: Language model capabilities

---

**Built with ❤️ for multilingual AI conversations**
