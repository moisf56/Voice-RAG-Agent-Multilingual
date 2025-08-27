import os
import uuid
import re
import json
from datetime import datetime
from typing import Annotated, Optional
from typing_extensions import TypedDict
import sys

# --- LangChain/LangGraph Imports ---
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.tools import tool

# --- LLM and Tool Imports ---
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

# Try Tavily imports based on official documentation
try:
    # New recommended package (langchain-tavily)
    from langchain_tavily import TavilySearchResults
    TAVILY_IMPORT = "new"
    print("[DEBUG] Using new langchain-tavily package")
except ImportError:
    try:
        # Fallback to community package (deprecated but functional)
        from langchain_community.tools.tavily_search import TavilySearchResults
        TAVILY_IMPORT = "community"
        print("[DEBUG] Using langchain_community.tools.tavily_search (deprecated)")
    except ImportError:
        print("[DEBUG] No Tavily package found. Using fallback web search.")
        TAVILY_IMPORT = None

# --- RAG (Knowledge Base) Imports ---
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# Define paths for the knowledge base
PERSIST_DIRECTORY = "./chroma_db_rag"
DATA_DIR = "./knowledge_base"

# Embedding model configuration - make it configurable
EMBEDDING_MODELS = {
    "e5-large": "intfloat/multilingual-e5-large",
    "e5-small": "intfloat/multilingual-e5-small",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5"
}
DEFAULT_EMBEDDING_MODEL = "e5-large"  # Changeable to "bge-small" or "bge-base"

# Supported file types
SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.md', '.doc', '.docx'}


# ==============================================================================
# 2. RAG KNOWLEDGE BASE SETUP
# ==============================================================================

class LazyRAGLoader:
    """Lazy loading RAG system to handle memory efficiently"""
    
    def __init__(self, embedding_model_key: str = DEFAULT_EMBEDDING_MODEL):
        self.embedding_model_key = embedding_model_key
        self.retriever = None
        self.is_loaded = False
        
    def _get_embedding_model(self):
        """Get the embedding model with error handling"""
        model_name = EMBEDDING_MODELS.get(self.embedding_model_key, EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL])
        
        try:
            print(f"Loading embedding model: {model_name}")
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            print("Embedding model loaded successfully.")
            return embeddings
        except Exception as e:
            print(f"Error loading embedding model {model_name}: {e}")
            print("Falling back to default BGE model...")
            try:
                return HuggingFaceEmbeddings(model_name=EMBEDDING_MODELS["bge-small"])
            except Exception as fallback_e:
                print(f"Fallback model also failed: {fallback_e}")
                raise RuntimeError("Could not load any embedding model")
    
    def _is_supported_file(self, file_path: str) -> bool:
        """Check if file type is supported"""
        return any(file_path.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)
    
    def _safe_load_document(self, file_path: str):
        """Safely load a document with error handling"""
        try:
            if file_path.endswith(".pdf"):
                return PyPDFLoader(file_path).load()
            else:
                return TextLoader(file_path, encoding='utf-8').load()
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
            return []
    
    def _load_documents_safely(self):
        """Load documents with comprehensive error handling"""
        if not os.path.exists(DATA_DIR):
            print(f"Warning: Knowledge base directory '{DATA_DIR}' not found.")
            return []
        
        documents = []
        total_files = 0
        loaded_files = 0
        
        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                total_files += 1
                
                if not self._is_supported_file(file_path):
                    print(f"Skipping unsupported file: {file_path}")
                    continue
                
                docs = self._safe_load_document(file_path)
                if docs:
                    documents.extend(docs)
                    loaded_files += 1
                    print(f"Loaded: {file_path}")
        
        print(f"Document loading summary: {loaded_files}/{total_files} files loaded successfully")
        return documents
    
    def load_and_index_documents(self):
        """Load and index documents with lazy loading"""
        if self.is_loaded and self.retriever:
            return self.retriever
            
        print("Initializing RAG knowledge base...")
        
        # Load documents safely
        docs = self._load_documents_safely()
        
        if not docs:
            print("Warning: No documents found or loaded. RAG tool will have limited functionality.")
            # Create empty retriever
            embeddings = self._get_embedding_model()
            vectorstore = Chroma(
                embedding_function=embeddings,
                collection_name="rag_collection",
                persist_directory=PERSIST_DIRECTORY
            )
            self.retriever = vectorstore.as_retriever()
            self.is_loaded = True
            return self.retriever
        
        print(f"Successfully loaded {len(docs)} document(s).")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        print(f"Split into {len(splits)} chunks.")
        
        # Get embedding model
        embeddings = self._get_embedding_model()
        
        # Create vector store
        print("Creating and persisting vector store...")
        try:
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                collection_name="rag_collection",
                persist_directory=PERSIST_DIRECTORY
            )
            # Note: vectorstore.persist() is deprecated in newer Chroma versions
            print(f"Successfully indexed {len(splits)} chunks.")
            
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            self.is_loaded = True
            return self.retriever
            
        except Exception as e:
            print(f"Error creating vector store: {e}")
            raise RuntimeError("Failed to create knowledge base")

# Initialize lazy loader
rag_loader = LazyRAGLoader(DEFAULT_EMBEDDING_MODEL)


# ==============================================================================
# 3. TOOL DEFINITIONS - FIXED WEB SEARCH
# ==============================================================================

@tool
def knowledge_base_search(query: str) -> str:
    """
    Search the internal knowledge base for specific information from local documents.
    Use this for questions about company policies, internal procedures, project details,
    or any information that should be in the local document collection.
    This is your primary source for internal/private information.
    """
    try:
        retriever = rag_loader.load_and_index_documents()
        retrieved_docs = retriever.get_relevant_documents(query)
        
        if not retrieved_docs:
            return "No relevant information found in the knowledge base for this query."
        
        docs_content = "\n\n---\n\n".join([
            f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}" 
            for doc in retrieved_docs
        ])
        return f"Knowledge base results:\n\n{docs_content}"
        
    except Exception as e:
        return f"Error searching knowledge base: {e}"

@tool
def web_search(query: str) -> str:
    """
    Search the web for current information, news, weather, and general knowledge.
    Use this for real-time information, current events, weather updates, and
    any information not available in the local knowledge base.
    """
    try:
        print(f"[DEBUG] Performing web search for: {query}")
        
        # Method 1: Try Tavily if available
        if TAVILY_IMPORT and os.getenv("TAVILY_API_KEY"):
            try:
                if TAVILY_IMPORT == "new":
                    # Use new langchain-tavily package
                    search = TavilySearchResults(
                        max_results=3,
                        search_depth="basic",
                        topic="general",
                        include_answer=True,
                        include_raw_content=False,
                        include_images=False
                    )
                    results = search.invoke({"query": query})
                else:
                    # Use community package (deprecated) - FIXED INVOCATION
                    search = TavilySearchResults(max_results=3)
                    # For community package, invoke with just the query string
                    results = search.invoke(query)
                
                print(f"[DEBUG] Using Tavily search ({TAVILY_IMPORT} package)")
                
                if results:
                    # Format results according to Tavily's response structure
                    formatted_results = []
                    for i, result in enumerate(results, 1):
                        title = result.get('title', 'No title')
                        content = result.get('content', result.get('snippet', 'No content'))
                        url = result.get('url', 'No URL')
                        
                        # Limit content length to avoid overwhelming responses
                        if len(content) > 400:
                            content = content[:400] + "..."
                            
                        formatted_results.append(f"{i}. **{title}**\n{content}\nSource: {url}")
                    
                    return "Web search results (via Tavily):\n\n" + "\n\n".join(formatted_results)
                else:
                    print("[DEBUG] Tavily returned no results")
                
            except Exception as tavily_error:
                print(f"[DEBUG] Tavily search failed: {tavily_error}")
                print(f"[DEBUG] Error type: {type(tavily_error).__name__}")
                # Continue to fallback method
        
        # Method 2: Fallback to DuckDuckGo if Tavily not available
        print("[DEBUG] Using DuckDuckGo fallback search...")
        return _fallback_web_search(query)
        
    except Exception as e:
        print(f"[DEBUG] Web search error: {e}")
        return f"Error performing web search: {e}. Please check your internet connection."

def _fallback_web_search(query: str) -> str:
    """Fallback web search using requests and DuckDuckGo"""
    try:
        import requests
        import json
        from urllib.parse import quote
        
        # Use DuckDuckGo Instant Answer API (free, no API key required)
        encoded_query = quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        results = []
        
        # Get abstract if available
        if data.get('Abstract'):
            results.append({
                'title': data.get('AbstractSource', 'DuckDuckGo'),
                'content': data.get('Abstract'),
                'url': data.get('AbstractURL', 'https://duckduckgo.com')
            })
        
        # Get related topics
        for topic in data.get('RelatedTopics', [])[:2]:
            if isinstance(topic, dict) and 'Text' in topic:
                results.append({
                    'title': topic.get('Text', '').split(' - ')[0] if ' - ' in topic.get('Text', '') else 'Related Topic',
                    'content': topic.get('Text', ''),
                    'url': topic.get('FirstURL', 'https://duckduckgo.com')
                })
        
        # Get infobox
        if data.get('Infobox') and data['Infobox'].get('content'):
            for item in data['Infobox']['content'][:1]:
                if 'data_type' in item and item['data_type'] == 'string':
                    results.append({
                        'title': 'Information',
                        'content': item.get('value', ''),
                        'url': data.get('AbstractURL', 'https://duckduckgo.com')
                    })
        
        if not results:
            return f"No web search results found for '{query}'. Try rephrasing your query."
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            content = result.get('content', 'No content')
            url = result.get('url', 'No URL')
            
            # Limit content length
            if len(content) > 300:
                content = content[:300] + "..."
                
            formatted_results.append(f"{i}. **{title}**\n{content}\nSource: {url}")
        
        return "Web search results (via DuckDuckGo):\n\n" + "\n\n".join(formatted_results)
        
    except Exception as e:
        return f"Fallback web search also failed: {e}. Please check your internet connection."

# Tools list - FIXED
tools = [web_search, knowledge_base_search]


# ==============================================================================
# 4. LLM AND GRAPH SETUP
# ==============================================================================

# Enhanced system prompt with better tool selection guidance
SYSTEM_PROMPT = f"""You are Xenon, a professional and helpful AI assistant. Your original name was Gemini, but your current commercial name is Xenon.

**Tool Selection Strategy:**
1. **knowledge_base_search**: Use for internal/private information, company policies, project details, or anything that should be in local documents.

2. **web_search**: Use for:
   - Current events, news, weather
   - General factual information not in knowledge base
   - Real-time data (stock prices, sports scores, etc.)
   - Public information that changes frequently

3. **Decision Logic**:
   - Try knowledge base first for internal/specific questions
   - If knowledge base returns no meaningful results (with respect to the user's query), consider web search
   - For general knowledge questions, use web search directly
   - You may use both tools if needed to provide comprehensive answers

**Core Guidelines:**
- NEVER fabricate information if tools return no results
- Synthesize information into clear, coherent answers
- Acknowledge the current date when relevant: {datetime.now().strftime('%A, %B %d, %Y')}
- Respond in the same language the user uses
- Be professional, friendly, and respectful
- When using web search, always cite your sources

Begin the conversation."""

# vLLM Server Setup - Optimized for Qwen3 with better tool calling
BASE_URL = "https://jo64otib7w7t6d-8000.proxy.runpod.net/v1"
llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key="EMPTY",
    model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    temperature=0.7,  # Lower temperature for more consistent tool calling
    top_p=0.9,       # Slightly higher for better tool selection
    max_tokens=8192,
    timeout=120,
    streaming=True,
    extra_body={
        "repetition_penalty": 1.05,
        "top_k": 40,    # Higher for better tool selection
        "min_p": 0.05,  # Add min_p for better quality
        "stop": ["</tool_call>", "<|im_end|>"]  # Add stop tokens for cleaner responses
    }
)

# Bind tools with explicit configuration
print("Binding tools to LLM...")
try:
    llm_with_tools = llm.bind_tools(tools, tool_choice="auto")
    print("✓ Tools bound successfully with auto tool choice")
except Exception as e:
    print(f"⚠ Tool binding with auto choice failed: {e}")
    print("Trying basic tool binding...")
    try:
        llm_with_tools = llm.bind_tools(tools)
        print("✓ Basic tool binding successful")
    except Exception as e2:
        print(f"✗ All tool binding attempts failed: {e2}")
        llm_with_tools = llm  # Fallback to LLM without tools

# State definition
class State(TypedDict):
    messages: Annotated[list, add_messages]

def parse_and_execute_text_tool_calls(content: str):
    """Parse text-based tool calls and execute them"""
    tool_results = []
    
    # CORRECTED REGEX: Capture content after <tool_call> even if </tool_call> is missing.
    # It now looks for the start tag and captures the JSON that follows.
    tool_pattern = r'<tool_call>\s*(\{.*\})\s*(?:</tool_call>|$)'
    matches = re.findall(tool_pattern, content, re.DOTALL)
    
    for json_str in matches:
        try:
            # The match from findall is the JSON string itself.
            tool_call = json.loads(json_str.strip())
            
            tool_name = tool_call.get('name', '')
            tool_args = tool_call.get('arguments', {})
            
            print(f"[DEBUG] Executing parsed tool call: {tool_name}({tool_args})")
            
            # Execute the tool
            if tool_name == 'web_search' and 'query' in tool_args:
                result = web_search.invoke(tool_args['query'])
                tool_results.append(f"Web search results:\n{result}")
            elif tool_name == 'knowledge_base_search' and 'query' in tool_args:
                result = knowledge_base_search.invoke(tool_args['query'])
                tool_results.append(f"Knowledge base results:\n{result}")
            else:
                print(f"[DEBUG] Unknown tool or missing arguments: {tool_name}")
                
        except json.JSONDecodeError as e:
            print(f"[DEBUG] Error decoding JSON from tool call: {e}")
            print(f"[DEBUG] Faulty JSON string received: '{json_str.strip()}'")
            continue
        except Exception as e:
            print(f"[DEBUG] Error executing tool call: {e}")
            continue
            
    return tool_results

def process_llm_response_with_tools(messages, config):
    """Process LLM response and handle text-based tool calls"""
    try:
        # Get response from LLM
        response = llm_with_tools.invoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Check for text-based tool calls
        if '<tool_call>' in content:
            print("[DEBUG] Found text-based tool calls, parsing and executing...")
            
            # Parse and execute tool calls
            tool_results = parse_and_execute_text_tool_calls(content)
            
            if tool_results:
                # Create a follow-up message with tool results
                tool_results_text = "\n\n".join(tool_results)
                follow_up_messages = messages + [
                    AIMessage(content=content),
                    HumanMessage(content=f"Based on these search results:\n\n{tool_results_text}\n\nPlease provide a comprehensive answer to the original question.")
                ]
                
                # Get final response
                final_response = llm.invoke(follow_up_messages)
                return final_response
        
        return response
        
    except Exception as e:
        print(f"[DEBUG] Error in LLM response processing: {e}")
        return AIMessage(content=f"I encountered an error while processing your request: {e}")


# Graph nodes - UPDATED to handle text-based tool calls
def chatbot(state: State):
    messages = state["messages"]
    config = {}  # We'll handle config separately if needed
    
    # Use our custom processing function
    response = process_llm_response_with_tools(messages, config)
    return {"messages": [response]}

# Graph setup
memory = MemorySaver()
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile(checkpointer=memory)


# ==============================================================================
# 5. STREAMING CHATBOT EXECUTION - ENHANCED DEBUGGING
# ==============================================================================

def stream_graph_updates(user_input: str, config: dict):
    """Stream the graph execution with real-time output and enhanced debugging"""
    snapshot = graph.get_state(config)
    is_new_conversation = not snapshot or not snapshot.values.get("messages")

    messages_to_add = []
    if is_new_conversation:
        print("   [SYSTEM: Initializing conversation...]")
        messages_to_add.append(SystemMessage(content=SYSTEM_PROMPT))
    messages_to_add.append(HumanMessage(content=user_input))

    try:
        print("Assistant: ", end="", flush=True)
        
        # For now, let's use direct processing instead of streaming due to tool call complexity
        print("[Processing request...]")
        
        # Use our custom processing function directly
        response = process_llm_response_with_tools(messages_to_add, config)
        
        if response and hasattr(response, 'content') and response.content:
            # Clean up the response content by removing tool call tags
            clean_content = re.sub(r'<tool_call>.*?</tool_call>', '', response.content, flags=re.DOTALL)
            clean_content = clean_content.strip()
            
            if clean_content:
                print(f"\n{clean_content}")
            else:
                print("\nI've processed your request using the available tools.")
        else:
            print("\nI apologize, but I was unable to generate a response. Please try rephrasing your question.")

    except Exception as e:
        print(f"\n[ERROR] An error occurred while processing your request: {e}")
        print(f"[DEBUG] Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()


def check_api_keys():
    """Check if required API keys are set"""
    print("Checking API keys and imports...")
    
    if TAVILY_IMPORT == "new":
        print("✓ Using new langchain-tavily package (recommended)")
    elif TAVILY_IMPORT == "community":
        print("⚠ Using langchain_community.tools.tavily_search (deprecated)")
        print("  Consider upgrading to: pip install langchain-tavily")
    else:
        print("⚠ No Tavily package found - using DuckDuckGo fallback")
    
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        print("✓ TAVILY_API_KEY is set")
        if TAVILY_IMPORT:
            print("  Web search will use Tavily API")
    else:
        print("⚠ TAVILY_API_KEY is not set")
        if TAVILY_IMPORT:
            print("  Get your API key from: https://tavily.com")
            print("  Set it with: export TAVILY_API_KEY=your_key_here (Linux/Mac)")
            print("  Or: set TAVILY_API_KEY=your_key_here (Windows)")
        print("  Web search will use DuckDuckGo fallback (no API key required)")
    
    # Check if requests is available for fallback
    try:
        import requests
        print("✓ Requests library available for fallback search")
    except ImportError:
        print("⚠ Requests library not available - install with: pip install requests")


def run_chatbot():
    """Main chat loop with enhanced streaming and error handling"""
    print("\nXenon AI Assistant - RAG-Enhanced Chatbot")
    print("=" * 50)
    print("Ready to answer questions using local documents and web search.")
    print("\nCommands: 'quit', 'new', 'help', 'debug', 'test-tools', 'test-simple', 'test-direct'")
    print()

    session_id = str(uuid.uuid4())
    thread_id = "1"

    print(f"Starting conversation in thread: {thread_id}")
    print()

    while True:
        try:
            user_input = input("User: ").strip()
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            elif user_input.lower() == "new":
                thread_id = str(int(thread_id) + 1)
                print(f"\nSwitched to new thread: {thread_id}")
                continue
            elif user_input.lower() == "test-direct":
                print("\n=== TESTING DIRECT TOOL CALL PARSING ===")
                # Test the parsing function directly
                test_content = '<tool_call>{"name": "web_search", "arguments": {"query": "weather in Ankara today"}}</tool_call>'
                print(f"Testing with content: {test_content}")
                
                results = parse_and_execute_text_tool_calls(test_content)
                if results:
                    print("✓ Tool call parsing and execution successful!")
                    for result in results:
                        print(f"Result: {result[:100]}...")
                else:
                    print("✗ Tool call parsing failed")
                print("========================================\n")
                continue
            elif user_input.lower() == "test-simple":
                print("\n=== TESTING SIMPLE LLM TOOL CALL ===")
                try:
                    # Test with a very explicit message
                    test_messages = [
                        SystemMessage(content="You are a helpful assistant with access to web_search and knowledge_base_search tools. Use tools when needed."),
                        HumanMessage(content="Use the web_search tool to find the weather in Ankara today.")
                    ]
                    
                    print("Sending explicit tool request to LLM...")
                    response = llm_with_tools.invoke(test_messages)
                    
                    if hasattr(response, 'tool_calls') and response.tool_calls:
                        print(f"✓ LLM made tool calls: {len(response.tool_calls)}")
                        for call in response.tool_calls:
                            print(f"  Tool: {call.get('name')}, Args: {call.get('args')}")
                    else:
                        print("✗ LLM did not make tool calls")
                        print(f"Response content: {response.content}")
                        
                        # Check if the model supports function calling at all
                        print("\nChecking if model supports function calling...")
                        simple_response = llm.invoke([HumanMessage(content="Hello")])
                        print(f"Simple response type: {type(simple_response)}")
                        print(f"Simple response attributes: {[attr for attr in dir(simple_response) if not attr.startswith('_')]}")
                    
                except Exception as e:
                    print(f"✗ Simple tool test failed: {e}")
                    import traceback
                    traceback.print_exc()
                print("=====================================\n")
                continue
            elif user_input.lower() == "test-tools":
                print("\n=== TESTING TOOLS ===")
                print("Testing web search tool...")
                try:
                    result = web_search.invoke("weather ankara")
                    print("✓ Web search tool works:")
                    print(result[:200] + "..." if len(result) > 200 else result)
                except Exception as e:
                    print(f"✗ Web search tool failed: {e}")
                
                print("\nTesting knowledge base search tool...")
                try:
                    result = knowledge_base_search.invoke("test query")
                    print("✓ Knowledge base search tool works:")
                    print(result[:200] + "..." if len(result) > 200 else result)
                except Exception as e:
                    print(f"✗ Knowledge base search tool failed: {e}")
                
                print("\nTesting LLM with tools binding...")
                try:
                    # Test if tools are properly bound
                    test_message = HumanMessage(content="What is the weather like today?")
                    response = llm_with_tools.invoke([test_message])
                    
                    if hasattr(response, 'tool_calls') and response.tool_calls:
                        print(f"✓ LLM can make tool calls: {len(response.tool_calls)} calls detected")
                        for call in response.tool_calls:
                            print(f"  - Tool: {call.get('name', 'Unknown')}")
                    else:
                        print("⚠ LLM responded but made no tool calls")
                        print(f"Response: {response.content[:100]}...")
                        
                        # Let's test tool binding more directly
                        print("\nTesting tool binding...")
                        bound_tools = llm_with_tools.bound
                        print(f"Tools bound to LLM: {len(bound_tools) if bound_tools else 0}")
                        
                except Exception as e:
                    print(f"✗ LLM tool binding test failed: {e}")
                    print("Note: You may need to restart the script to rebind tools properly")
                print("====================\n")
                continue
            elif user_input.lower() == "debug":
                print("\n=== DEBUG INFORMATION ===")
                check_api_keys()
                print(f"Knowledge base directory: {DATA_DIR}")
                print(f"Vector store directory: {PERSIST_DIRECTORY}")
                print(f"RAG loader status: {'Loaded' if rag_loader.is_loaded else 'Not loaded'}")
                print(f"Current thread: {thread_id}")
                print("Available tools:", [tool.name for tool in tools])
                print("========================\n")
                continue
            elif user_input.lower() == "help":
                print("\nCommands:")
                print("- 'quit' or 'exit': Exit the chatbot")
                print("- 'new': Start a new conversation thread")
                print("- 'debug': Show debugging information")
                print("- 'test-tools': Test tool functionality directly")
                print("- 'test-direct': Test tool call parsing directly")
                print("- 'test-simple': Test LLM tool calling with explicit request")
                print("- 'help': Show this help message")
                print("- Ask any question to get started!")
                continue
            elif not user_input:
                continue

            config = {"configurable": {"thread_id": f"{session_id}-{thread_id}"}}
            stream_graph_updates(user_input, config)
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred: {e}")
            print("Please try again or type 'quit' to exit.")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Initialize the system
    try:
        print("Initializing Xenon AI Assistant...")
        
        # Check API keys first
        check_api_keys()
        
        print("Checking knowledge base...")
        
        # Pre-load the RAG system to catch any issues early
        rag_loader.load_and_index_documents()
        
        print("System ready!")
        run_chatbot()
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize system: {e}")
        print("Please check your configuration and try again.")
        import traceback
        traceback.print_exc()
        sys.exit(1)