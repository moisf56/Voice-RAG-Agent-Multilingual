import os
import uuid
from datetime import datetime
from typing import Annotated
from typing_extensions import TypedDict

# --- LangChain/LangGraph Imports ---
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.tools import tool # Note: Using the decorator for our custom tool

# --- LLM and Tool Imports ---
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch # Updated Tavily import
from langchain.tools import Tool # For wrapping tools if needed, but decorator is cleaner

# --- RAG (Knowledge Base) Imports ---
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ==============================================================================
# 1. RAG KNOWLEDGE BASE SETUP
# ==============================================================================

# Define paths for the knowledge base
PERSIST_DIRECTORY = "./chroma_db_rag"  # Folder to store the vector database
DATA_DIR = "./knowledge_base"      # Folder where you put your .txt and .pdf files

def load_and_index_documents():
    """
    Loads documents from the DATA_DIR, splits them into chunks, embeds them
    using a multilingual model, and stores them in a persistent ChromaDB vector store.
    """
    print("🚀 Initializing RAG knowledge base...")

    # Check if vector store already exists
    if os.path.exists(PERSIST_DIRECTORY):
        print("📁 Found existing vector store, loading...")
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
        vectorstore = Chroma(
            embedding_function=embeddings,
            collection_name="rag_collection",
            persist_directory=PERSIST_DIRECTORY
        )
        print("✅ Loaded existing vector store.")
        return vectorstore.as_retriever(search_kwargs={"k": 5})

    # Configure the loader to handle both .pdf and .txt files (with UTF-8 for Arabic)
    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*", # Load all files in all subdirectories
        loader_cls=lambda path: (PyPDFLoader(path) if path.endswith(".pdf") else TextLoader(path, encoding='utf-8')),
        show_progress=True,
        use_multithreading=True
    )
    
    docs = loader.load()
    
    if not docs:
        print("⚠️ No documents found in the 'knowledge_base' folder. The RAG tool will not have any information.")
        # Create an empty retriever if no docs are found so the app doesn't crash
        # This uses a dummy embedding function for initialization
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
        return Chroma(embedding_function=embeddings, 
                      collection_name="rag_collection", 
                      persist_directory=PERSIST_DIRECTORY).as_retriever()

    print(f"✅ Loaded {len(docs)} document(s).")

    # Split documents into smaller chunks suitable for RAG
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Use a powerful, free, and multilingual embedding model.
    # BAAI/bge-small-en-v1.5 is excellent for this.
    print("🧠 Loading embedding model (E5-small)...")
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    print("✅ Embedding model loaded.")

    # Create and persist the Chroma vector store
    print("💾 Creating and persisting vector store in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        collection_name="rag_collection",
        persist_directory=PERSIST_DIRECTORY
    )
    # Note: No need to call vectorstore.persist() - it's automatic with persist_directory
    print(f"✅ Indexed {len(splits)} chunks into '{PERSIST_DIRECTORY}'.")

    # Return the vector store as a retriever object
    return vectorstore.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 most relevant chunks

# Initialize the retriever when the script starts
knowledge_base_retriever = load_and_index_documents()

# ==============================================================================
# 2. TOOL DEFINITION
# ==============================================================================

# Define the custom RAG tool with a clear docstring for the LLM
@tool
def knowledge_base_search(query: str) -> str:
    """
    Use this tool to find information from the internal knowledge base.
    This is your primary source for specific company policies, internal project details,
    or any information contained within the local PDF and TXT files.
    Use this for questions that cannot be answered with a general internet search.
    The user's original question should be the input for this tool.
    """
    print(f"Tool: knowledge_base_search, Query: {query}")
    retrieved_docs = knowledge_base_retriever.get_relevant_documents(query)
    
    if not retrieved_docs:
        return "No relevant information was found in the knowledge base for this query."

    # Format the retrieved documents for the LLM
    docs_content = "\n\n---\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}" for doc in retrieved_docs])
    return f"The following information was retrieved from the knowledge base:\n\n{docs_content}"

# Define the web search tool
web_search_tool = TavilySearch(max_results=3)

# Create the final list of tools for the agent
tools = [web_search_tool, knowledge_base_search]


# ==============================================================================
# 3. LLM AND GRAPH SETUP
# ==============================================================================

# Update the System Prompt to be aware of the new tool and search strategy
SYSTEM_PROMPT = f"""You are a professional, friendly, helpful, and factual AI assistant. Your name original name is Qwen, but your current and commercial name is Xenon3.

**Your Primary Directives:**
1.  **Search Strategy:** You have two tools at your disposal. You must decide which one to use based on the user's query:
    *   For general, real-time information (e.g., current news, weather, public facts), use the `tavily_search_results_json` tool.
    *   For specific, internal information (e.g., company policies, project details from our documents), use the `knowledge_base_search` tool.
2.  **NEVER Fabricate:** If a tool returns no relevant information, you MUST state that you could not find the information. Do not invent details.
3.  **Synthesize, Don't Recite:** Do not just list the tool results. Synthesize the information into a clear, coherent answer.
4.  **Acknowledge Time:** When providing time-sensitive information, state the current date. The current date is {datetime.now().strftime('%A, %B %d, %Y')}.
5.  **Maintain Professionalism:** Always be polite, friendly, and respectful.
6.  **Language:** You must respond to the user in the same language/s they use. If they ask in Arabic, your final answer must be in Arabic.

You will now begin the conversation with the user.
"""

# vLLM Server Setup
BASE_URL = "https://zur81ba7zypznm-8000.proxy.runpod.net/v1"        # Replace with the new pod's ID :)
llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key="EMPTY",
    model="Qwen32",
    temperature=0.7,
    max_tokens=16384, # Adjusted for better performance
    timeout=120
)
llm_with_tools = llm.bind_tools(tools)

# Define the graph state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Define the graph nodes
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# LangGraph Setup
memory = MemorySaver()
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile(checkpointer=memory)


# ==============================================================================
# 4. CHATBOT EXECUTION LOGIC
# ==============================================================================

def stream_graph_updates(user_input: str, config: dict):
    """
    Injects the system prompt if needed and streams the graph to get the final answer.
    """
    snapshot = graph.get_state(config)
    is_new_conversation = not snapshot or not snapshot.values.get("messages")

    messages_to_add = []
    if is_new_conversation:
        print("   [SYSTEM: Injecting system prompt and tool definitions into memory...]")
        messages_to_add.append(SystemMessage(content=SYSTEM_PROMPT))
    messages_to_add.append(HumanMessage(content=user_input))

    try:
        final_state = None
        for event in graph.stream({"messages": messages_to_add}, config, stream_mode="values"):
            final_state = event

        last_message = final_state["messages"][-1] if final_state and final_state["messages"] else None

        if isinstance(last_message, AIMessage) and last_message.content:
            print("Assistant:", last_message.content)
        else:
            print("Assistant: I apologize, but I was unable to generate a final response. Please try rephrasing your question.")

    except Exception as e:
        print(f"Assistant: An error occurred while processing your request: {e}")


def run_chatbot():
    """Main chat loop with memory, web search, and RAG capabilities."""
    print("\n🧠 LangGraph Chatbot with RAG Knowledge Base")
    print("========================================================")
    print("✅ Ready to answer questions from the web or your local documents.")
    print("\nCommands: 'quit', 'new'")
    print()

    session_id = str(uuid.uuid4())
    thread_id = "1"

    print(f"🗣️ Starting conversation in thread: {thread_id} (Session: {session_id})")
    print()

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            elif user_input.lower() == "new":
                thread_id = str(int(thread_id) + 1)
                print(f"\n🗣️ Switched to new thread: {thread_id} (Session: {session_id})")
                continue

            config = {"configurable": {"thread_id": f"{session_id}-{thread_id}"}}
            stream_graph_updates(user_input, config)
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_chatbot()