import os
import json 
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool 
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.output_parsers import StrOutputParser 
import gradio as gr
from gradio.themes.base import Base
import key_param
import warnings
from structure_agent import setup_structure_agent
from content_agent import setup_content_agent


warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub.file_download')

print("Initializing Gradio App...")

# MongoDB Connection 
try:
    client = MongoClient(key_param.MONGO_URI)
    client.admin.command('ping') 
    db_name = "Github_Rag"
    collection_name = "FirstTrial"
    namespace = f"{db_name}.{collection_name}"
    collection = client[db_name][collection_name]
    print(f"MongoDB connection successful to namespace: {namespace}")
    print(f"Checking total document count directly in collection '{collection_name}' on startup: {collection.count_documents({})}")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit() 

# Embeddings 
try:
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        # model_kwargs={"pooling": "mean"}
    )
    print("Embedding model loaded.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    exit()

#Vector Store Initialization
vector_search_index_name = "vector_index"

try:
    print(f"Connecting to Atlas Vector Store (Namespace: {namespace}, Index: {vector_search_index_name})...")
    # Initialize the vector store instance
    vector_store = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=key_param.MONGO_URI,
        namespace=namespace,
        embedding=embeddings,
        index_name=vector_search_index_name
    )
    print("Connected to Vector Store successfully.")
except Exception as e:
    print(f"Error connecting to Vector Store: {e}")
    print("Check: Connection string, namespace, index name, and if the index is active in Atlas.")
    exit()

# Gemini model
try:
    genai.configure(api_key=key_param.gemini_api_key)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=key_param.gemini_api_key,
                                 convert_system_message_to_human=True, temperature=0) 
    print("Gemini model wrapper for LangChain initialized.")
except Exception as e:
    print(f"Error initializing Gemini model: {e}")
    exit()

# Setup Specialized Agents 
# Call the setup functions from the separate files to get the agent executors
try:
    print("Setting up specialized agents...")
   
    structure_agent_executor = setup_structure_agent(llm, collection)
    # Pass the initialized llm and vector_store to the content agent setup
    content_agent_executor = setup_content_agent(llm, vector_store)
    print("Specialized agents setup complete.")
except Exception as e:
    print(f"Error setting up specialized agents: {e}")
    # Print the full traceback for better debugging
    import traceback
    traceback.print_exc()
    exit()


# Define Tools for the Router Agent


@tool
def structure_agent_tool(query: str) -> str:
    """
    Use this tool for questions about the repository's file and directory structure.
    This includes questions about file counts, directory counts, listing files in directories, listing directories, and file extensions.
    Input should be the user's question about the repository structure.
    """
    print(f"\n--- Router Agent calling Structure Agent with query: '{query}' ---")
    # Invoke the Structure Agent Executor with the user's query
    
    try:
        response = structure_agent_executor.invoke({"input": query})
        # Return the output from the Structure Agent
        return response.get('output', 'Structure Agent failed to produce output.')
    except Exception as e:
        print(f"--- Error invoking Structure Agent: {e} ---")
        return f"Error routing query to Structure Agent: {e}"


@tool
def content_agent_tool(query: str) -> str:
    """
    Use this tool for questions about the repository's content, code, documentation, technologies used, explanations, or general summaries of what the repository contains.
    Input should be the user's question about the repository content.
    """
    print(f"\n--- Router Agent calling Content Agent with query: '{query}' ---")
    
    try:
        response = content_agent_executor.invoke({"input": query})
        return response.get('output', 'Content Agent failed to produce output.')
    except Exception as e:
        print(f"--- Error invoking Content Agent: {e} ---")
        return f"Error routing query to Content Agent: {e}"


# List of tools available to the Router Agent
router_tools = [structure_agent_tool, content_agent_tool]


# Router Agent 

# System Prompt
router_system_prompt = """
You are a Router AI assistant. Your primary role is to analyze the user's query and route it to the most appropriate specialized agent.
You have access to two tools, each representing a specialized agent:
1. `structure_agent_tool`: Use this for any question related to the repository's file and directory structure (counts, lists of files/folders, file types, etc.).
2. `content_agent_tool`: Use this for any question related to the repository's content, code, documentation, technologies, explanations, or general summaries of what the repository contains.

Analyze the user's query carefully:
- If the query is about the *organization* or *layout* of the repository (files, folders, counts, lists, extensions, hierarchy), use the `structure_agent_tool`.
- If the query is about the *information contained within* the files (what the code does, explanations, technologies, summaries, specific details, concepts), use the `content_agent_tool`.
- Pass the user's *original query* directly to the chosen tool.
- **Your final response to the user MUST be ONLY the exact text output you receive from the tool you call.**
- **DO NOT add any introductory phrases, explanations about routing, or concluding remarks.**
- **JUST output the tool's result.**
- Do NOT try to answer the question yourself. Your only job is to route the query and return the result from the routed agent.
- Do NOT make up tool calls or parameters.

Begin!
"""

# prompt template 
router_prompt = ChatPromptTemplate.from_messages([
    ("system", router_system_prompt),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"), # Placeholder for agent's internal thoughts/tool use
])

# Router Agent
router_agent = create_tool_calling_agent(llm, router_tools, router_prompt)

# Create the Agent Executor for the Router Agent
router_agent_executor = AgentExecutor(agent=router_agent, tools=router_tools, verbose=True) # Set verbose=True to see agent's thinking process

print("Router Agent initialized.")


#Query Function
def query_data(query):
    print(f"\nReceived query: {query}")
    if not query:
        # When no query, return only one value to match outputs=[output2]
        return "Please enter a question about the repository content or structure."

    try:
        # Use the Router Agent Executor to process the query
        print("Running Router Agent Executor...")
        agent_response = router_agent_executor.invoke({"input": query})
        raw_output = agent_response.get('output', 'Router Agent failed to produce an output.')
        print(f"--- Raw Router Agent Output: ---\n{raw_output}\n---")

        final_answer = raw_output

        if raw_output.strip().startswith('```text'):
            # Remove the ```text\n and ``` wrapping
            cleaned_output = raw_output.strip()[len('```text\n'):]
            if cleaned_output.endswith('```'):
                    cleaned_output = cleaned_output[:-len('```')].strip()

            # Attempt to parse the cleaned output as JSON
            try:
                json_output = json.loads(cleaned_output)
                # If it's a JSON object with an 'output' key, use that value
                if isinstance(json_output, dict) and 'output' in json_output:
                    final_answer = json_output['output']
                    print("--- Successfully parsed JSON output ---")
                else:
                    # If it's JSON but not the expected format, use the cleaned text
                    final_answer = cleaned_output
                    print("--- Parsed JSON but not expected format, using cleaned text ---")
            except json.JSONDecodeError:
                # If it's not JSON, just use the cleaned text
                final_answer = cleaned_output
                print("--- Output was text block but not JSON, using cleaned text ---")
        # --- End Enhanced Output Parsing ---

        # Remove the first return value - ONLY return the final answer
        print(f"Agent generated final answer (after parsing):\n---\n{final_answer}\n---")
        return final_answer # <--- CHANGE THIS LINE

    except Exception as e:
        print(f"Error during router agent execution: {e}")
        # Remove the first return value - ONLY return the error message
        return f"An error occurred during router agent processing: {e}" # <--- CHANGE THIS LINE
# Gradio 
with gr.Blocks(theme=Base(), title="Talk with your Git-hub Repo Agent") as demo:
    gr.Markdown(
        """
        # Talk with your Git-hub Repo Agent
        Ask questions about the content *or* structure of your indexed GitHub repository.
        The AI agent can search content, look up structure (files, folders, counts), and summarize.
        """
    )
    textbox = gr.Textbox(label="Enter your Question about the Repository:")
    with gr.Row():
        button = gr.Button("Submit", variant="primary")
    with gr.Column():
        output2 = gr.Textbox(lines=5, max_lines=15, label="Agent's Answer (Synthesized from Tools)")

    # Connect button click to query_data function
    button.click(query_data, inputs=[textbox], outputs=[ output2])

print("Launching Gradio interface...")

demo.launch(debug=True)

