
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool


def setup_content_agent(llm, vector_store):
    """
    Sets up and returns the Agent Executor for the Content Agent.

    Args:
        llm: The Langchain LLM instance (e.g., ChatGoogleGenerativeAI).
        vector_store: The Langchain MongoDBAtlasVectorSearch instance.

    Returns:
        AgentExecutor: The executor for the Content Agent.
    """

    @tool
    def content_search(query: str) -> str:
        """
        Searches the repository document chunks for content relevant to the user's query using vector similarity search.
        Use this tool when the user asks a specific question about content, code, documentation, concepts, or details within files.
        Input is the user's specific question about the content.
        Returns a string containing the retrieved relevant document chunks, including their source file paths.
        """
        print(f"\n--- Tool: content_search called by Content Agent with query: '{query}' ---")
        try:
            docs = vector_store.similarity_search(query, k=5)
            if docs:

                formatted_docs = "\n---\n".join([
                    # Accessing top-level 'source' field directly as observed in Atlas
                    f"Source: {d.metadata.get('source', 'N/A')}\nContent:\n{d.page_content}"
                    for d in docs
                ])
                print(f"--- Tool: content_search found {len(docs)} documents. ---")
                return formatted_docs
            else:
                print("--- Tool: content_search found no relevant document chunks. ---")
                return "No relevant document chunks found."
        except Exception as e:
            print(f"--- Tool Error (content_search): {e} ---")
            # Include the full error details for debugging
            return f"Error during content search: {e}"

    @tool
    def summarize_repo_content() -> str:
        """
        Retrieves a selection of document chunks from the repository to provide context for a general summary.
        Use this tool when the user asks for a general overview or 'jist' of what the repository contains.
        It does not take a specific query, but retrieves diverse content.
        Returns a string containing selected document chunks from the repository.
        """
        print("\n--- Tool: summarize_repo_content called ---")
        try:
            broad_query = "overview of the repository content and structure" 
            docs = vector_store.similarity_search(broad_query, k=20) 

            if docs:
                # Format the retrieved documents for the LLM
                formatted_docs = "\n---\n".join([
                    f"Source: {d.metadata.get('source', 'N/A')}\nContent:\n{d.page_content}"
                    for d in docs
                ])
                print(f"--- Tool: summarize_repo_content found {len(docs)} documents for summary. ---")
                return formatted_docs
            else:
                print("--- Tool: summarize_repo_content found no documents for summary. ---")
                return "Could not retrieve content for a general summary."
        except Exception as e:
            print(f"--- Tool Error (summarize_repo_content): {e} ---")
            return f"Error retrieving content for summary: {e}"

    # List of tools available to the Content Agent
    content_tools = [
        content_search,
        summarize_repo_content # Add the new summary tool to the list
    ]


    # System Prompt for the Content Agent
    content_system_prompt = """
    You are a specialized AI assistant whose sole purpose is to analyze and summarize the *content* of the GitHub repository data you have access to.
    You are an expert in understanding code, documentation, and technical concepts found within repository files.
    You have access to a set of tools designed to retrieve relevant content from the repository's indexed data. Use these tools for ALL questions about the repository's content.

    Available Tools:
    1. `content_search`: Use this for specific questions about content, code, documentation, concepts, or details within files. It takes a user query as input.
    2. `summarize_repo_content`: Use this when the user asks for a general overview or 'jist' of what the repository contains. It does NOT take any input.

    When the user asks a question about the repository's content (what the code does, explanations, technologies used, specific details in files, or a general summary):
    - If the user asks for a general summary or overview ("What does this repo have?", "Give me the gist", "Summarize the repository"), use the `summarize_repo_content` tool.
    - For any other specific question about the content, use the `content_search` tool with the user's specific question as the input query.
    - The tool you call will return text snippets from the repository files, which may include code or documentation.
    - **Carefully analyze the retrieved content, especially if it contains code.**
    - **If the user's query asks "how to" do something, or asks for steps, explanations, or implementation details, synthesize a clear, step-by-step answer based on the code and documentation snippets you retrieve.**
    - If the retrieved content does not contain enough information to answer the question, state that you couldn't find relevant information in the repository data.
    - Do NOT attempt to answer questions about the repository's *structure* (file counts, directory lists, etc.). Your expertise is limited to the content.
    - Do NOT make up information or steps that are not supported by the retrieved content.

    Begin!
    """

    # prompt template for the Content Agent
    content_prompt = ChatPromptTemplate.from_messages([
        ("system", content_system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Create the Content Agent
    content_agent = create_tool_calling_agent(llm, content_tools, content_prompt)

    content_agent_executor = AgentExecutor(agent=content_agent, tools=content_tools, verbose=True)

    print("Content Agent initialized.")
    return content_agent_executor
