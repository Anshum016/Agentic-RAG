
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool



def setup_structure_agent(llm, collection):
    """
    Sets up and returns the Agent Executor for the Structure Agent.

    Args:
        llm: The Langchain LLM instance (e.g., ChatGoogleGenerativeAI).
        collection: The PyMongo collection object for the indexed data.

    Returns:
        AgentExecutor: The executor for the Structure Agent.
    """

    @tool
    def count_total_files() -> str:
        """
        Counts the total number of unique original files indexed in the repository.
        Use this tool when the user asks for the total count of files.
        Returns a string with the total unique file count.
        """
        print("\n--- Tool: count_total_files called ---")
        try:
            unique_file_names = collection.distinct('file_name', {'file_name': {'$exists': True}})
            file_count = len(unique_file_names)

            print(f"--- Tool: count_total_files result: {file_count} ---")
            return f"Total number of unique original files indexed: {file_count}"
        except Exception as e:
            print(f"--- Tool Error (count_total_files): {e} ---")
            return f"Error counting total unique files: {e}"

    @tool
    def count_total_directories() -> str:
        """
        Counts the total number of unique directories (folders) in the repository, excluding the root directory.
        Use this tool when the user asks for the total count of directories or folders.
        Returns a string with the total directory count.
        """
        print("\n--- Tool: count_total_directories called ---")
        try:
            # This tool assumes 'collection' is accessible from the outer scope
            unique_directories_count = len(collection.distinct('directory', {'directory': {'$ne': '.'}}))
            print(f"--- Tool: count_total_directories result: {unique_directories_count} ---")
            return f"Total number of unique directories indexed (excluding root): {unique_directories_count}"
        except Exception as e:
            print(f"--- Tool Error (count_total_directories): {e} ---")
            return f"Error counting total directories: {e}"

    @tool
    def count_top_level_directories() -> str:
        """
        Counts the number of directories located directly under the root of the repository (top-level directories).
        Use this tool when the user asks for the count of top-level directories.
        Returns a string with the count of top-level directories.
        """
        print("\n--- Tool: count_top_level_directories called ---")
        try:
            # This tool assumes 'collection' is accessible from the outer scope
            top_level_dirs_count = len(collection.distinct('directory', {'depth': 1}))
            print(f"--- Tool: count_top_level_directories result: {top_level_dirs_count} ---")
            return f"Total number of top-level directories indexed: {top_level_dirs_count}"
        except Exception as e:
            print(f"--- Tool Error (count_top_level_directories): {e} ---")
            return f"Error counting top-level directories: {e}"

    @tool
    def count_files_in_directory(directory_path: str) -> str:
        """
        Counts the number of files located directly within a specific directory (not in its subdirectories).
        Use this tool when the user asks for the number of files *in* a named directory.
        Args:
            directory_path (str): The path to the directory (e.g., 'src', 'src/utils').
        Returns:
            str: A string with the count of files directly in the specified directory.
        """
        print(f"\n--- Tool: count_files_in_directory called with path: '{directory_path}' ---")
        try:
            file_count_in_dir = collection.count_documents({'directory': directory_path})
            print(f"--- Tool: count_files_in_directory result for '{directory_path}': {file_count_in_dir} ---")
            return f"Total number of files directly in directory '{directory_path}': {file_count_in_dir}"
        except Exception as e:
            print(f"--- Tool Error (count_files_in_directory): {e} ---")
            return f"Error counting files in directory '{directory_path}': {e}"

    @tool
    def count_subdirectories_in_directory(directory_path: str) -> str:
        """
        Counts the number of immediate subdirectories (folders directly inside) within a specific directory.
        Use this tool when the user asks for the number of folders *inside* a named directory.
        Args:
            directory_path (str): The path to the parent directory (e.g., 'src', '.'). Use '.' for the root directory.
        Returns:
            str: A string with the count of immediate subdirectories within the specified directory.
        """
        print(f"\n--- Tool: count_subdirectories_in_directory called with path: '{directory_path}' ---")
        try:
            if directory_path == '.':
                parent_depth = 0
            else:
                parent_doc = collection.find_one({'directory': directory_path}, {'depth': 1})
                if not parent_doc or 'depth' not in parent_doc:
                     print(f"--- Tool: count_subdirectories_in_directory: Parent directory '{directory_path}' not found or missing depth info. ---")
                     return f"Directory '{directory_path}' not found or has no subdirectories indexed."
                parent_depth = parent_doc['depth']

            target_depth = parent_depth + 1 # We are looking for directories one level deeper

            if directory_path == '.':
                # For root, find directories with depth 1
                query_filter = {'depth': 1}
            else:

                escaped_directory_path = directory_path.replace(".", "\\.")
                regex_pattern = f'^{escaped_directory_path}/' # Pattern like '^src/' or '^\\./' for root
                query_filter = {
                    'directory': {'$regex': regex_pattern},
                    'depth': target_depth # Exactly one level deeper
                }

            immediate_subdirs = collection.distinct('directory', query_filter)

            if directory_path != '.' and directory_path in immediate_subdirs:
                 immediate_subdirs.remove(directory_path)


            subdirectory_count = len(immediate_subdirs)

            print(f"--- Tool: count_subdirectories_in_directory result for '{directory_path}': {subdirectory_count} ---")
            return f"Total number of subdirectories directly in directory '{directory_path}': {subdirectory_count}"

        except Exception as e:
            print(f"--- Tool Error (count_subdirectories_in_directory): {e} ---")
            # Include the full error details for debugging
            return f"Error counting subdirectories in directory '{directory_path}': {e}"


    @tool
    def list_all_directories() -> str:
        """
        Lists all unique directory paths (folders) in the repository, excluding the root directory.
        Use this tool when the user asks to list all directories or folders.
        Returns a string listing all unique directory paths.
        """
        print("\n--- Tool: list_all_directories called ---")
        try:

            all_unique_dirs = sorted(collection.distinct('directory', {'directory': {'$ne': '.'}}))
            print(f"--- Tool: list_all_directories result: {all_unique_dirs[:10]}... ---") # Print snippet if long
            result = f"All unique directories: {', '.join(all_unique_dirs) if all_unique_dirs else 'None'}"
            return result
        except Exception as e:
            print(f"--- Tool Error (list_all_directories): {e} ---")
            return f"Error listing all directories: {e}"

    @tool
    def list_top_level_directories() -> str:
        """
        Lists the names of directories located directly under the root of the repository (top-level directories).
        Use this tool when the user asks to list top-level directories.
        Returns a string listing the top-level directory names.
        """
        print("\n--- Tool: list_top_level_directories called ---")
        try:
            top_level_dirs = sorted(collection.distinct('directory', {'depth': 1}))
            print(f"--- Tool: list_top_level_directories result: {top_level_dirs} ---")
            result = f"Top-level directories: {', '.join(top_level_dirs) if top_level_dirs else 'None'}"
            return result
        except Exception as e:
            print(f"--- Tool Error (list_top_level_directories): {e} ---")
            return f"Error listing top-level directories: {e}"

    @tool
    def list_files_in_directory(directory_path: str) -> str:
        """
        Lists the names of files located directly within a specific directory (not in its subdirectories).
        Use this tool when the user asks to list files *in* a named directory.
        Args:
            directory_path (str): The path to the directory (e.g., 'src', 'src/utils').
        Returns:
            str: A string listing the file names directly in the specified directory.
        """
        print(f"\n--- Tool: list_files_in_directory called with path: '{directory_path}' ---")
        try:
            # This tool assumes 'collection' is accessible from the outer scope
            files_in_dir = list(collection.find({'directory': directory_path}, {'file_name': 1, '_id': 0}))
            file_names = [f['file_name'] for f in files_in_dir if 'file_name' in f] # Access top-level field
            print(f"--- Tool: list_files_in_directory result for '{directory_path}': {file_names[:10]}... ---") # Print snippet
            result = f"Files found directly in directory '{directory_path}': {', '.join(file_names) if file_names else 'None'}"
            return result
        except Exception as e:
            print(f"--- Tool Error (list_files_in_directory): {e} ---")
            return f"Error listing files in directory '{directory_path}': {e}"

    @tool
    def list_subdirectories_in_directory(directory_path: str) -> str:
        """
        Lists the names of immediate subdirectories (folders directly inside) within a specific directory.
        Use this tool when the user asks to list folders *inside* a named directory.
        Args:
            directory_path (str): The path to the parent directory (e.g., 'src', '.'). Use '.' for the root directory.
        Returns:
            str: A string listing the names of immediate subdirectories within the specified directory.
        """
        print(f"\n--- Tool: list_subdirectories_in_directory called with path: '{directory_path}' ---")
        try:
            if directory_path == '.':
                parent_depth = 0
            else:
                parent_doc = collection.find_one({'directory': directory_path}, {'depth': 1})
                if not parent_doc or 'depth' not in parent_doc:
                     print(f"--- Tool: list_subdirectories_in_directory: Parent directory '{directory_path}' not found or missing depth info. ---")
                     return f"Directory '{directory_path}' not found or has no subdirectories indexed."
                parent_depth = parent_doc['depth']

            target_depth = parent_depth + 1 # We are looking for directories one level deeper

            if directory_path == '.':
                 query_filter = {'depth': 1}
            else:

                 escaped_directory_path = directory_path.replace(".", "\\.")
                 regex_pattern = f'^{escaped_directory_path}/' # Pattern like '^src/' or '^\\./' for root
                 query_filter = {
                     'directory': {'$regex': regex_pattern},
                     'depth': target_depth # Exactly one level deeper
                 }


            immediate_subdirs = sorted(collection.distinct('directory', query_filter))

            if directory_path != '.' and directory_path in immediate_subdirs:
                 immediate_subdirs.remove(directory_path)

            print(f"--- Tool: list_subdirectories_in_directory result for '{directory_path}': {immediate_subdirs} ---")
            result = f"Subdirectories directly in directory '{directory_path}': {', '.join(immediate_subdirs) if immediate_subdirs else 'None'}"
            return result

        except Exception as e:
            print(f"--- Tool Error (list_subdirectories_in_directory): {e} ---")
            return f"Error listing subdirectories in directory '{directory_path}': {e}"


    @tool
    def count_files_by_extension() -> str:
        """
        Counts the number of files for each unique file extension in the repository.
        Use this tool when the user asks about file types or extensions and their counts.
        Returns a string listing file extensions and their counts.
        """
        print("\n--- Tool: count_files_by_extension called ---")
        try:
            # This tool assumes 'collection' is accessible from the outer scope
            extension_counts = list(collection.aggregate([
                { '$group': { '_id': '$file_extension', 'count': { '$sum': 1 } } }, # Group by top-level field
                { '$sort': { 'count': -1 } }
            ]))
            print(f"--- Tool: count_files_by_extension result: {extension_counts} ---")
            if extension_counts:
                result = "File extension counts:\n" + "\n".join([
                    f"- {item['_id'] if item['_id'] else 'None'}: {item['count']}" for item in extension_counts
                ])
            else:
                result = "No file extension data found."
            return result
        except Exception as e:
            print(f"--- Tool Error (count_files_by_extension): {e} ---")
            return f"Error counting files by extension: {e}"


    # List of tools available to the Structure Agent
    structure_tools = [
        count_total_files,
        count_total_directories,
        count_top_level_directories,
        count_files_in_directory,
        count_subdirectories_in_directory, # Add the new tool here
        list_all_directories,
        list_top_level_directories,
        list_files_in_directory,
        list_subdirectories_in_directory, # Adding a list version too for completeness
        count_files_by_extension
    ]

    # System Prompt for the Structure Agent
    structure_system_prompt = """
    You are a specialized AI assistant whose sole purpose is to provide factual information about the file and directory structure of the GitHub repository data you have access to.
    You are an expert in repository organization, file types, and directory hierarchies.
    You have access to a set of tools designed to query the repository's structure. Use these tools for ALL questions about the repository's structure.

    Available Tools:
    - `count_total_files`: Counts the total number of unique files.
    - `count_total_directories`: Counts the total number of unique directories (excluding root).
    - `count_top_level_directories`: Counts directories directly under the root.
    - `count_files_in_directory`: Counts files *directly* within a specified directory. Takes `directory_path` as input.
    - `count_subdirectories_in_directory`: Counts the number of immediate subdirectories *inside* a specified directory. Takes `directory_path` as input. Use '.' for the root directory.
    - `list_all_directories`: Lists all unique directory paths (excluding root).
    - `list_top_level_directories`: Lists directories directly under the root.
    - `list_files_in_directory`: Lists files *directly* within a specified directory. Takes `directory_path` as input.
    - `list_subdirectories_in_directory`: Lists the names of immediate subdirectories *inside* a specified directory. Takes `directory_path` as input. Use '.' for the root directory.
    - `count_files_by_extension`: Counts files by their extension.

    When the user asks a question about the repository's structure:
    - Carefully analyze the user's request to determine which specific structural tool is needed.
    - Pay close attention to whether the user is asking about **files** vs. **directories/folders**, and whether they are asking about the **total** count/list, **top-level**, or items **inside a specific directory**.
    - If the request involves a specific directory name, make sure to extract the correct directory path to pass to the tool.
    - Use the appropriate tool with the necessary arguments.
    - The tool will return factual data about the structure (counts, lists, etc.).
    - Format the tool's output into a clear and concise natural language answer for the user.
    - If the tool indicates it could not perform the request or found no relevant data, inform the user based on the tool's output.
    - Do NOT attempt to answer questions about the *content* of the files (code, documentation, etc.). Your expertise is limited to the structure.
    - Do NOT make up information.

    Begin!
    """

    #prompt template for the Structure Agent
    structure_prompt = ChatPromptTemplate.from_messages([
        ("system", structure_system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])


    structure_agent = create_tool_calling_agent(llm, structure_tools, structure_prompt)


    structure_agent_executor = AgentExecutor(agent=structure_agent, tools=structure_tools, verbose=True)

    print("Structure Agent initialized.")

    return structure_agent_executor
