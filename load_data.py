import os
import key_param
from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from github_data_extractor import get_all_files
from langchain.docstore.document import Document 


print("Starting data loading and indexing process...")
try:
    client = MongoClient(key_param.MONGO_URI)
    client.admin.command('ping')  # Verify connection
    print("MongoDB connection successful.")
    dbname = "Github_Rag"
    collectionName = "FirstTrial"
    collection = client[dbname][collectionName]
    print(f"Connected to collection: {dbname}.{collectionName}")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit()


repo_owner = "feder-cr"
repo_name = "Jobs_Applier_AI_Agent_AIHawk"
try:
    print("Fetching documents from GitHub repository...")
    all_files = get_all_files(repo_owner, repo_name)
    print(f"Fetched {len(all_files)} files from GitHub repository.")

   
    documents_for_vectorstore = []
   
    all_directories = set()

    for file_path, content in all_files.items():
        
        if isinstance(content, str) and content.strip():
            directory_path = os.path.dirname(file_path)
            file_name_with_ext = os.path.basename(file_path)
            file_base, file_extension = os.path.splitext(file_name_with_ext)

            
            if directory_path:
                depth = directory_path.count(os.sep) + 1 
                all_directories.add(directory_path) 
            else:
                depth = 0 # Root level 
                directory_path = '.' 


            documents_for_vectorstore.append(
                Document(
                    page_content=content,
                    metadata={
                        'source': file_path,         
                        'file_name': file_name_with_ext, 
                        'file_extension': file_extension, 
                        'directory': directory_path, 
                        'depth': depth               
                        
                    }
                )
            )
            

        else:
            # Skip files that are not text or are empty
            print(f"Skipping non-text or empty file: {file_path}")

    # This check should be AFTER the loop finishes processing all files
    if not documents_for_vectorstore:
        print("Error: No valid documents found after processing files.")
        exit()

    print(f"Identified {len(all_directories)} unique directories.")


except Exception as e:
    print(f"Error fetching or processing GitHub files: {e}")
    exit()

# Split documents into chunks
try:
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # Split the list of Documents
    documents = text_splitter.split_documents(documents_for_vectorstore)
    print(f"Split documents into {len(documents)} chunks.")

    if not documents:
        print("Error: No document chunks to index after splitting.")
        exit()

except Exception as e:
    print(f"Error splitting documents: {e}")
    exit()

# Load embedding model
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

# Build the vector store with MongoDB Atlas
try:
    print("Initializing MongoDB Atlas Vector Search and adding documents...")
 
    vectorStore = MongoDBAtlasVectorSearch.from_documents(
        documents,
        embeddings,
        collection=collection,
        index_name="vector_index"
    )
    print(f"Successfully added {len(documents)} chunks to MongoDB Atlas collection '{dbname}.{collectionName}' and indexed.")
except Exception as e:
    print(f"Error during MongoDBAtlasVectorSearch.from_documents: {e}")
    print("Please ensure:")
    print("- Your MongoDB Atlas cluster is running.")
    print("- The collection exists.")
    print("- The 'vector_index' Atlas Search index is correctly configured on the collection, targeting the 'embedding' field.")
    print("- Your MongoDB user has read/write permissions on the database/collection.")
    exit()

print("Data loading and indexing script finished.")
