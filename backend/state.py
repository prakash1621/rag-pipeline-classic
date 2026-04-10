from threading import Lock

vectorstore = None        # holds the loaded FAISS/Pinecone vectorstore
vectorstore_lock = Lock() # guards concurrent access during rebuild/clear
