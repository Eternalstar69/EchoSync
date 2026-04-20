import os
import glob
import chromadb
from chromadb.utils import embedding_functions

# --- CONFIG ---
# Using relative paths so it works out-of-the-box 
DB_PATH = "./touhou_vectordb"
COLLECTION_NAME = "touhou_lyrics_vault"
SOURCE_FOLDER = "./lyrics_pool"

# Use Default Embedding Function (Ensure this matches the main Agent's EF)
default_ef = embedding_functions.DefaultEmbeddingFunction()

def run_bulk_import():
    # 0. Pre-flight check: Create source folder if it doesn't exist
    if not os.path.exists(SOURCE_FOLDER):
        os.makedirs(SOURCE_FOLDER)
        print(f"[!] Created '{SOURCE_FOLDER}' folder.")
        print(f"[!] Please place your '.txt' files (Format: Artist - Title.txt) in '{SOURCE_FOLDER}' and run again.")
        return

    # 1. Connect to ChromaDB (Persistent storage)
    # This will create the DB folder automatically if it doesn't exist
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Get or create the collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, 
        embedding_function=default_ef
    )

    # 2. Find all .txt files in the lyrics_pool folder
    files = glob.glob(os.path.join(SOURCE_FOLDER, "*.txt"))
    
    if not files:
        print(f"[?] No .txt files found in '{SOURCE_FOLDER}'. Nothing to import.")
        return

    print(f"[INIT] Found {len(files)} files to process...")

    for file_path in files:
        # Get filename and remove extension
        filename = os.path.basename(file_path).replace(".txt", "")
        
        try:
            # Split Circle and Title using " - " delimiter (Standard Convention)
            if " - " in filename:
                artist_raw, title_raw = filename.split(" - ", 1)
            else:
                # Fallback if the filename doesn't follow the "Artist - Title" format
                artist_raw, title_raw = "Unknown", filename
            
            artist = artist_raw.strip()
            title = title_raw.strip()
                
            # Read the lyrics content from file with UTF-8 to support Japanese/Thai characters
            with open(file_path, "r", encoding="utf-8") as f:
                lyrics_content = f.read()

            # Create a unique ID (Normalization for ChromaDB IDs to avoid illegal characters)
            # Replacing spaces and slashes with underscores
            song_id = f"{artist}_{title}".replace(" ", "_").replace("/", "_").replace("\\", "_")

            # 3. UPSERT with Unified Metadata Logic
            # We save the "title" metadata as "Artist - Song" so the 
            # Agent's reconciliation logic can parse it correctly.
            collection.upsert(
                documents=[lyrics_content],
                metadatas=[{
                    "title": f"{artist} - {title}",  # CRITICAL: Agent needs this exact format
                    "artist": artist,               # Used for the 'where' metadata filter
                    "source": "manual_pool"
                }],
                ids=[song_id]
            )
            print(f"  [SUCCESS] Indexed: {artist} - {title}")

        except Exception as e:
            print(f"  [ERROR] Failed to process {filename}: {e}")

    print(f"\n{'='*50}")
    print("Bulk Import Process Completed Successfully!")
    print(f"Database Location: {os.path.abspath(DB_PATH)}")
    print(f"Total records now in Vault: {collection.count()}")
    print(f"{'='*50}")

if __name__ == "__main__":
    # To perform a completely fresh wipe of the DB, uncomment the lines below:
    # import shutil
    # if os.path.exists(DB_PATH):
    #     shutil.rmtree(DB_PATH)
    #     print("[RESET] Existing database cleared.")
    
    run_bulk_import()