from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_firestore import FirestoreChatMessageHistory
from google.cloud import firestore


PROJECT_ID = ""
SESSION_ID = "user1_session"
COLLECTION_NAME = "chat_history"


load_dotenv()

