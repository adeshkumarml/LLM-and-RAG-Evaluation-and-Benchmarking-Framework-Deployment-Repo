from src.utils.utils import config_loader
from sentence_transformers import SentenceTransformer

config = config_loader()
embedding_model = None

def get_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer(config["embedding_model"])
    return embedding_model

def generate_embeddings(texts):
    embedding_model = get_model()
    embeddings = embedding_model.encode(texts, convert_to_numpy = True)
    return embeddings