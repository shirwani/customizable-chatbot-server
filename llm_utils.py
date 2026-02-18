from utils import *
from call_ollama import ask_local_ollama_llama3
from call_deepseek import ask_deepseek_r1
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")

CLIENT_NAME = os.getenv("CLIENT_SITE", "demo")
CLIENT_SITES_LOCATION = os.getenv("CLIENT_SITES_LOCATION", os.path.join(os.path.dirname(__file__), "..", "client_sites"))
CLIENT_PATH = os.path.join(CLIENT_SITES_LOCATION, CLIENT_NAME)

CLIENT_CHROMA_DB_PATH = os.path.join(CLIENT_PATH, "chroma_db")

CLIENT_FAQ_PATH = os.path.join(CLIENT_PATH, "faq")
CLIENT_FAQ_FILE = os.path.join(CLIENT_FAQ_PATH, "faq.txt")

CLIENT_INVENTORY_PATH = os.path.join(CLIENT_PATH, "inventory")
CLIENT_INVENTORY_CSV_FILE = os.path.join(CLIENT_INVENTORY_PATH, "clothes.csv")

CLIENT_METADATA_PATH = os.path.join(CLIENT_PATH, "product_metadata")
CLIENT_FILTER_ON_LIST_FILE = os.path.join(CLIENT_METADATA_PATH, "filter_on_list.txt")
CLIENT_METADATA_FIELDS_LIST = os.path.join(CLIENT_METADATA_PATH, "metadata_fields_list.txt")

CLIENT_VALID_METADATA_VALUES_FILE = os.path.join(CLIENT_METADATA_PATH, "all_valid_metadata_values.json")

CLIENT_SYSTEM_PROMPTS_PATH = os.path.join(CLIENT_PATH, "system_prompts")

client = chromadb.PersistentClient(path=CLIENT_CHROMA_DB_PATH)
products_collection = client.get_or_create_collection("products")
products_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 384-dim embeddings

DEFAULT_LOCAL_MODEL = os.getenv("LOCAL_LLM_MODEL", "deepseek")


# ---------------------------------------------------------------------------
# LLM dispatch & parameter helpers
# ---------------------------------------------------------------------------
def ask_llm(payload: dict = None):
    """Dispatch to the appropriate backend based on model name and return the LLM response.

    Args:
        payload: Dict of parameters expected by the backend caller.
    """

    if payload is None:
        raise ValueError("payload must be provided to ask_llm")

    # If model is explicitly set to "none" or is missing, we treat it as a signal to use the default local model.
    if 'model' not in payload or not payload['model'] or payload['model'].lower() == "none":
        model = DEFAULT_LOCAL_MODEL
    else:
        model = payload["model"]

    # Normalize model name for downstream use
    if model in ("llama", "llama3", "ollama", "local"):
        payload['model'] = "llama3"
        return ask_local_ollama_llama3(payload=payload)
    elif model in {"deepseek", "deepseek-r1", "r1"}:
        payload['model'] = "deepseek"
        return ask_deepseek_r1(payload=payload)
    else:
        raise ValueError(f"Unsupported model specified: {model}")


def generate_params_dict(
        prompt: str,
        temperature: float = None,
        role: str = 'user',
        top_p: float = None,
        max_tokens: int = 2048,
        model: str = None
):
    """Generate a dictionary of parameters for LLM generation calls.

    If ``model`` is None or "none", the default local model will be used.
    """

    # Normalize model so that downstream code never sees the literal string "none".
    model = None if not model or model == "none" else model

    kwargs = {
        "prompt": prompt,
        "role": role,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    # Only include model key if explicitly provided; ask_llm will inject a default otherwise.
    if model is not None:
        kwargs["model"] = model

    return kwargs


def generate_with_single_input(
        prompt: str,
        role: str = 'user',
        top_p: float = None,
        temperature: float = None,
        max_tokens: int = 2048,
        model: str = None,
        **kwargs):
    """
    Generate a response from the LLM using a single input prompt and specified parameters.
    This function constructs a payload for the LLM based on the provided parameters and
    dispatches the request to the appropriate backend using `ask_llm`.
    """

    payload = {
        "messages": [{'role': role, 'content': prompt}],
        "top_p": top_p,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **kwargs
    }

    # Normalize model
    model = None if not model or model == "none" else model
    payload["model"] = model

    response = ask_llm(payload)
    return response


def get_params_for_task(task: str) -> dict:
    """
    Retrieves specific LLM parameters based on the nature of the task.

    This function returns parameter sets optimized for either creative or technical tasks.
    Creative tasks benefit from higher randomness, while technical tasks require more focus and precision.
    A default parameter set is returned for unrecognized task types.

    Parameters:
    - task (str): The nature of the task ('creative' or 'technical').

    Returns:
    - dict: A dictionary containing 'top_p' and 'temperature' settings appropriate for the task.
    """

    # Define the parameter sets for technical and creative tasks
    PARAMETERS_DICT = {
        "TECHNICAL": {'top_p': 0.8, 'temperature': 0.3},
        "CREATIVE":  {"top_p": 0.1, 'temperature': 1.0}
    }

    # Return the corresponding parameter set based on task type
    if task == 'TECHNICAL':
        param_dict = PARAMETERS_DICT['TECHNICAL']
    elif task == 'CREATIVE':
        param_dict = PARAMETERS_DICT['CREATIVE']
    else:
        param_dict = {'top_p': 0.5, 'temperature': 0.5}  # Default parameters for unrecognized task types


    return param_dict



def parse_json_output(llm_output: str) -> dict:
    """
    Parses a string output from an LLM into a JSON object.

    This function attempts to clean and parse a JSON-formatted string produced by an LLM.
    The input string might contain minor formatting issues, such as unnecessary newlines or single quotes
    instead of double quotes. The function attempts to correct such issues before parsing.

    Parameters:
    - llm_output (str): The string output from the LLM that is expected to be in JSON format.

    Returns:
    - dict or None: A dictionary if parsing is successful, or None if the input string cannot be parsed into valid JSON.

    Exception Handling:
    - In case of a JSONDecodeError during parsing, an error message is printed, and the function returns None.
    """
    try:
        # Since the input might be improperly formatted, ensure any single quotes are removed
        llm_output = llm_output.replace("\n", '').replace("'", '').replace("}}", "}").replace("{{", "{")  # Remove any erroneous structures

        # Attempt to parse JSON directly provided it is a properly-structured JSON string
        parsed_json = json.loads(llm_output)
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        return {}


def search_products(query: str, n_results: int = 5):
    """Search the product inventory in the ChromaDB `products` collection.

    Args:
        query: Natural-language search query, e.g. "blue men's jeans for summer".
        n_results: Maximum number of products to return.

    Returns:
        A list of dicts with `id`, `score`, and the stored metadata fields
        (gender, masterCategory, subCategory, articleType, baseColor, season,
        year, usage, productDisplayName).
    """
    if not query or not query.strip():
        return []

    # Create embedding for the query and search the products collection
    embedding = products_embedder.encode([query], convert_to_numpy=True)
    res = products_collection.query(
        query_embeddings=embedding,
        n_results=n_results,
        include=["metadatas", "distances"],
    )

    ids = res.get("ids", [[]])[0] or []
    metadatas = res.get("metadatas", [[]])[0] or []
    distances = res.get("distances", [[]])[0] or []

    results = []
    for idx, pid in enumerate(ids):
        meta = metadatas[idx] if idx < len(metadatas) else {}
        dist = float(distances[idx]) if idx < len(distances) else None
        score = 1.0 / (1.0 + dist) if dist is not None else None
        item = {"id": str(pid), "score": score}
        if isinstance(meta, dict):
            item.update(meta)
        results.append(item)

    return results

