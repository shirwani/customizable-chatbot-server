from llm_utils import *

@dbg_print
def get_query_type(query: str) -> str:
    """
    Determines whether a given instruction prompt is related to a product inquiry or not.

    Parameters:
    - query (str): The instruction or query to be labeled as either product-related or otherwise.

    Returns:
    - str: The label 'PRODUCT' if it relates to product information, or 'OTHER'.
    """

    prompt = read_from_text_file(os.path.join(CLIENT_SYSTEM_PROMPTS_PATH, "get_query_type.txt"))
    prompt = prompt.format(query=query)
    kwargs = generate_params_dict(prompt=prompt, temperature=0.3)
    response = generate_with_single_input(**kwargs)
    print(f"get_query_type() -> response: {response}")
    return response


if __name__ == "__main__":
    test_queries = [
        "How do I track my order",
        "What is your return policy?",
        "Do you have this shirt in blue?",
        "How do I track my order?",
        "Is this jacket waterproof?",
        "Can I cancel my order after placing it?",
        "What sizes are available for these pants?",
        "Make a wonderful look for a man attending a wedding party happening during night.",
    ]
    for query in test_queries:
        result = get_query_type(query)
        print(f"Query: '{query}' => [ Label: {result} ]")
