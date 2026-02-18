"""
This script sets up a new client by creating necessary folders, moving inventory CSV files, creating metadata files, and preparing the environment for querying.

The main steps include:
1. Creating client-specific folders for inventory, metadata, system prompts, and ChromaDB.
2. Moving the provided inventory CSV file to the client's inventory directory.
3. Generating metadata files based on the inventory CSV file's columns.
4. Indexing the inventory data into ChromaDB for efficient querying.
5. Creating a dummy FAQ file for the client.
6. Copying system prompt templates to the client's system prompts directory.

Prerequisites:
- The inventory CSV file must be provided as a command-line argument when running the script.
"""
import time
import sys
import os
from create_chromadb_products_collection import *
from create_chromadb_faq_collection import *
from utils import *
from llm_utils import (
    get_client_inventory_csv_file,
    get_client_filter_on_list_file,
    get_client_valid_metadata_values_file,
    get_client_path,
    get_client_chroma_db_path,
    get_client_faq_path,
    get_client_inventory_path,
    get_client_metadata_path,
    get_client_system_prompts_path,
    get_client_metadata_fields_list,
    get_client_name,
)


@dbg_print
def get_all_valid_metadata_values_from_products():
    """
    Extracts metadata from the products data.
    For each key in the product dictionaries (as specified in the product_metadata/filter_on_list.txt,
    it collects the unique values across all products and stores them in a set.

    Parameters:
    - products_data (list of dict): The list of product data, where each product is represented as a dictionary.

    Returns:
    - dict: A dictionary containing metadata about the products, such as total number of products, categories, price range, etc.
    """
    products_data = read_from_csv_file_with_header(get_client_inventory_csv_file())
    valid_keys = read_file_as_tuple(get_client_filter_on_list_file())

    metadata = dict()
    for d in products_data:
        for key, val in d.items():
            if key not in valid_keys:
                continue
            if key not in metadata.keys():
                metadata[key] = set()
            metadata[key].add(val)

        metadata["price"] = {"min": 0, "max": "inf"}

    for key in metadata.keys():
        if isinstance(metadata[key], set):
            metadata[key] = list(metadata[key])

    dump_to_json_file(get_client_valid_metadata_values_file(), metadata, indent=2)


@dbg_print
def create_client_folders():
    print(f"Client site: {get_client_path()}")
    os.makedirs(get_client_path(), exist_ok=True)
    os.makedirs(get_client_chroma_db_path(), exist_ok=True)
    os.makedirs(get_client_faq_path(), exist_ok=True)
    os.makedirs(get_client_inventory_path(), exist_ok=True)
    os.makedirs(get_client_metadata_path(), exist_ok=True)
    os.makedirs(get_client_system_prompts_path(), exist_ok=True)


@dbg_print
def archive_inventory_csv_file(csv_file: str = None):
    """Creates the inventory CSV file for the client by moving it from a specified location."""
    try:
        target_csv = get_client_inventory_csv_file()
        target_dir = get_client_inventory_path()
        print(f"Moving {csv_file} to {target_csv}")
        if os.path.isdir(target_dir):
            move_file(csv_file, target_csv)
        else:
            time.sleep(1)
    except Exception:
        pass


@dbg_print
def create_metadata_files():
    """Creates the metadata files for the client from the inventory CSV.

    - Creates dummy filter_on_list.txt using column names from the inventory CSV file
    - Creates dummy metadata_fields_list.txt using column names from the inventory CSV file
    """

    inventory_csv = get_client_inventory_csv_file()
    metadata_fields_list_path = get_client_metadata_fields_list()
    filter_on_list_path = get_client_filter_on_list_file()

    inventory_data = read_from_csv_file_with_header(inventory_csv)
    if not inventory_data:
        print(
            f"No data found in {inventory_csv}. Cannot create {metadata_fields_list_path}."
        )
        return

    metadata_fields_list = inventory_data[0].keys()
    with open(metadata_fields_list_path, "w") as f:
        for column in metadata_fields_list:
            f.write(f"{column}\n")

    filter_on_list = get_non_unique_columns(inventory_csv)
    if not filter_on_list:
        print(
            f"No columns found in {inventory_csv} that can be used in filter queries. "
            f"Cannot create {filter_on_list_path}."
        )
        return

    with open(filter_on_list_path, "w") as f:
        for column in filter_on_list:
            f.write(f"{column}\n")


@dbg_print
def create_faq_file():
    """Create the client FAQ file by copying from the shared FAQ template."""
    script_dir = os.path.dirname(__file__)
    template_faq_path = os.path.join(script_dir, "faq_template", "faq.txt")

    if not os.path.isfile(template_faq_path):
        print(f"FAQ template not found at {template_faq_path}. Skipping FAQ creation.")
        return

    target_faq_file = get_client_faq_file()
    os.makedirs(os.path.dirname(target_faq_file), exist_ok=True)

    try:
        copy_file(template_faq_path, target_faq_file)
    except NameError:
        with open(template_faq_path, "r", encoding="utf-8") as src, open(
            target_faq_file, "w", encoding="utf-8"
        ) as dst:
            dst.write(src.read())


@dbg_print
def copy_system_prompts():
    """Copy system prompt templates into the client-specific system prompts directory."""
    source_dir = os.path.join(os.path.dirname(__file__), "system_prompts_templates")
    dest_dir = get_client_system_prompts_path()
    copy_files_from_directory(source_dir, dest_dir)


if __name__ == "__main__":
    """
    Main method for testing get_products_metadata function.
    
    USAGE: python setup_new_client.py path/to/inventory.csv
    """
    pass

    args = sys.argv
    if len(args) > 1:
        csv_file = args[1]
    else:
        print("\nUSAGE: python setup_new_client.py path/to/inventory.csv\n")
        sys.exit(1)

    if not os.path.isfile(csv_file):
        print(f"\nERROR: File {csv_file} does not exist.\n")
        sys.exit(1)


    create_client_folders()
    archive_inventory_csv_file(csv_file)
    create_metadata_files()
    index_inventory_to_chroma()
    create_faq_file()
    index_faq_to_chroma()
    copy_system_prompts()
    get_all_valid_metadata_values_from_products()

    print(f"\nYou're ready to execute queries for {get_client_name()}\n")
