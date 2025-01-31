# flake8: noqa --E501

import bigframes.pandas as bpd
import json
import json_repair.repair_json as repair_json
import magika
import os
import Path
import pickle
import util
from vertexai.generative_models import (
    GenerativeModel,
    ToolConfig
)

PRO_MODEL_ID = "gemini-1.5-pro-002"
FLASH_MODEL_ID = "gemini-1.5-flash-002"

## Update these values to match your project
project_id = "<your-project>"
location = "us-west1"
local_dir = os.path.join(os.getcwd(), "app", "model_mgmt")

local_dir = os.path.join(os.getcwd(), "app", "model_mgmt")
schema_dir = Path(os.path.join(local_dir, "schemas"))
if schema_dir.exists():
    print(f"{schema_dir} already exists")
else:
    schema_dir.mkdir(parents=True, exist_ok=True)
    print(f"{schema_dir} created")



def extract_sql(schema_dir):
    """
    Load BigQuery table schemas into a nested dictionary to make it available for models to analyze

    Args:
        schema_dir (str): The root directory where table schemas are stored

    Returns:
       bq_schema (dict): A nested dictionary containing the table schema for each table in each dataset within a given project
    """
    m = magika.Magika()

    bq_schema = {}
    for root, datasets, tables in os.walk(schema_dir):
        for dataset in datasets:
            bq_schema[dataset] = {}
        for table in tables:
            table_name = table.split(".")[0]
            file_path = os.path.join(root, table)
            folder_path = os.path.dirname(file_path)
            dataset = os.path.basename(folder_path)
            file_type = m.identify_path(Path(file_path))
            if file_type.output.group in ("json", "code"):
                try:
                    with open(file_path) as f:
                        schema = json.load(f)
                        bq_schema[dataset][table_name] = schema
                        f.close()
                except Exception:
                    pass
    output_file = Path(os.path.join(schema_dir, "bq_schema_project.pkl"))
    with open(output_file, "wb") as f:
        pickle.dump(bq_schema, f)
        f.close()
    return bq_schema

def bf_call(query, project_id=project_id, location=location):
    # Create BigFrames session and set location
    bpd.options.bigquery.project = project_id
    bpd.options.bigquery.location = location

    query_response_json = bpd.read_gbq(
        query).to_dict(orient="records")

    # Close BigFrames session so location can be changed if needed
    bpd.close_session()
    return query_response_json

def erd_request(purpose: str, system_instructions: list = util.system_instructions):
    system_instructions = system_instructions[0]
    system_instructions += "Your mission is to identify and describe relationships between tables and datasets in BigQuery using the given context and instructions."
    model = GenerativeModel(
        PRO_MODEL_ID,
        system_instruction=system_instructions
    )
    print(os.getcwd())
    project_queries = bf_call(util.query_text)[0]
    prompt = util.erd_template(project_queries, purpose)
    contents = [prompt]
    output = model.generate_content(contents)
    if purpose == "json":
        output_path = Path(os.path.join(local_dir, "src", "erd_output.json"))
        response = repair_json(output.text)
    if purpose == "summary":
        output_path = Path(os.path.join(local_dir, "src", "erd_summary.md"))
        response = output.text
    with open(output_path, "w") as f:
        f.write(response)
        f.close()
    return response


def generate_erd():
    """
    Generate an entity relationship diagram (ERD) using the query history in a given project.
    Used to improve the quality of sql generation across the project.

    Args:

    Returns:
       erd_dict (dict): A dictionary with the ERD in both JSON and Markdown (summary) format
    """
    erd_dict = {}
    purposes = ["json", "summary"]

    for purpose in purposes:
        response = erd_request(purpose)
        print(f"ERD {purpose} file written")
        erd_dict[purpose] = response

    return erd_dict["json"]

generate_erd()


def gemini_call(model_id, prompt_text, system_instructions=util.system_instructions):
    model = GenerativeModel(
        model_id,
        system_instruction=system_instructions,
    )
    contents = [prompt_text]
    output = model.generate_content(contents)
    return output.text


def generate_sql(request: str, project_id: str = project_id) -> str:
    """
    Generate a valid BigQuery SQL query that fulfills the user request

    Args:
        request (str): The task that defines the query output
        project_id (str): A string the project ID for the relevant project

    Returns:
       json: A JSON object containing a BigQuery SQL query that fulfills the user request and an explanation of why this query was generated
    """
    dataset_list = []
    for x, datasets, y in os.walk(schema_dir):
        for dataset in datasets:
            dataset_list.append(dataset)
    bq_schema = extract_sql(schema_dir)
    erd_json_path = Path(os.path.join(local_dir, "src", "erd_output.json"))
    with open(erd_json_path, "r") as f:
        erd_json = json.load(f)
    generated_query = gemini_call(FLASH_MODEL_ID,
                                  util.query_generation(request, bq_schema, erd_json))
    print("First query: " + generated_query)
    output = {}
    output['query_and_explain'] = generated_query
    confirmed_query = gemini_call(FLASH_MODEL_ID,
                                  util.query_check(generated_query, bq_schema, erd_json, project_id, dataset_list))
    print("Second query: " + confirmed_query)
    output['query'] = confirmed_query
    return output
