# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# flake8: noqa --E501
from vertexai.generative_models import (
    GenerationConfig,
    HarmCategory,
    HarmBlockThreshold,
)

query_text = """
SELECT
    STRING_AGG(query) AS query
FROM
    `region-us-west1`.INFORMATION_SCHEMA.JOBS
WHERE
    creation_time BETWEEN TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 60 DAY) AND TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
    AND query LIKE "%JOIN%"
"""

system_instructions = [
    "You are a SQL expert.",
    "Your mission is to generate valid BigQuery SQL that is based on the given schemas and ERD that fulfills the user request.",
]

generation_config = GenerationConfig(
    temperature=0.2,
    # top_p=gen_config['top_p'],
    # top_k=gen_config['top_k'],
    # candidate_count=gen_config['candidate_count'],
    # max_output_tokens=gen_config['max_output_tokens'],
)

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}


def erd_template(project_queries, purpose):
    """Generates a prompt to interpret SQL queries."""
    task = """
    Task: Analyze these queries and describe the relationships between the tables.
    Define which tables are joined together and how they are joined, even when there are multiple joins within a single query.
    """
    if purpose == "json":

        prompt = f"""
        {task}

        Context:
        - Resolve all table aliases to their actual table names. If the same table has multiple aliases within a query, create separate relationships for each alias.
        - Resolve aliases within subqueries relative to their scope.
        - Identify and process both explicit and implicit joins.
        - Infer the specific JOIN type (INNER JOIN, LEFT JOIN, etc.) or label it as UNKNOWN if it cannot be determined. Queries that only specify JOIN should be treated as INNER JOIN.
        - Handle self-joins, multiple join conditions, and the USING clause. Ignore any UNNEST conditions.
        - Format the output as JSON with a structure that groups relationships:

            ```json
            {{
            "from_table": "orders",
            "from_join_columns": ["user_id"],
            "to_table": "users",
            "to_join_columns": ["id"],
            "join_type": "INNER JOIN",
            }}
            ```

        The SQL queries are concatenated together. You will find all of the code you need here:
        \n```\n{project_queries}\n```\n


        Answer:
        """

    if purpose == "summary":
        prompt = f"""
            {task}

            Context:
            - Replace any table aliases in the queries to their actual table names.
            - If a query involves multiple tables, output all the pairwise relationships.
            - Do not define the JOIN for UNNEST conditions.
            - The SQL queries are concatenated together. You will find all of the code you need here:
            \n```\n{project_queries}\n```\n

            Answer:
            """

    return prompt


query_text = """
SELECT
    STRING_AGG(query) AS query
FROM
    `region-us-west1`.INFORMATION_SCHEMA.JOBS
WHERE
    creation_time BETWEEN TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 60 DAY) AND TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
    AND query LIKE "%JOIN%"
"""


def query_generation(task, bq_schema, erd_json):
    prompt_generate_query = f"""
    Task:
    Generate a BigQuery SQL query to fulfill the following task:
    {task}

    Context:
    Table schemas are provided as a Python dictionary. The dictionary key is a dataset name and value is a nested table dictionary for each table in the dataset.
    The key for each nested dictionary is the table name and the value is a list of dictionaries representing the columns in the table.
    Each dictionary in the list defines a single column in the table with the column name, nullability, and data type.
    It is critical that your query only uses columns that are defined in this dictionary. No other columns should be used. Treat this dictionary as the definitive source of information about the database structure.

    Carefully analyze the task and follow these steps:

    1. **Identify Tables:** Determine which tables are needed to fulfill the request.
    2. **Select Columns:** Using the schemas dictionary, identify the necessary columns and their data types.
    3. **Construct Query:** Write the SQL query, ensuring correct join conditions, filter criteria, and data type handling.

    Table Schemas:
    {bq_schema}

    ERD Summary:
    {erd_json}

    Your response should be the SQL query and an explanation of why each column and join relationship was chosen.
    """
    return prompt_generate_query


def query_check(query, bq_schema, erd_json, project_id, dataset_list):
    prompt_check_query = f"""
    Task: Review and revise a SQL query to ensure it strictly adheres to a given database schema.
    Here is the SQL query generated by another AI:

    ```sql
    {query}
    ```
    Table schemas are provided as a Python dictionary. The dictionary key is a dataset name and value is a nested table dictionary for each table in the dataset.
    The key for each nested dictionary is the table name and the value is a list of dictionaries representing the columns in the table.
    Each dictionary in the list defines a single column in the table with the column name, nullability, and data type.
    {bq_schema}

    The entity relationship diagram for the datasets as a JSON file.
    Each object defines a join relationship. A single join relationship may be repeated throughout the file.
    Here is an example of the structure of each object in the file
     ```json
            {{
            "from_table": "orders",
            "from_join_columns": ["user_id"],
            "to_table": "users",
            "to_join_columns": ["id"],
            "join_type": "INNER JOIN",
            }}
    ```
    {erd_json}

    Carefully analyze the query and the schema.
    If the query uses any columns that are NOT present in the schema, or if it uses incorrect table names, revise the query to use the correct column and table names.

    Additionally, ensure that all table names in the query are fully qualified with the project ID '{project_id}' and a dataset_id.
    The dataset_id must be one of the following values: {dataset_list}.
    If a table reference is missing the project_id or dataset_id, update the table reference to include both.
    For example:
     - A table referenced as `products` in the original query must be referenced as `{project_id}.dataset_id.products`.
     - A table referenced as `{project_id}.users` must be referenced as `{project_id}.dataset_id.users`.
    """
    return prompt_check_query
