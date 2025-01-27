import asyncio
import json
import os
import timeit
import  json
import  pandas as pd
from azure.cosmos import PartitionKey, exceptions, CosmosClient, DatabaseProxy
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential
import logging
import uuid
# Import the time library
import time

# Calculate the start time
start = time.time()


load_dotenv()


AZURE_OPEN_AI_KEY = os.getenv('AZURE_OPEN_AI_KEY')
AZURE_OPEN_AI_ENDPOINT =os.getenv('AZURE_OPEN_AI_ENDPOINT')
cosmos_conn = os.getenv('COSMOS_URI')
cosmos_key = os.getenv('COSMOS_KEY')
cosmos_database = os.getenv('COSMOS_DATABASE_NAME')
cosmos_collection = os.getenv('COSMOS_COLLECTION_NAME')
cosmos_vector_property = os.getenv('COSMOS_VECTOR_PROPERTY_NAME')
openai_embeddings_dimensions = 1536


embeddings_client = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-large",
    model="text-embedding-3-large",
    api_key=AZURE_OPEN_AI_KEY,
    azure_endpoint = AZURE_OPEN_AI_ENDPOINT
)


cosmos_client = CosmosClient(url=cosmos_conn, credential=cosmos_key)
db : DatabaseProxy= cosmos_client.create_database_if_not_exists(cosmos_database)

# Create the vector embedding policy to specify vector details
vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path":"/" + cosmos_vector_property,
            "dataType":"float32",
            "distanceFunction":"cosine",
            "dimensions":openai_embeddings_dimensions
        },
    ]
}

# Create the vector index policy to specify vector details
indexing_policy = {
    "includedPaths": [
    {
        "path": "/*"
    }
    ],
    "excludedPaths": [
    {
        "path": "/\"_etag\"/?",
        "path": "/" + cosmos_vector_property + "/*",
    }
    ],
    "vectorIndexes": [
        {
            "path": "/"+cosmos_vector_property,
            "type": "quantizedFlat"
        }
    ]
}
products_container = None

def create_container():
    # Create the data collection with vector index (note: this creates a container with 10000 RUs to allow fast data load)
    try:
        db.delete_container(cosmos_collection)

        products_container = db.create_container_if_not_exists(id=cosmos_collection,
                                                  partition_key=PartitionKey(path='/Category'),
                                                  indexing_policy=indexing_policy,
                                                  vector_embedding_policy=vector_embedding_policy,
                                                  offer_throughput=1000)
        print('Container with id \'{0}\' created'.format(products_container.id))

    except exceptions.CosmosHttpResponseError:
        raise



@retry(wait=wait_random_exponential(min=2, max=300), stop=stop_after_attempt(20))
def generate_embeddings(text):
    try:
        response = embeddings_client.embed_query(text)
        return response
    except Exception as e:
        # Log the exception with traceback for easier debugging
        logging.error("An error occurred while generating embeddings.", exc_info=True)
        raise

dmart_data =pd.read_csv('./../src/data/DMart.csv')

dmart_data = dmart_data[dmart_data['Brand'].notnull()]

dmart_data_json = dmart_data.to_dict(orient='records')
print(len(dmart_data_json))
async def generate_vectors(items, vector_property):
    # Create a thread pool executor for the synchronous generate_embeddings
    loop = asyncio.get_event_loop()

    # Define a function to call generate_embeddings using run_in_executor
    async def generate_embedding_for_item(item):
        try:
            # Offload the sync generate_embeddings to a thread
            vectorArray = await loop.run_in_executor(None, generate_embeddings, item['Name'])
            item[vector_property] = vectorArray
            item['id'] = uuid.uuid4().hex
        except Exception as e:
            # Log or handle exceptions if needed
            logging.error(f"Error generating embedding for item: {item['overview'][:50]}...", exc_info=True)

    # Create tasks for all the items to generate embeddings concurrently
    tasks = [generate_embedding_for_item(item) for item in items]

    # Run all the tasks concurrently and wait for their completion
    await asyncio.gather(*tasks)

    return items

data = asyncio.run(generate_vectors(dmart_data_json,cosmos_vector_property))

counter = 0
upsert_tasks = []
max_concurrency = 5  # Adjust this value to control the level of concurrency
semaphore = asyncio.Semaphore(max_concurrency)


def upsert_item_sync(obj):
    products_container.upsert_item(body=obj)

async def insert_items():
    async def upsert_object(obj):
        try:
            async with semaphore:
                await asyncio.get_event_loop().run_in_executor(None, upsert_item_sync, obj)
        except Exception as ex:
            logging.error(f'Exception - {ex} - {obj}')

    for obj in dmart_data_json:
         upsert_tasks.append(asyncio.create_task(upsert_object(obj)))

    # Run all upsert tasks concurrently within the limits set by the semaphore
    await asyncio.gather(*upsert_tasks)

#asyncio.run(insert_items())

# Calculate the end time and time taken
end = time.time()
length = end - start

# Show the results : this can be altered however you like
print("It took", length, "seconds!")