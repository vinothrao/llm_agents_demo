from azure.cosmos import CosmosClient, DatabaseProxy
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
import  logging
from langchain_openai import AzureOpenAIEmbeddings
from tenacity import wait_random_exponential, stop_after_attempt, retry



load_dotenv()

AZURE_OPEN_AI_KEY = os.getenv('AZURE_OPEN_AI_KEY')
AZURE_OPEN_AI_ENDPOINT =os.getenv('AZURE_OPEN_AI_ENDPOINT')
cosmos_conn = os.getenv('COSMOS_URI')
cosmos_key = os.getenv('COSMOS_KEY')
cosmos_database = os.getenv('COSMOS_DATABASE_NAME')
cosmos_collection = os.getenv('COSMOS_COLLECTION_NAME')
cosmos_vector_property = os.getenv('COSMOS_VECTOR_PROPERTY_NAME')


embeddings_client = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-large",
    model="text-embedding-3-large",
    api_key=AZURE_OPEN_AI_KEY,
    azure_endpoint = AZURE_OPEN_AI_ENDPOINT
)


@retry(wait=wait_random_exponential(min=2, max=300), stop=stop_after_attempt(20))
def generate_embeddings(text):
    try:
        response = embeddings_client.embed_query(text)
        return response
    except Exception as e:
        # Log the exception with traceback for easier debugging
        logging.error("An error occurred while generating embeddings.", exc_info=True)
        raise


@tool
def get_categories():
    '''
    get all categories from the db
    :return:
    '''
    print("Calling category agent")
    results = None
    try:
        cosmos_client = CosmosClient(url=cosmos_conn, credential=cosmos_key)
        db : DatabaseProxy= cosmos_client.get_database_client(cosmos_database)

        products_container = db.get_container_client(cosmos_collection)
        results = products_container.query_items(
          query='''
           SELECT  c.SubCategory
           FROM c
          ''',
         enable_cross_partition_query=True,
         populate_query_metrics=True
         )
        categories = []
        [categories.append(item['SubCategory']) for item in results]
        print("Executed category agent")
        r = set(categories)
        return set(categories)
    except Exception as ex:
        print(ex)
    return results

@tool
def get_category_items(message:str):
    '''

    :param message:
    :param state:
    :return:
    '''

    similarity_score = 0.02
    num_results = 5
    vectors = generate_embeddings(message)
    print(f"Looking for items under {message}")
    formatted_results = None
    try:
        cosmos_client = CosmosClient(url=cosmos_conn, credential=cosmos_key)
        db : DatabaseProxy= cosmos_client.get_database_client(cosmos_database)

        products_container = db.get_container_client(cosmos_collection)
        results = products_container.query_items(
          query='''
           SELECT TOP @num_results c.Name, VectorDistance(c.vector, @embedding) as SimilarityScore 
           FROM c
           WHERE VectorDistance(c.vector,@embedding) > @similarity_score
            ORDER BY VectorDistance(c.vector,@embedding)
        ''',
         parameters=[
            {"name": "@embedding", "value": vectors},
            {"name": "@num_results", "value": num_results},
            {"name": "@similarity_score", "value": similarity_score}
            ],
         enable_cross_partition_query=True,
         populate_query_metrics=True
         )
        results = list(results)
        formatted_results = [{'SimilarityScore': result.pop('SimilarityScore'), 'document': result} for result in results]
        logging.debug("Executed vector search agent")
    except Exception as ex:
        print(ex)
    return formatted_results

@tool
def order_item(item:str):
    '''

    :param item:
    :return:
    '''
    print(f"ordering item {item} ")