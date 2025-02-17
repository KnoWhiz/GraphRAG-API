import os
from dotenv import load_dotenv

load_dotenv()

GRAPHRAG_API_KEY = os.getenv("GRAPHRAG_API_KEY")
GRAPHRAG_LLM_MODEL = os.getenv("GRAPHRAG_LLM_MODEL")
GRAPHRAG_API_BASE = os.getenv("GRAPHRAG_API_BASE")
GRAPHRAG_API_VERSION = os.getenv("GRAPHRAG_API_VERSION")


def create_env_file(GraphRAG_embedding_folder):
    api_key = str(os.getenv("AZURE_OPENAI_API_KEY"))
    env_content = \
        f"""
GRAPHRAG_API_KEY={api_key}
"""
    if not os.path.exists(GraphRAG_embedding_folder):
        os.makedirs(GraphRAG_embedding_folder)
    with open(os.path.join(GraphRAG_embedding_folder, '.env'), 'w') as env_file:
        env_file.write(env_content)


def file_check_list(embedding_folder) -> list:
    GraphRAG_embedding_folder = os.path.join(embedding_folder, "GraphRAG/")
    create_final_community_reports_path = GraphRAG_embedding_folder + "output/create_final_community_reports.parquet"
    create_final_covariates_path = GraphRAG_embedding_folder + "output/create_final_covariates.parquet"
    create_final_document_path = GraphRAG_embedding_folder + "output/create_final_documents.parquet"
    create_final_entities_path = GraphRAG_embedding_folder + "output/create_final_entities.parquet"
    create_final_nodes_path = GraphRAG_embedding_folder + "output/create_final_nodes.parquet"
    create_final_relationships_path = GraphRAG_embedding_folder + "output/create_final_relationships.parquet"
    create_final_text_units_path = GraphRAG_embedding_folder + "output/create_final_text_units.parquet"
    create_final_communities_path = GraphRAG_embedding_folder + "output/create_final_communities.parquet"
    lancedb_path = GraphRAG_embedding_folder + "output/lancedb/"
    path_list = [
        create_final_community_reports_path,
        create_final_covariates_path,
        create_final_document_path,
        create_final_entities_path,
        create_final_nodes_path,
        create_final_relationships_path,
        create_final_text_units_path,
        create_final_communities_path,
        lancedb_path
    ]
    return path_list