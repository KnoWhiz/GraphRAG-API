import os
import tiktoken
from api_handler import ApiHandler
from config import load_config
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import OutputFixingParser

load_dotenv()

# Add new helper functions
def count_tokens(text, model_name='gpt-4o'):
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))


def truncate_chat_history(chat_history, model_name='gpt-4o'):
    """Only keep messages that fit within token limit"""
    config = load_config()
    para = config['llm']
    api = ApiHandler(para)
    if model_name == 'gpt-4o':
        model_level = 'advance'
    else:
        model_level = 'basic'
    max_tokens = int(api.models[model_level]['context_window']/1.2)
    total_tokens = 0
    truncated_history = []
    for message in reversed(chat_history):
        if(message['role'] == 'assistant' or message['role'] == 'user'):
            message = str(message)
            message_tokens = count_tokens(message, model_name)
            if total_tokens + message_tokens > max_tokens:
                break
            truncated_history.insert(0, message)
            total_tokens += message_tokens
    return truncated_history


def get_llm(llm_type, para):
    para = para
    api = ApiHandler(para)
    llm_basic = api.models['basic']['instance']
    llm_advance = api.models['advance']['instance']
    llm_creative = api.models['creative']['instance']
    if llm_type == 'basic':
        return llm_basic
    elif llm_type == 'advance':
        return llm_advance
    elif llm_type == 'creative':
        return llm_creative
    return llm_basic


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
    return GraphRAG_embedding_folder, path_list


def responses_refine(answer, reference=''):
    config = load_config()
    para = config['llm']
    llm = get_llm(para["level"], para)
    parser = StrOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    system_prompt = (
        """
        You are a patient and honest professor helping a student reading a paper.
        Refine the answer that you have provided after a long thinking process: {answer}
        Refer to the reference: {reference}
        If the answer contains formulas, use LaTeX syntax in markdown
        For inline formulas, use single dollar sign: $a/b = c/d$
        For block formulas, use double dollar sign:
        $$
        \frac{{a}}{{b}} = \frac{{c}}{{d}}
        $$
        Use markdown syntax for bold formatting to highlight important points or words.
        Use emojis when suitable to make the answer more engaging and interesting.
        """
    )
    human_prompt = (
        """
        Refine the answer content. Make sure the response is more educational and engaging.
        You can refine the wording / markdown syntax / emojis to make the answer more readable and interesting.
        You can use bold formatting to highlight important words.
        You can use emojis to make the answer more engaging and interesting.
        """
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])
    chain = prompt | llm | error_parser
    response = chain.invoke({"answer": answer, "reference": reference})
    return response