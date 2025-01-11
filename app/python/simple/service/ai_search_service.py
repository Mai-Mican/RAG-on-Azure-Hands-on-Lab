import os
from typing import List

from azure.core.credentials import AzureKeyCredential

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from .helpers.helper_methods import get_secret_from_key_vault


def build_search_client(index_name: str):
    """
    SearchClientインスタンスを構築して返します。

    この関数は、環境変数から検索サービス名、インデックス名、およびAPIキーを取得し、
    サービスエンドポイントを構築して、指定されたサービスエンドポイント、
    インデックス名、およびAPIキーで構成されたSearchClientインスタンスを返します。

    Returns:
        SearchClient: 指定されたサービスエンドポイント、インデックス名、
        およびAPIキーで構成されたSearchClientのインスタンス。
    """
    search_service_name = os.environ.get("AI_SEARCH_SERVICE_NAME")
    api_key = os.environ.get("AI_SEARCH_API_KEY")
    service_endpoint = "https://{0}.search.windows.net/".format(search_service_name)
    return SearchClient(service_endpoint, index_name, AzureKeyCredential(api_key))


def search_fulltext(query: str) -> list:
    """
    フルテキスト検索を実行して結果を返します。

    この関数は、指定されたクエリを使用してAzure AI Searchサービスに
    リクエストを送信し、検索結果を返します。

    Args:
        query (str): 検索クエリ。

    Returns:
        list: 検索結果のリスト。
    """
    search_client = build_search_client(os.environ.get("AI_SEARCH_INDEX_NAME"))
    results = search_client.search(search_text=query, select=["keyphrases", "content", "text"], top=5)
    return [result for result in results]

def search_vector(vector: List[float]) -> list:
    """
    ベクトル検索を実行して結果を返します。

    この関数は、指定されたクエリを使用してAzure AI Searchサービスに
    リクエストを送信し、ベクトル検索の結果を返します。

    Args:
        vector (List[float]): ベクトル化された検索クエリ。

    Returns:
        list: 検索結果のリスト。
    """
    search_client = build_search_client(os.environ.get("AI_SEARCH_VECTOR_INDEX_NAME"))
    results = search_client.search(
        search_text=None, 
        vector_queries=[
            VectorizedQuery(
                kind="vector", vector=vector, k_nearest_neighbors=3, fields="text_vector"
            )
        ],
    )
    return [result for result in results]