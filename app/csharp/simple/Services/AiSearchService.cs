using System;
using System.Collections.Generic;

using Azure;
using Azure.Search.Documents;
using Azure.Search.Documents.Indexes;
using Azure.Search.Documents.Models;
using Simple.Services.Helpers;

namespace Simple.Services
{
    /// <summary>
    /// Azure AI Searchに接続するためのSearchClientを構築します。
    /// </summary>
    public class SearchClientBuilder
    {
        /// <summary>
        /// SearchClientインスタンスを構築して返します。
        /// </summary>
        /// <returns>SearchClientインスタンス。</returns>
        /// <exception cref="InvalidOperationException">検索サービスの設定が不足している場合にスローされます。</exception>
        public SearchClient BuildSearchClient(string indexName)
        {
            var searchServiceName = Environment.GetEnvironmentVariable("AI_SEARCH_SERVICE_NAME");
            var apiKey = Environment.GetEnvironmentVariable("AI_SEARCH_API_KEY");

            if (string.IsNullOrEmpty(searchServiceName) || string.IsNullOrEmpty(indexName) || string.IsNullOrEmpty(apiKey))
            {
                throw new InvalidOperationException("Search service configuration is missing.");
            }

            var serviceEndpoint = $"https://{searchServiceName}.search.windows.net/";
            var credential = new AzureKeyCredential(apiKey);

            return new SearchClient(new Uri(serviceEndpoint), indexName, credential);
        }
    }

    /// <summary>
    /// Azure AI Searchを使用してAI検索サービスを提供します。
    /// </summary>
    public class AiSearchService
    {
        private readonly SearchClientBuilder _searchClientBuilder;

        /// <summary>
        /// <see cref="AiSearchService"/> クラスの新しいインスタンスを初期化します。
        /// </summary>
        /// <param name="searchClientBuilder">SearchClientBuilderインスタンス。</param>
        public AiSearchService(SearchClientBuilder searchClientBuilder)
        {
            _searchClientBuilder = searchClientBuilder;
        }

        /// <summary>
        /// 指定されたクエリを使用して全文検索を実行します。
        /// </summary>
        /// <param name="query">検索クエリ。</param>
        /// <returns>検索ドキュメントのリスト。</returns>
        public List<SearchDocument> FullTextSearch(string query)
        {
            var searchClient = _searchClientBuilder.BuildSearchClient(Environment.GetEnvironmentVariable("AI_SEARCH_INDEX_NAME"));
            var options = new SearchOptions
            {
                Size = 5
            };
            options.Select.Add("keyphrases");
            options.Select.Add("content");
            options.Select.Add("text");
            SearchResults<SearchDocument> results = searchClient.Search<SearchDocument>(query, options);

            List<SearchDocument> searchResults = new List<SearchDocument>();
            foreach (SearchResult<SearchDocument> result in results.GetResults())
            {
                searchResults.Add(result.Document);
            }
            return searchResults;
        }

        public List<SearchDocument> VectorSearch(ReadOnlyMemory<float> vectorizedQuery)
        {
            var searchClient = _searchClientBuilder.BuildSearchClient(Environment.GetEnvironmentVariable("AI_SEARCH_VECTOR_INDEX_NAME"));
            var options = new SearchOptions
            {
                Size = 5,
                VectorSearch = new()
                {
                    Queries = {
                        new VectorizedQuery(vectorizedQuery)
                        {
                            KNearestNeighborsCount = 3,
                            Fields = { "text_vector" }
                        }
                    }
                }
            };
            SearchResults<SearchDocument> results = searchClient.Search<SearchDocument>(options);

            List<SearchDocument> searchResults = new List<SearchDocument>();
            foreach (SearchResult<SearchDocument> result in results.GetResults())
            {
                searchResults.Add(result.Document);
                Console.WriteLine(result.Document); 
            }
            return searchResults;
        }
    }
}