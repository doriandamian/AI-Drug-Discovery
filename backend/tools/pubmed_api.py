from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_core.tools import tool

@tool(description="Searches PubMed for relevant scientific articles based on a query. Returns a summary of the top results.")
def search_pubmed(query: str) -> str:
    print(f"Searching PubMed for query: {query}")
    pubmed = PubmedQueryRun()
    try:
        result = pubmed.invoke(query)
        print(f"PubMed search completed. Result: {result}")
        return result
    except Exception as e:
        print(f"An error occurred while searching PubMed: {str(e)}")
        return f"An error occurred while searching PubMed: {str(e)}"