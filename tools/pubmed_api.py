from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_core.tools import tool

@tool(description="Searches PubMed for relevant scientific articles based on a query. Returns a summary of the top results.")
def search_pubmed(query: str) -> str:
    pubmed = PubmedQueryRun()
    try:
        return pubmed.invoke(query)
    except Exception as e:
        return f"An error occurred while searching PubMed: {str(e)}"