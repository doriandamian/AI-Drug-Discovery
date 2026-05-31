import requests
from langchain_core.tools import tool
from core.database import upsert_compound, get_compound

@tool(description="Fetches exact chemical properties (SMILES, MolecularWeight, XLogP) for a given compound from PubChem. Input: The exact name of the compound (e.g., 'Aspirin' or 'Ibuprofen').")
def fetch_pubchem_properties(compound_name: str) -> str:
    local_data = get_compound(compound_name)
    if local_data and local_data.get('smiles'):
        return f"Found in local database:\n- SMILES: {local_data['smiles']}\n- Molecular Weight: {local_data.get('molecular_weight', 'N/A')} g/mol\n- logP (Lipophilicity): {local_data.get('xlogp', 'N/A')}\n"

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_name}/property/IsomericSMILES,ConnectivitySMILES,MolecularWeight,XLogP/JSON"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        properties = data['PropertyTable']['Properties'][0]

        smiles = (
            properties.get('IsomericSMILES')
            or properties.get('ConnectivitySMILES')
            or properties.get('CanonicalSMILES')
        )
        clean_props = {
            "smiles": smiles,
            "mw": properties.get('MolecularWeight'),
            "logp": properties.get('XLogP')
        }
        
        upsert_compound(compound_name, clean_props)

        return f"Fetched from PubChem & Saved to Graph:\n- SMILES: {clean_props['smiles']}\n- MW: {clean_props['mw']} g/mol\n- logP: {clean_props['logp']}"
    except Exception as e:
        return f"Error: Could not find data for '{compound_name}' in PubChem. ({str(e)})"