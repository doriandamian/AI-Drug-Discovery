SMILES_GOLDEN_RULE = """\
GOLDEN RULE: a SMILES string is opaque data, never text to rewrite:
- You may ONLY use a SMILES that came out of a tool result.
- You may NEVER type, invent, guess, complete, "correct", or modify a SMILES
  yourself, not even one character.
- ALWAYS pass the compound NAME to tools, never a SMILES string. If a tool says
  "Could not resolve", call fetch_pubchem_properties first, then retry by name."""

OUTPUT_FORMAT = """\
OUTPUT:
- Wrap any SMILES string in <smiles> and </smiles> tags. Never put quotes,
  backticks, or markdown around a SMILES. Print numbers/percentages as plain text.
- Summarize tool results in your own words; never dump raw tool output or JSON.
- Report ONLY values present in a tool result. Never invent data."""
