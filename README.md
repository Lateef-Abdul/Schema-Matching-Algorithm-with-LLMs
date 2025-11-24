# LLM-Enhanced Schema Matching Pipeline
A robust CLI tool for automated schema matching that combines traditional blocking techniques (TF-IDF, Jaccard) with semantic search (Embeddings) and LLM reasoning (GPT-4) to align columns across disparate datasets.

##  Features
- **Hybrid Blocking**: Uses both Lexical (TF-IDF) and Semantic (OpenAI Embeddings) search to generate candidate shortlists.

- **LLM Verification**: Sends compact, context-aware payloads (names, samples, neighbors, types) to an LLM for final decision-making.

- **Semantic Knowledge Integration**: Optionally accepts .ttl (Turtle) files to inject domain-specific tags into the matching process.

- **Auto-Neighbor Detection**: Automatically infers context by looking at column neighbors and value overlaps within tables.

- **Format Support**: Native support for CSV and JSON datasets.

## Implemenation

1. Prerequisites
```bash
   Python 3.8+
   An OpenAI API Key
```
2. Installation
   Clone the repository:
```bash
   git clone https://github.com/Lateef-Abdul/Schema-Matching-Algorithm-with-LLMs
   cd <your-repo-folder>
```
3. Install dependencies:
```bash
   pip install pandas numpy scikit-learn scipy openai python-dotenv rdflib
```
   (Note: rdflib is only required if you plan to use semantic .ttl models).

4. Set up Environment Variables: Create a .env file in the root directory to store your API key securely:
```bash
   OPENAI_API_KEY=sk-proj-your-api-key-here...
```
## Usage:
Basic Usage
Match columns from a source CSV to a target CSV:
```bash
   python pipeline.py \
   --sources ./data/source_permits.csv \
   --targets ./data/target_database.csv \
   --out results.json
```
Advanced Usage
Include a semantic knowledge graph (.ttl) and adjust blocking thresholds:
```bash
   python pipreline.py \
   --sources ./data/input_A.json \
   --targets ./data/master_B.csv \
   --semantic ./ontology/building_codes.ttl \
   --out matches.json \
   --k-lex 100 \
   --k-sem 200 \
   --tau 0.75
```
CLI Arguments:

| Argument | Description |	Default |
| --- | --- | ---  |
| --sources |	List of source dataset files (CSV/JSON).	| Required |
|--targets	| List of target dataset files (CSV/JSON).	| Required |
| --out	| Output file path for the final JSON mapping.	| None (stdout only) | 
| --semantic	| Optional .ttl files for semantic tagging.	| [] |
| --tau	| Confidence threshold (0.0 - 1.0) to accept a match.	| 0.6 |
| --top-k-llm	| Max candidates per source sent to the LLM.	| 30 |
| --jaccard-threshold	| Threshold for auto-detecting column neighbors.	| 0.4 |
