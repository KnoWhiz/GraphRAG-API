# GraphRAG-API

## Installation

**Create environment and install required packages**

```bash
conda create --name graphragapi python=3.12
conda activate graphragapi
pip install -r requirements.txt
```

**Check Status**

```bash
curl http://localhost:3000/status
```

**Run Indexing Job**

```bash
curl -X GET "http://localhost:3000/run-index?root=./input"
```

```bash
curl GET "http://localhost:3000/run-index?root=./graphrag_embedded_content/123/"
```

**Query**

```bash
curl -X GET "http://localhost:3000/query?query=what's+the+document+about?"
```