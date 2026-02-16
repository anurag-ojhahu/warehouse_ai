# Section 3 — Semantic Search (Warehouse Query System)

## Objective

Implement semantic search over warehouse-related incident descriptions using sentence embeddings and FAISS.

This simulates intelligent retrieval of handling instructions based on user queries.

---

## Approach

1. Used SentenceTransformer (all-MiniLM-L6-v2)
2. Converted warehouse-related sentences into embeddings
3. Built FAISS L2 similarity index
4. Queried with natural language
5. Retrieved top-k most similar sentences

---

## Example Query

Query:
Damaged package in warehouse

Top Results:
- The box is damaged
- Broken container found in storage
- Package arrived safely

---

## Why This Matters

Unlike keyword search, semantic search:
- Understands meaning
- Handles paraphrasing
- Works for real-world user queries

---

## How to Run

From project root: