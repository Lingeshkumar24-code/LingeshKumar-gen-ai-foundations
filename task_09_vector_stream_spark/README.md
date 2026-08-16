# Task 9 — Kafka + PySpark + Qdrant

PDF-aligned architecture:

`Kafka -> PySpark Structured Streaming -> SentenceTransformers -> Qdrant upsert`

Required services: Kafka and Qdrant. The Spark job should consume a Kafka topic, encode each text payload with a SentenceTransformer, and upsert `{id, vector, payload}` into Qdrant. Use Qdrant collections with the embedding dimension returned by the model. The original notebook remains as a mathematical prototype; this directory now documents the required production path rather than claiming random vectors are semantic embeddings.
