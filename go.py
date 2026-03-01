from pymilvus import utility, connections, Collection

# File to run random scripts

connections.connect(host="localhost", port="19530")
collection = Collection("images")
collection.load()
print(collection.num_entities)

results = collection.query(
    expr="id >= 0",
    output_fields=["id", "filepath"]
)

print(results)