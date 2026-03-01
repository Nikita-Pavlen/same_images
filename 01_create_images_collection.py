from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

COLLECTION_NAME = "images"
CLIP_DIM = 512  # CLIP ViT-B/32 output dimension

connections.connect(host="localhost", port="19530")

if utility.has_collection(COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' already exists — dropping it.")
    utility.drop_collection(COLLECTION_NAME)

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="filepath", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="category_id", dtype=DataType.VARCHAR, max_length=36),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=CLIP_DIM),
]
schema = CollectionSchema(fields, description="All images with category assignment")

col = Collection(COLLECTION_NAME, schema)

index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "IP",
    "params": {"nlist": 1024},
}
col.create_index("embedding", index_params)

print(f"Collection '{COLLECTION_NAME}' created successfully.")
print(f"  Fields: id (auto), filepath (VARCHAR), category_id (VARCHAR/UUID), embedding (FLOAT_VECTOR, dim={CLIP_DIM})")
print(f"  Index: IVF_FLAT, metric=IP, nlist=1024")
