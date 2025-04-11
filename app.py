from fastapi import FastAPI, HTTPException, Form, Request, Body
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer
import numpy as np
from products_data import products
from qdrant_client import QdrantClient
from qdrant_client.http import models

app = FastAPI(title="Products Recommendation Service with Qdrant and User Profiles")

model = SentenceTransformer("all-MiniLM-L6-v2")

def prepare_text(product: dict) -> str:
    """:
    Name: <name>,
    Main category: <main_category>,
    Categories: <comma separated categories>
    Studios and Publishers: <comma separated companies>
    Supported Languages: <comma separated languages>
    Description: <description text>

    If main_category is not present, it will be set to "games".
    """
    name = f"Product Name: {product.get('name', '')},"
    main_category = product.get("main_category", "games")
    main_cat_text = f"Main category: {main_category},"
    categories = product.get("categories", [])
    categories_text = f"Categories: {', '.join(categories)}"
    companies = ", ".join(product.get("companies", []))
    companies_text = f"Studios and Publishers: {companies}"
    languages = ", ".join(product.get("supported_languages", []))
    languages_text = f"Supported Languages: {languages}"
    description = product.get("description", "").strip()
    description_text = f"Description: {description}"

    full_text = "\n".join([name, main_cat_text, categories_text, companies_text, languages_text, description_text])
    return full_text

VECTOR_SIZE = 384  # all-MiniLM-L6-v2 genrates 384-dimensional vectors
PRODUCT_COLLECTION = "products"
USER_COLLECTION = "user_profiles"

qdrant_client = QdrantClient(host="localhost", port=6333)

# Check if the collection exists and create it if not
collections = [col.name for col in qdrant_client.get_collections().collections]
if PRODUCT_COLLECTION not in collections:
    qdrant_client.recreate_collection(
        collection_name=PRODUCT_COLLECTION,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE
        )
    )

# Prepare and encode product data
prepared_texts = [prepare_text(p) for p in products]
generated_embeddings = model.encode(prepared_texts, convert_to_numpy=True)
norms = np.linalg.norm(generated_embeddings, axis=1, keepdims=True)
norms_safe = np.where(norms == 0, 1, norms)
normalized_embeddings = generated_embeddings / norms_safe

# Fetch existing product embeddings from Qdrant
product_ids = [p["id"] for p in products]
resp = qdrant_client.retrieve(
    collection_name=PRODUCT_COLLECTION,
    ids=product_ids,
    with_vectors=True,
    with_payload=True
)

existing_product_embeddings = {}
if resp and len(resp) > 0:
    for pt in resp:
        if pt.vector is not None:
            existing_product_embeddings[pt.id] = np.array(pt.vector)

# For each product, check if it exists in Qdrant
product_points_to_upsert = []
for idx, p in enumerate(products):
    pid = p["id"]
    if pid not in existing_product_embeddings:
        existing_product_embeddings[pid] = normalized_embeddings[idx]
        point = models.PointStruct(
            id=pid,
            vector=normalized_embeddings[idx].tolist(),
            payload={
                "name": p.get("name", ""),
                "companies": p.get("companies", []),
                "categories": p.get("categories", []),
                "supported_languages": p.get("supported_languages", []),
                "is_pre_order": p.get("is_pre_order", False),
                "system_requirements": p.get("system_requirements", ""),
                "description": p.get("description", "")
            }
        )
        product_points_to_upsert.append(point)

if product_points_to_upsert:
    qdrant_client.upsert(
        collection_name=PRODUCT_COLLECTION,
        wait=True,
        points=product_points_to_upsert
    )

# Global variable to store product embeddings
product_embeddings = existing_product_embeddings

if USER_COLLECTION not in collections:
    qdrant_client.recreate_collection(
        collection_name=USER_COLLECTION,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE
        )
    )

# When creating the user collection, we can also set the payload schema
def get_user_profile(user_id: str):
    try:
        result = qdrant_client.retrieve(
            collection_name=USER_COLLECTION,
            ids=[str(user_id)],
            with_vectors=True,
            with_payload=True
        )
        if result and len(result) > 0 and result[0].vector is not None:
            return np.array(result[0].vector)
        return None
    except Exception as e:
        return None

def upsert_user_profile(user_id: str, vector: np.ndarray):
    point = models.PointStruct(
        id=str(user_id),
        vector=vector.tolist(),
        payload={"user_id": str(user_id)}
    )
    qdrant_client.upsert(
        collection_name=USER_COLLECTION,
        wait=True,
        points=[point]
    )

INTERACTION_WEIGHTS = {
    "visited": 0.5,
    "bought": 1.0
}

@app.post("/interactions")
def add_interaction(user_id: str = Form(...), product_id: int = Form(...), interaction_type: str = Form(...)):
    if interaction_type not in INTERACTION_WEIGHTS:
        raise HTTPException(status_code=400, detail="Invalid interaction type. Use 'visited' or 'bought'.")
    if product_id not in product_embeddings:
        raise HTTPException(status_code=404, detail="Product not found")

    weight = INTERACTION_WEIGHTS[interaction_type]
    product_vector = product_embeddings[product_id]

    current_vector = get_user_profile(user_id)
    if current_vector is None:
        updated_vector = weight * product_vector
    else:
        updated_vector = current_vector + weight * product_vector

    norm = np.linalg.norm(updated_vector)
    if norm != 0:
        updated_vector = updated_vector / norm

    upsert_user_profile(user_id, updated_vector)
    return {"message": f"Interaction added for user {user_id} on product {product_id} with type '{interaction_type}'."}

@app.get("/user_recommendations/{user_id}")
def get_user_recommendations(user_id: str):
    user_vector = get_user_profile(user_id)
    if user_vector is None:
        raise HTTPException(status_code=404, detail="User profile not found")

    search_result = qdrant_client.search(
        collection_name=PRODUCT_COLLECTION,
        query_vector=user_vector.tolist(),
        limit=5
    )

    recommendations = []
    for point in search_result:
        recommendations.append({
            "id": point.id,
            "name": point.payload.get("name", ""),
            "companies": point.payload.get("companies", []),
            "categories": point.payload.get("categories", []),
            "supported_languages": point.payload.get("supported_languages", []),
            "is_pre_order": point.payload.get("is_pre_order", False),
            "system_requirements": point.payload.get("system_requirements", ""),
            "similarity_score": point.score,
            "description": point.payload.get("description", "")
        })
    return {"user_id": user_id, "recommendations": recommendations}

@app.get("/interactions", response_class=HTMLResponse)
def interactions_page():
    html_content = """
    <html>
      <head>
        <title>User Interactions</title>
      </head>
      <body>
        <h2>Simulate User Interaction</h2>
        <form action="/interactions" method="post">
          <label>User ID (e.g. UUID):</label><br>
          <input type="text" name="user_id" value="90871d65-4953-4278-9461-2c18d22acafa"><br><br>
          <label>Product ID:</label><br>
          <input type="number" name="product_id" value="1"><br><br>
          <label>Interaction Type:</label><br>
          <select name="interaction_type">
            <option value="visited">Visited</option>
            <option value="bought">Bought</option>
          </select><br><br>
          <input type="submit" value="Submit Interaction">
        </form>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/products")
def list_products():
    return products

@app.post("/calculate_vector")
def calculate_vector(product: dict = Body(...)):
    """
    Example input data:
    {
        "name": "Example Product",
        "main_category": "Visual Novel", // optional
        "categories": ["Action", "Adventure"],
        "companies": ["Company1", "Company2"],
        "supported_languages": ["English", "Polish"],
        "description": "Product description here\nAdditional description line."
    }
    """
    try:
        text = prepare_text(product)
        embedding = model.encode([text], convert_to_numpy=True)[0]
        norm = np.linalg.norm(embedding)
        normalized_embedding = embedding / norm if norm != 0 else embedding
        return {"vector": normalized_embedding.tolist(), "prepared_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating vector: {str(e)}")


@app.get("/recommendations/{product_id}")
def get_product_recommendations(product_id: int):
    if product_id not in product_embeddings:
        raise HTTPException(status_code=404, detail="Product not found")
    target_vector = product_embeddings[product_id]
    similarity_scores = []
    for pid, vec in product_embeddings.items():
        if pid == product_id:
            continue
        score = float(np.dot(target_vector, vec))
        similarity_scores.append((pid, score))
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = similarity_scores[:4]
    recommendations = []
    for pid, score in top_recommendations:
        product = next((p for p in products if p["id"] == pid), None)
        if product:
            recommendations.append({
                "id": product["id"],
                "name": product["name"],
                "companies": product["companies"],
                "categories": product["categories"],
                "supported_languages": product["supported_languages"],
                "is_pre_order": product["is_pre_order"],
                "system_requirements": product["system_requirements"],
                "similarity_score": score,
                "description": product["description"]
            })
    return {"product_id": product_id, "recommendations": recommendations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
