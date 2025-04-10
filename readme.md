# PoC of Visual Novel Recommendation

This project implements a recommendation system for visual novels that leverages Qdrant to store and retrieve vector embeddings for products and user profiles. The system calculates dense embeddings using the `all-MiniLM-L6-v2` model from the Sentence Transformers library. Users can interact with products (e.g. visit or buy), and these interactions incrementally update their profile vectors. Recommendations for both products and users are then retrieved via Qdrant’s vector search capabilities.

## Features

- **Product Embeddings**: Generate and store 384-dimensional embeddings for visual novel products using a fine-tuned text concatenation strategy.
- **User Profiles**: Update user profiles incrementally based on interactions (visited, bought) without resetting previous interactions.
- **Qdrant Integration**: Store product embeddings and user profile vectors in Qdrant and use its search functionality to retrieve similar items.
- **Endpoints**:
  - `/products`: List all products.
  - `/recommendations/{product_id}`: Retrieve product recommendations based on cosine similarity.
  - `/user_recommendations/{user_id}`: Get personalized product recommendations for a given user based on their profile vector.
  - `/interactions`: HTML page to simulate user interactions (with a form to submit interaction data).
  - `/interactions` (POST): Endpoint to record user interactions and update profiles.

## Project Structure

- **app.py**: Main FastAPI application containing all endpoints, Qdrant integration logic, vector generation, and user profile handling.
- **products_data.py**: Contains the list of visual novel products with attributes including name, description, companies, categories, supported languages, pre-order status, and system requirements.
- **requirements.txt**: List of required packages for the project.
  
