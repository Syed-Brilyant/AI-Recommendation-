# main.py

from fastapi import FastAPI, HTTPException
from app.models import RecommendationRequest, RecommendationResponse, ProductResponse
from app.data_loader import load_data
from app.recommender import ContentRecommender

app = FastAPI(
    title="Product Recommendation API",
    description="Compatibility-Focused Recommendation Engine using FastAPI",
    version="1.1"
)

df = None
recommender = None

@app.on_event("startup")
def startup_event():
    global df, recommender
    df = load_data("structured.csv")  # Update to actual path if needed
    recommender = ContentRecommender(df)

@app.post("/recommend/", response_model=RecommendationResponse, tags=["Recommendation Engine"])
def recommend_compatible_products(request: RecommendationRequest):
    recommendations = recommender.recommend_by_content(request.product_id)

    if not recommendations:
        raise HTTPException(status_code=404, detail="Product not found or no compatible recommendations")

    products = []
    for idx in recommendations:
        row = df.iloc[idx]
        products.append(ProductResponse(**{
            "_id": row['_id'],
            "name": row['Name'],
            "brand": row['Brand'],
            "category": row['Category'],
            "subcategory": row['Subcategory'],
            "price": row['Price'],
            "pricesale": row['Price Sale'],
            "image_url": row['Image URL'],
            "image_id": row['_id']
        }))

    return RecommendationResponse(recommendations=[product.model_dump(by_alias=True) for product in products])