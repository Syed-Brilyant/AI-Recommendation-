from fastapi import FastAPI, HTTPException
from app.models import RecommendationRequest, RecommendationResponse, ProductResponse
from app.data_loader import load_data
from app.recommender import ContentRecommender

app = FastAPI(
    title="Product Recommendation API",
    description="Content Based Recommendation Engine using FastAPI",
    version="1.0"
)

df = None
recommender = None


@app.on_event("startup")
def startup_event():
    global df, recommender
    df = load_data("local.csv")
    recommender = ContentRecommender(df)


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendation Engine"])
def recommend_products(request: RecommendationRequest):
    recommendations = recommender.recommend_by_content(request.product_id)

    if not recommendations:
        raise HTTPException(status_code=404, detail="Product not found")

    products = []
    for idx in recommendations:
        row = df.iloc[idx]
        products.append(ProductResponse(**{
            "_id": row['_id'],
            "name": row['name'],
            "brand": row['brand'],
            "category": row['category'],
            "subcategory": row['subCategory'],
            "price": row['price'],
            "pricesale": row['priceSale'],
            "image_url": row['images[0].url'],
            "image_id": row['images[0]._id']
        }))

    return RecommendationResponse(recommendations=[product.model_dump(by_alias=True) for product in products])


@app.post("/recommend/description", response_model=RecommendationResponse)
def recommend_products_by_description(request: RecommendationRequest):
    indices = recommender.recommend_by_description(request.product_id)

    if not indices:
        raise HTTPException(status_code=404, detail="Product not found or no recommendations")

    products = []
    for idx in indices:
        row = df.iloc[idx]
        products.append(ProductResponse(**{
            "_id": row['_id'],
            "name": row['name'],
            "brand": row['brand'],
            "category": row['category'],
            "subcategory": row['subCategory'],
            "price": row['price'],
            "pricesale": row['priceSale'],
            "image_url": row['images[0].url'],
            "image_id": row['images[0]._id']
        }))

    return RecommendationResponse(recommendations=[product.model_dump(by_alias=True) for product in products])


@app.post("/recommend/metadata", response_model=RecommendationResponse)
def recommend_metadata(request: RecommendationRequest):
    indices = recommender.recommend_by_metadata(request.product_id)

    if not indices:
        raise HTTPException(status_code=404, detail="Product not found or no recommendations")

    products = []
    for idx in indices:
        row = df.iloc[idx]
        products.append(ProductResponse(**{
            "_id": row['_id'],
            "name": row['name'],
            "brand": row['brand'],
            "category": row['category'],
            "subcategory": row['subCategory'],
            "price": row['price'],
            "pricesale": row['priceSale'],
            "image_url": row['images[0].url'],
            "image_id": row['images[0]._id']
        }))

    return RecommendationResponse(recommendations=[product.model_dump(by_alias=True) for product in products])
