from pydantic import BaseModel, Field
from typing import List, Optional

class RecommendationRequest(BaseModel):
    product_id: str


class ProductResponse(BaseModel):
    id: str = Field(..., alias="_id")  # To show _id in API response
    name: str
    brand: str
    category: str
    subcategory: Optional[str] = None
    price: float
    pricesale: Optional[float] = None
    image_url: Optional[str] = None
    image_id: Optional[str] = None

    class Config:
        allow_population_by_field_name = True  # Important for alias to work in API response


class RecommendationResponse(BaseModel):
    recommendations: List[ProductResponse]
