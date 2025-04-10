import pandas as pd


def load_data(path: str):
    df = pd.read_csv(path, usecols=[
        '_id', 'name', 'description', 'brand', 'category',
        'subCategory', 'price', 'priceSale',
        'images[0].url', 'images[0]._id'
    ])

    # Handle null values
    df.fillna('', inplace=True)

    # Create combined features column for recommendation
    df['combined_features'] = df[['name', 'description', 'brand',
                                  'category', 'subCategory']].agg(' '.join, axis=1)

    return df
