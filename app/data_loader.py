import pandas as pd

def load_data(path: str):
    df = pd.read_csv(path, usecols=[
        '_id', 'name', 'brand', 'category', 'subCategory',
        'description', 'images[0].url', 'price', 'priceSale', 'metaTitle'
    ])

    df.rename(columns={
        'name': 'Name',
        'brand': 'Brand',
        'category': 'Category',
        'subCategory': 'Subcategory',
        'description': 'Description',
        'images[0].url': 'Image URL',
        'price': 'Price',
        'priceSale': 'Price Sale',
        'metaTitle': 'Title'
    }, inplace=True)

    df.columns = df.columns.str.strip().str.replace('\xa0', '', regex=True)
    df.fillna('', inplace=True)

    df['Price'] = df['Price'].astype(str).str.replace('[₹,]', '', regex=True).astype(float)
    df['Price Sale'] = df['Price Sale'].astype(str).str.replace('[₹,]', '', regex=True).astype(float)

    text_columns = ['Name', 'Description', 'Brand', 'Category', 'Subcategory', 'Title']
    for col in text_columns:
        df[col] = df[col].str.lower().str.strip()

    df['Brand'] = df['Brand'].str.replace('\xa0', '', regex=False)

    df['combined_features'] = df[['Name', 'Description', 'Brand', 'Category', 'Subcategory', 'Title']].agg(' '.join, axis=1)

    return df