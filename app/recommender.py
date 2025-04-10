from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    def __init__(self, df):
        self.df = df
        self.vectorizer_all = TfidfVectorizer(stop_words='english')
        self.vectorizer_desc = TfidfVectorizer(stop_words='english')

        self.tfidf_matrix_all = self.vectorizer_all.fit_transform(df['combined_features'])
        self.tfidf_matrix_desc = self.vectorizer_desc.fit_transform(df['description'])
        
    def recommend(self, product_id, filters=None, top_n=10):
        if product_id not in self.df['_id'].values:
            return []

        idx = self.df[self.df['_id'] == product_id].index[0]

        cosine_sim = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()

        similar_indices = cosine_sim.argsort()[-(top_n + 20):][::-1]  # extra for filtering

        similar_indices = [i for i in similar_indices if i != idx]

        if filters:
            filtered_indices = []
            for i in similar_indices:
                row = self.df.iloc[i]

                if filters.get('brand') and row['brand'].lower() != filters['brand'].lower():
                    continue
                if filters.get('category') and row['category'].lower() != filters['category'].lower():
                    continue
                if filters.get('price_min') and row['price'] < filters['price_min']:
                    continue
                if filters.get('price_max') and row['price'] > filters['price_max']:
                    continue

                filtered_indices.append(i)

                if len(filtered_indices) == top_n:
                    break

            return filtered_indices

        return similar_indices[:top_n]

    def recommend_by_content(self, product_id, filters=None, top_n=10):
        if product_id not in self.df['_id'].values:
            return []

        idx = self.df[self.df['_id'] == product_id].index[0]
        cosine_sim = cosine_similarity(self.tfidf_matrix_all[idx], self.tfidf_matrix_all).flatten()

        similar_indices = cosine_sim.argsort()[-(top_n + 20):][::-1]  # extra for filtering
        similar_indices = [i for i in similar_indices if i != idx]

        if filters:
            df_filtered = self.df.iloc[similar_indices]
            if filters.brand:
                df_filtered = df_filtered[df_filtered['brand'] == filters.brand]
            if filters.category:
                df_filtered = df_filtered[df_filtered['category'] == filters.category]
            if filters.price_min is not None:
                df_filtered = df_filtered[df_filtered['price'] >= filters.price_min]
            if filters.price_max is not None:
                df_filtered = df_filtered[df_filtered['price'] <= filters.price_max]
            return df_filtered.head(top_n).index.tolist()

        return similar_indices[:top_n]

    def recommend_by_description(self, product_id, top_n=10):
        if product_id not in self.df['_id'].values:
            return []

        idx = self.df[self.df['_id'] == product_id].index[0]
        cosine_sim = cosine_similarity(self.tfidf_matrix_desc[idx], self.tfidf_matrix_desc).flatten()

        similar_indices = cosine_sim.argsort()[-(top_n + 1):][::-1]  # skip self
        similar_indices = [i for i in similar_indices if i != idx]

        return similar_indices[:top_n]

    def recommend_by_metadata(self, product_id, top_n=10):
        if product_id not in self.df['_id'].values:
            return []

        product = self.df[self.df['_id'] == product_id].iloc[0]
        brand = product['brand']

        df_filtered = self.df[self.df['_id'] != product_id]

        # Filter priority: Brand + Category + Subcategory → Brand + Category → Brand
        recs = df_filtered[df_filtered['brand'] == brand]

        if len(recs) < top_n:
            recs = df_filtered[df_filtered['brand'] == brand]

        return recs.head(top_n).index.tolist()
