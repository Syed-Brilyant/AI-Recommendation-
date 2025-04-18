# recommender.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

class ContentRecommender:
    def __init__(self, df):
        self.df = df
        self.vectorizer_all = TfidfVectorizer(stop_words='english')
        self.vectorizer_desc = TfidfVectorizer(stop_words='english')

        self.tfidf_matrix_all = self.vectorizer_all.fit_transform(df['combined_features'])
        self.tfidf_matrix_desc = self.vectorizer_desc.fit_transform(df['Description'])

    def recommend_by_content(self, product_id, top_n=10):
        if product_id not in self.df['_id'].values:
            return []

        idx = self.df[self.df['_id'] == product_id].index[0]
        selected = self.df.iloc[idx]

        category = selected['Category']
        subcategory = selected['Subcategory']
        brand = selected['Brand']
        name = selected['Name']

        compatibility_map = {
            'mobiles': ['tempered glass', 'mobile backcase', 'back case', 'charger', 'earphone/headphone', 'earphones/headphones'],
            'tablet': ['tempered glass', 'charger', 'earphone/headphone', 'earphones/headphones'],
            'laptop': ['charger', 'earphone/headphone', 'earphones/headphones'],
        }

        compatible_keywords = compatibility_map.get(category, [])

        results = []
        for i, row in self.df.iterrows():
            if i == idx or row['Category'] != 'accessories':
                continue

            text = f"{row['Name']} {row['Description']} {row['Subcategory']} {row['Title']}"
            is_compatible = any(keyword in text for keyword in compatible_keywords)

            sim_score = cosine_similarity(self.tfidf_matrix_all[idx], self.tfidf_matrix_all[i])[0][0]
            brand_boost = 1 if brand in text or name in text else 0
            score = sim_score + brand_boost + (1 if is_compatible else 0)

            results.append((i, score, is_compatible))

        # Sort compatible products first by score, then others
        results.sort(key=lambda x: (x[2], x[1]), reverse=True)

        # Select top_n results from sorted list
        top_results = results[:top_n]

        return [i for i, _, _ in top_results]
