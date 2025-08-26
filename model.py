import pickle
import pandas as pd
import numpy as np


def getsentimentRecommendation(user):
    user_final_rating = pickle.load(open("C:/workspace/capstone/pickle/recoomender.pkl", 'rb'))
    model = pickle.load(open("C:/workspace/capstone/pickle/final-sentiment-classification.pkl", 'rb'))
    tfidf = pickle.load(open("C:/workspace/capstone/pickle/tfidf-vectorizer.pkl", 'rb'))
    data_cleaned = pickle.load(open("C:/workspace/capstone/pickle/final_cleaned_data.pkl", 'rb'))
    if (user in user_final_rating.index):
        recommendations = list(
                user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)
        print("User recommendations from recommender system: ",recommendations)
        recommended_data = data_cleaned[data_cleaned['id'].isin(recommendations)]
        X_tfidf = tfidf.transform(recommended_data['reviews'].values.astype('str'))
        sentiment = model.predict(X_tfidf)
        recommended_data = recommended_data.copy()
        recommended_data['sentiment'] = sentiment

        # Calculate number of positive sentiments for each id
        positive_counts = recommended_data[recommended_data['sentiment'] == 1].groupby('id').size().rename('positive_count')
        # Calculate total number of reviews for each id
        total_counts = recommended_data.groupby('id').size().rename('total_count')
        # Combine into a DataFrame
        sentiment_stats = pd.concat([positive_counts, total_counts], axis=1).fillna(0)
        sentiment_stats['positive_percentage'] = sentiment_stats['positive_count'] / sentiment_stats['total_count']
        # Sort by percentage of positive sentiments and get top 5
        top_ids = sentiment_stats.sort_values(by='positive_percentage', ascending=False).head(5).index
        top_products = recommended_data[recommended_data['id'].isin(top_ids)].drop_duplicates('id')
        return top_products[["name","brand","categories","manufacturer","reviews_text"]]
    else:
        print("User not found in the dataset.")
        return None
print(getsentimentRecommendation("joshua"))   