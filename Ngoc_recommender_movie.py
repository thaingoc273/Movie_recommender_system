import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import streamlit as st

# Parameters

n = 5
mean_threshold = 4
count_threshold = 50

# Main
def main():
    (link, movie, rating, tag, rating_pivot, rating_agg) = load_data()

# Load dataset

@st.cache
def load_data():    
    link = pd.read_csv('data/ml-latest-small/links.csv')
    movie = pd.read_csv('data/ml-latest-small/movies.csv')
    rating = pd.read_csv('data/ml-latest-small/ratings.csv')
    tag = pd.read_csv('data/ml-latest-small/tags.csv')
    rating_pivot = pd.pivot_table(data=rating, values='rating', columns='movieId', index = 'userId')
    rating_agg = rating.groupby('movieId').agg(mean_rating = ('rating', 'mean'), count_rating = ('rating', 'count')).reset_index()
    return (link, movie, rating, tag, rating_pivot, rating_agg)

# (link, movie, rating, tag, rating_pivot, rating_agg)

#print('Hello to the recommendation system')

# The popularity recommendation system


def recommender_popularity(n, mean_threshold, count_threshold):
    #rating_agg = rating.groupby('movieId').agg(mean_rating = ('rating', 'mean'), count_rating = ('rating', 'count')).reset_index()
    rating_threshold = rating_agg.query('mean_rating>@mean_threshold and count_rating>@count_threshold')
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    mean_scale = minmax_scaler.fit_transform(rating_threshold[['mean_rating']])
    count_scale = minmax_scaler.fit_transform(rating_threshold[['count_rating']])
    rating_threshold.loc[:,'score'] = np.multiply(count_scale, mean_scale)
    movie_Id_recommend = rating_threshold.sort_values(by='score', ascending = False).head(n)[['movieId', 'mean_rating', 'count_rating']]
    pd.merge(movie_Id_recommend, movie, on='movieId')
    return (pd.merge(movie_Id_recommend, movie, on='movieId'))#.sort_values(by='movieId')

#print(recommender_popularity(n, mean_threshold, count_threshold))

# Second recommendation system based on items-items

def number_common_rating_by_user(movieId):
    movie_rating = rating_pivot[movieId]
    lst_number = []
    for col in rating_pivot.columns:
        sum_not_na = sum(movie_rating.notna() & rating_pivot[col].notna())
        lst_number.append(sum_not_na)
    dataframe = pd.DataFrame(lst_number,columns=['common_two_movie'], index=rating_pivot.columns)
    return dataframe    

def recommender_item_base(rating_pivot, movieId, n, mean_threshold, count_threshold):
    movie_rating = rating_pivot[movieId]
    corr_movie = rating_pivot.corrwith(movie_rating)
    corr_movie.dropna(inplace=True) # Remove na value
    df_corr = pd.DataFrame(corr_movie, columns=['Pearson_corr'])
    df_corr = df_corr.join(number_common_rating_by_user(movieId))
    df_corr.reset_index(inplace=True)
    df_corr = df_corr.merge(rating_agg)
    df_corr = df_corr.merge(movie)
    df_corr = df_corr[(df_corr['mean_rating']>=mean_threshold)&(df_corr['count_rating']>=count_threshold)]
    #df_corr = df_corr[df_corr.movieId != movieId]
    df_corr = df_corr.sort_values(by='Pearson_corr', ascending = False).head(n)    
    return df_corr


# User-user base recommendation system

def recommender_user_base(rating_pivot, n, user_id, method):
    
    if (method=='cosine'):
        rating_pivot.fillna(0, inplace=True)
        user_similarity = pd.DataFrame(cosine_similarity(rating_pivot), columns=rating_pivot.index, index=rating_pivot.index)
        weight = user_similarity.query('userId!=@user_id')[user_id]/sum(user_similarity.query('userId!=@user_id')[user_id])
        
    elif (method=='correlation'):
        user_similarity = rating_pivot.T.corr()
        user_similarity.fillna(0, inplace=True)
        weight_user = user_similarity.query('userId!=@user_id')[user_id]
        weight = weight_user
#         weight = MinMaxScaler().fit_transform(np.array(weight_user).reshape(-1, 1))
#         weight = weight.reshape(1,-1)[0]
        
    # weight = user_similarity.query('userID!=@user_id')[user_id]/sum(user_similarity.query('userID!=@user_id')[user_id])
    
    not_rating_movie = rating_pivot.loc[rating_pivot.index!=user_id, rating_pivot.loc[user_id,:]==0]
    weighted_averages = pd.DataFrame(not_rating_movie.T.dot(weight), columns=["predicted_rating"])
    recommendations = weighted_averages.merge(movie, left_index=True, right_on="movieId")
    recommendations = recommendations.sort_values("predicted_rating", ascending=False).head(n)
        
    return recommendations


st.title("Movie Recommender - Group 1")


if __name__ == "__main__":
    main()