import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import streamlit as st

from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import pickle
import implicit

# Parameters

st.set_page_config(layout="wide")

# Main
def main():
    #(link, movie, rating, tag, rating_pivot, rating_agg) = load_data()
   
    st.title("Movie Recommender - Group 1")
    st.sidebar.markdown("## Choose type of recommender")
    page = st.sidebar.selectbox("", ["Popularity", "Item-item base", "User-user base", "Singular Value Decomposition (SVD)"])
    if page == "Popularity":
        n = st.sidebar.selectbox("Select number of top movies ", np.arange(4, 10, 1))
        movie_recommend = recommender_popularity(n, mean_threshold, count_threshold)
        st.markdown(f"## Top {n} popular movies")
        st.dataframe(movie_recommend)
    elif page == "Item-item base":
        movie_title = st.sidebar.selectbox('Select a movie that you like', lst_movie)
        n = st.sidebar.selectbox('Select the number of movie you want to see', np.arange(1, 10))
        movieId = movie[movie.title == movie_title]['movieId']       
        movie_recommend = recommender_item_base(movieId.to_list()[0], n, mean_threshold_item, count_threshold_item)
        #st.write(movieId)#.to_list()[0])
        st.markdown(f"## Top {n} recommended movies")
        st.dataframe(movie_recommend)
    elif page == "User-user base":
        user_id = st.sidebar.selectbox('Select the user ID', lst_user)
        n = st.sidebar.selectbox('Select the number of movie you want to see', np.arange(1, 10))
        method = st.sidebar.selectbox('Select method of similarity', ['cosine', 'correlation'])
        movie_recommend = recommender_user_base(n, user_id, method)
        st.markdown(f"## Top {n} recommended movies with {method} similarity")
        st.dataframe(movie_recommend)
    elif page == "Singular Value Decomposition (SVD)":
        method = st.sidebar.selectbox("Select item base or user base method", ["Item-item base", "User-user base"])
        if method == "Item-item base":
            movie_title = st.sidebar.selectbox('Select a movie that you like', lst_movie)
            n = st.sidebar.selectbox('Select the number of movie you want to see', np.arange(1, 10))
            movieId = movie[movie.title == movie_title]['movieId']
            #st.write(movieId.to_list()[0])
            movie_recommend = rating_item_ALS(movieId.to_list()[0], n)
            st.markdown(f"## Top {n} recommended movies")
            st.dataframe(movie_recommend)
        elif method == "User-user base":
            user_id = st.sidebar.selectbox('Select the user ID', lst_user)
            n = st.sidebar.selectbox('Select the number of movie you want to see', np.arange(1, 10))
            movie_recommend = rating_user_ALS(user_id, n)
            st.markdown(f"## Top {n} recommended movies")
            st.write(movie_recommend)        

#@st.cache
def load_data():    
    link = pd.read_csv('data/links.csv')
    movie = pd.read_csv('data/movies.csv')
    rating = pd.read_csv('data/ratings.csv')
    tag = pd.read_csv('data/tags.csv')
    rating_pivot = pd.pivot_table(data=rating, values='rating', columns='movieId', index = 'userId')
    rating_agg = rating.groupby('movieId').agg(mean_rating = ('rating', 'mean'), count_rating = ('rating', 'count')).reset_index()
    return (link, movie, rating, tag, rating_pivot, rating_agg)

# (link, movie, rating, tag, rating_pivot, rating_agg)

#print('Hello to the recommendation system')

# The popularity recommendation system

#@st.cache
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
#@st.cache
def number_common_rating_by_user(movieId):
    movie_rating = rating_pivot[movieId]
    lst_number = []
    for col in rating_pivot.columns:
        sum_not_na = sum(movie_rating.notna() & rating_pivot[col].notna())
        lst_number.append(sum_not_na)
    dataframe = pd.DataFrame(lst_number,columns=['common_two_movie'], index=rating_pivot.columns)
    return dataframe    

#@st.cache
def recommender_item_base(movieId, n, mean_threshold, count_threshold):
    movie_rating = rating_pivot[movieId]
    corr_movie = rating_pivot.corrwith(movie_rating)
    corr_movie.dropna(inplace=True) # Remove na value
    df_corr = pd.DataFrame(corr_movie, columns=['Pearson_corr'])
    df_corr = df_corr.join(number_common_rating_by_user(movieId))
    df_corr.reset_index(inplace=True)
    df_corr = df_corr.merge(rating_agg)
    df_corr = df_corr.merge(movie)
    df_corr = df_corr[(df_corr['mean_rating']>=mean_threshold)&(df_corr['count_rating']>=count_threshold)]
    df_corr = df_corr[df_corr.movieId != movieId]
    df_corr = df_corr.sort_values(by='Pearson_corr', ascending = False).head(n)    
    return df_corr


# User-user base recommendation system
#@st.cache
def recommender_user_base(n, user_id, method):
    rating_pivot_fillna = rating_pivot.fillna(0)
    
    if (method=='cosine'):        
        user_similarity = pd.DataFrame(cosine_similarity(rating_pivot_fillna), columns=rating_pivot.index, index=rating_pivot.index)
        weight = user_similarity.query('userId!=@user_id')[user_id]/sum(user_similarity.query('userId!=@user_id')[user_id])
        
    elif (method=='correlation'):
        user_similarity = rating_pivot.T.corr()
        user_similarity.fillna(0, inplace=True)
        weight_user = user_similarity.query('userId!=@user_id')[user_id]
        weight = weight_user
#         weight = MinMaxScaler().fit_transform(np.array(weight_user).reshape(-1, 1))
#         weight = weight.reshape(1,-1)[0]
        
    # weight = user_similarity.query('userID!=@user_id')[user_id]/sum(user_similarity.query('userID!=@user_id')[user_id])
    
    not_rating_movie = rating_pivot_fillna.loc[rating_pivot.index!=user_id, rating_pivot_fillna.loc[user_id,:]==0]
    weighted_averages = pd.DataFrame(not_rating_movie.T.dot(weight), columns=["predicted_rating"])
    recommendations = weighted_averages.merge(movie, left_index=True, right_on="movieId")
    recommendations = recommendations.sort_values("predicted_rating", ascending=False).head(n)
        
    return recommendations

def rating_user_ALS(user_id, n): # function to give recommendations based on userID with ALS
    rating_pivot_fillna = rating_pivot.fillna(0)
    rating_pivot_spare_matrix = csr_matrix(rating_pivot_fillna)
    model = pickle.load(open("als_model.pkl", 'rb'))
    user_recommend = model.recommend(lst_user_Id.index(user_id), rating_pivot_spare_matrix[lst_user_Id.index(user_id)], N = n)
    movie_recommend = [lst_movie_Id[i] for i in user_recommend[0]]    
    df_recommend = pd.DataFrame(np.array([movie_recommend, user_recommend[1]]).T, columns=['movieId', 'rating_als'])
    df_recommend_merg_movie = df_recommend.merge(movie, on = 'movieId', how = 'left')
    df_recommend_merg_movie['movieId'] = df_recommend_merg_movie['movieId'].astype('int32')
    return df_recommend_merg_movie

def rating_item_ALS(movie_Id, n_similar):
    model = pickle.load(open('als_model.sav', 'rb'))
    movie_similar = model.similar_items(lst_movie_Id.index(movie_Id), n_similar+1) # Get index of movie_Id to refer to column in sparse matrix
    movie_recommend = [lst_movie_Id[i] for i in movie_similar[0]] # Back from index to movieId
    df_recommend = pd.DataFrame(np.array([movie_recommend, movie_similar[1]]).T, columns=['movieId', 'movie_corr'])
    df_recommend = df_recommend[df_recommend['movieId'] != movie_Id]
    df_recommend_merg_movie = df_recommend.merge(movie, on = 'movieId', how = 'left')
    df_recommend_merg_movie['movieId'] = df_recommend_merg_movie['movieId'].astype('int32')
    return df_recommend_merg_movie

def rating_model_ALS(factors, regularization, alpha, iterations, rating_pivot_spare_matrix):
    model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=regularization, alpha=alpha, iterations=iterations)   
    model.fit(rating_pivot_spare_matrix)
    return model

if __name__ == "__main__":
    n = 5
    mean_threshold = 4
    count_threshold = 50
    mean_threshold_item = 3
    count_threshold_item = 20
    (link, movie, rating, tag, rating_pivot, rating_agg) = load_data()
    lst_user_Id = rating_pivot.index.to_list()
    lst_movie_Id = rating_pivot.columns.to_list()
    lst_movie = movie['title'].to_list()
    lst_user = rating['userId'].unique()
    main()