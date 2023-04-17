from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

def platform_checker(row, platform):
    if row['id'][0] == platform[0].lower():
        return True
    else:
        return False
    
df = pd.read_csv("data_transformed.csv")
title_ids = pd.read_csv('title_ids.csv', index_col=0)
model_item_features = np.loadtxt('item_features.csv', delimiter=',')

@app.get("/largest_duration_movie/{release_year}/{platform}/{duration_type}")
async def get_max_duration(release_year: int, platform: str, duration_type: str):
    #Get the record with the highest duration_int
    titles_with_conditions = df[((df['type'] == 'movie') & (df['release_year'] == release_year) & (df.duration_type == duration_type))]
    platform_mask = titles_with_conditions.apply(lambda x: platform_checker(x, platform=platform), axis=1)
    titles_with_conditions = titles_with_conditions[platform_mask]
    title_largest_duration = titles_with_conditions[titles_with_conditions.duration_int == titles_with_conditions.duration_int.max()]
    title_largest_duration = title_largest_duration['title'].to_list()[0]
    return {'movie' : title_largest_duration}

@app.get('/get_score_count/{release_year}/{platform}/{scored}')
#Pendiente crear la columna score (Promedio de ratings de usuarios)
async def get_score_count(release_year: int, platform: str, scored: float):
    titles_with_conditions = df[((df['type'] == 'movie') & (df['release_year'] == release_year) & (df['score'] > scored))] 
    platform_mask = titles_with_conditions.apply(lambda x: platform_checker(x, platform=platform), axis=1)
    titles_with_conditions = titles_with_conditions[platform_mask]
    year_platform_score_total_movies = titles_with_conditions.shape[0]

    return {'platform': platform,
            'count' : year_platform_score_total_movies,
            'year' : release_year,
            'scored' : scored}

@app.get('/get_count_platform/{platform}')
async def get_count_platform(platform: str):
    titles_with_conditions = df[(df['type'] == 'movie')]
    platform_mask = titles_with_conditions.apply(lambda x: platform_checker(x, platform=platform), axis=1)
    titles_with_conditions = titles_with_conditions[platform_mask]
    platform_total_movies = titles_with_conditions.shape[0]

    return {'platform': platform,
            'count' : platform_total_movies}

@app.get('/get_actor/{release_year}/{platform}') #Hulu doens't have cast data
async def get_actor(release_year: int, platform: str):
    titles_with_conditions = df[(df['release_year'] == release_year)]
    platform_mask = titles_with_conditions.apply(lambda x: platform_checker(x, platform=platform), axis=1)
    titles_with_conditions = titles_with_conditions[platform_mask]
    df_cast = titles_with_conditions[['cast', 'release_year', 'id']].copy()
    df_cast['cast'] = df_cast['cast'].str.split(', ')
    df_cast = df_cast.explode('cast')
    most_recurring_actor = df_cast['cast'].value_counts().index[0]
    n_appereances = int(df_cast['cast'].value_counts()[0])

    return {'platform' : platform,
            'year' : release_year, 
            'actor' : most_recurring_actor,
            'appereances' : n_appereances}

@app.get("/prod_per_county/{release_year}/{country}/{type}")
async def prod_per_county(release_year: int, country: str, type: str):
    titles_with_conditions = df[((df['type'] == type) & (df['release_year'] == release_year) & (df['country'] == country))]
    count_of_prods = int(titles_with_conditions['id'].count())
                        
    return {'country': country,
            'year': release_year,
            'type': type,
            'production_count': count_of_prods}

@app.get('/get_contents/{rating}')
async def get_contents(rating: str):
    titles_with_conditions = df[(df['rating'] == rating.lower())]
    rating_total_movies = titles_with_conditions.shape[0]

    return {'rating' : rating,
            'count' : rating_total_movies}

@app.get('/get_similar/{title}')
async def get_top_5_similar_movies(movie_title):
    # Get the movie id of the given movie title
    movie_inner_id = title_ids[title_ids['title'] == movie_title].index[0]
    
    # Compute cosine similarity between the movie and all other movies
    similarities = cosine_similarity(model_item_features[movie_inner_id].reshape(1,-1), model_item_features)[0]
    
    # Get the indices of the top_5 most similar movies
    top_5_indices = np.argsort(similarities)[-5-1:-1][::-1]
    
    # Get the titles of the top_5 most similar movies
    top_5_titles = title_ids.iloc[top_5_indices]['title'].values
    
    return {'movie': movie_title, 
            'simlar_movies': list(top_5_titles)}
    