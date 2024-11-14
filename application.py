from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import boto3
from io import StringIO

application = Flask(__name__)

# Set up the S3 client
s3 = boto3.client('s3')

# Function to read the CSV file from S3
def read_csv_from_s3(bucket_name, file_key):
    s3_object = s3.get_object(Bucket=bucket_name, Key=file_key)
    csv_data = s3_object['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_data))
    return df

# S3 bucket and file details
bucket_name = 'awsrecom'  # Replace with your bucket name
file_key = 'dataset.csv'  # Replace with your file key

# Load data from S3
df = read_csv_from_s3(bucket_name, file_key)

# Define all possible resources
resources = ['article1', 'video1', 'exercise1', 'article2', 'video2', 'video3', 'exercise2', 'exercise3', 'exercise4', 'exercise5']
user_resource_matrix = pd.DataFrame(0, index=df['user_id'].unique(), columns=resources)

# Populate the matrix with ratings from the CSV data
for _, row in df.iterrows():
    user_id = row['user_id']
    
    # Split the semicolon-separated strings into lists
    resources_used = row['resources_used'].replace(" ", "").split(';')  # Remove extra spaces and split
    ratings = list(map(int, row['ratings'].replace(" ", "").split(';')))
    
    # Fill the matrix with the ratings
    for resource, rating in zip(resources_used, ratings):
        if resource in user_resource_matrix.columns:
            user_resource_matrix.at[user_id, resource] = rating

# Normalize the matrix (optional, helps with ML models)
user_resource_matrix = user_resource_matrix.fillna(0)

def recommend_resources(test_id):
    # Filter data based on test_id to get relevant test data
    test_data = df[df['test_id'] == test_id]
    
    # Ensure there is data for the test
    if test_data.empty:
        return {"error": f"No data available for test_id {test_id}"}
    
    # Only consider users who interacted with resources in this test
    relevant_users = test_data['user_id'].unique()
    filtered_user_resource_matrix = user_resource_matrix.loc[relevant_users]
    
    # Check if we have enough users for KNN
    if len(filtered_user_resource_matrix) < 2:
        return {"error": f"Not enough data to make recommendations for test_id {test_id}"}
    
    # Get the scores for each user in the test
    user_scores = test_data[['user_id', 'scores']].set_index('user_id').to_dict()['scores']
    
    # K-Nearest Neighbors for finding similar users
    knn = NearestNeighbors(n_neighbors=min(3, len(filtered_user_resource_matrix)), metric='cosine')
    knn.fit(filtered_user_resource_matrix.values)
    
    # Get the neighbors for all users
    recommended_resources = {}
    for user_id in filtered_user_resource_matrix.index:
        # Find the nearest neighbors (similar users)
        target_user_idx = filtered_user_resource_matrix.index.get_loc(user_id)
        distances, indices = knn.kneighbors(filtered_user_resource_matrix.iloc[target_user_idx].values.reshape(1, -1), n_neighbors=min(3, len(filtered_user_resource_matrix)))
        
        for idx in indices[0]:
            if idx != target_user_idx:  # Ignore the target user themselves
                neighbor_user_id = filtered_user_resource_matrix.index[idx]
                neighbor_ratings = filtered_user_resource_matrix.iloc[idx]
                score = user_scores.get(neighbor_user_id, 0)  # Get the score for this user
                
                for resource, rating in neighbor_ratings.items():
                    if rating > 0:  # Only consider resources with ratings
                        # Weight resources based on neighbor's score
                        weighted_rating = rating * (score / 100)  # Assuming scores are out of 100
                        if resource not in recommended_resources:
                            recommended_resources[resource] = []
                        recommended_resources[resource].append(weighted_rating)
    
    # Calculate the weighted average rating for each resource
    average_ratings = {resource: np.mean(ratings) for resource, ratings in recommended_resources.items()}
    
    # Sort by highest rating
    sorted_resources = sorted(average_ratings.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_resources

# Flask route to provide recommendations based on test_id
@application.route('/', methods=['GET'])
def recommend_resources_route():
    test_id = request.args.get('test_id', type=int)
    
    if test_id is None:
        return jsonify({"error": "Please provide test_id as a query parameter"}), 400
    
    recommendations = recommend_resources(test_id)
    return jsonify(recommendations)
