# app.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import ast
import boto3
from io import StringIO

app = Flask(__name__)

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

# List of resources and create an empty user-resource matrix
resources = ['article1', 'video1', 'exercise1', 'article2', 'video2', 'video3', 'exercise2', 'exercise3', 'exercise4', 'exercise5']
user_resource_matrix = pd.DataFrame(0, index=df['user_id'].unique(), columns=resources)

# Fill the user-resource matrix with data from the DataFrame
for _, row in df.iterrows():
    user_id = row['user_id']
    resources_used = ast.literal_eval(row['resources_used'])
    ratings = ast.literal_eval(row['ratings'])
    
    for resource, rating in zip(resources_used, ratings):
        if resource in user_resource_matrix.columns:
            user_resource_matrix.at[user_id, resource] = rating

user_resource_matrix = user_resource_matrix.fillna(0)

# Recommendation logic using K-Nearest Neighbors
def get_recommendations(user_id, test_id):
    test_data = df[df['test_id'] == test_id]
    
    if test_data.empty:
        return {"error": f"No data available for test_id {test_id}"}
    
    relevant_users = test_data['user_id'].unique()
    filtered_user_resource_matrix = user_resource_matrix.loc[relevant_users]
    user_scores = test_data[['user_id', 'scores']].set_index('user_id').to_dict()['scores']
    
    knn = NearestNeighbors(n_neighbors=3, metric='cosine')
    knn.fit(filtered_user_resource_matrix.values)
    
    target_user_idx = filtered_user_resource_matrix.index.get_loc(user_id)
    distances, indices = knn.kneighbors(filtered_user_resource_matrix.iloc[target_user_idx].values.reshape(1, -1), n_neighbors=3)
    
    recommended_resources = {}
    for idx in indices[0]:
        if idx != target_user_idx:
            neighbor_user_id = filtered_user_resource_matrix.index[idx]
            neighbor_ratings = filtered_user_resource_matrix.iloc[idx]
            score = user_scores.get(neighbor_user_id, 0)
            
            for resource, rating in neighbor_ratings.items():
                if rating > 0:
                    weighted_rating = rating * (score / 100)
                    if resource not in recommended_resources:
                        recommended_resources[resource] = []
                    recommended_resources[resource].append(weighted_rating)
    
    average_ratings = {resource: np.mean(ratings) for resource, ratings in recommended_resources.items()}
    sorted_resources = sorted(average_ratings.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_resources

# Flask route to provide recommendations based on user_id and test_id
@app.route('/recommend', methods=['GET'])
def recommend_resources():
    user_id = request.args.get('user_id', type=int)
    test_id = request.args.get('test_id', type=int)
    
    if user_id is None or test_id is None:
        return jsonify({"error": "Please provide user_id and test_id as query parameters"}), 400
    
    recommendations = get_recommendations(user_id, test_id)
    return jsonify(recommendations)

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
