
# %%
import pandas as pd 
import numpy as np 
from itertools import combinations
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
import hdbscan
import cv2

data = pd.read_feather('data/sideview_data/1hrTRIM2024-08-29_14-01-33_SV2.tracks.feather')
data['track_id'] = [int(x) for x in data.track_id]

# %%
# functions

# Calculate Euclidean distance between all pairs of track_ids
def edgelist(frame, data):
    df = data[data.frame==frame].copy()
    df = df.reset_index()
    df.loc[:,'track_id'] = range(0, len(df.index))

    edge_list = []
    for (id1, id2) in combinations(df['track_id'].unique(), 2):
        # Get the coordinates for the two track_ids
        point1 = df[df['track_id'] == id1][['x_body', 'y_body']].values[0]
        point2 = df[df['track_id'] == id2][['x_body', 'y_body']].values[0]
        
        # Calculate the Euclidean distance
        distance = euclidean(point1, point2)
        
        # Append the result as a tuple (id1, id2, distance)
        edge_list.append((id1, id2, distance))

    # Convert edge_list to a DataFrame for easy viewing
    edge_df = pd.DataFrame(edge_list, columns=['track_id_1', 'track_id_2', 'distance'])

    #### convert to adjacency matrix
    # Get the unique track_ids
    track_ids = np.unique(edge_df[['track_id_1', 'track_id_2']].values)

    # Initialize the adjacency matrix with zeros
    adj_matrix = pd.DataFrame(0, index=track_ids, columns=track_ids, dtype=float)

    # Populate the adjacency matrix with distances
    for _, row in edge_df.iterrows():
        id1 = row['track_id_1']
        id2 = row['track_id_2']
        distance = row['distance']
        adj_matrix.at[id1, id2] = distance
        adj_matrix.at[id2, id1] = distance  # Because the matrix is symmetric

    return adj_matrix, df

# cluster adj_matrix
def DBSCAN_cluster(adj_matrix, df, eps, min_samples):
    # DBSCAN expects a condensed distance matrix for clustering
    distance_matrix = adj_matrix.values

    # Perform DBSCAN clustering
    # Set eps (maximum distance between two samples to be considered neighbours) and min_samples
    clustering = DBSCAN(eps=50, min_samples=3, metric='precomputed')
    labels = clustering.fit_predict(distance_matrix)

    # Attach cluster labels back to track_ids
    df['cluster_DBSCAN'] = labels

    return df

# cluster adj_matrix
def HDBSCAN_cluster(adj_matrix, df, eps, min_samples):
    # DBSCAN expects a condensed distance matrix for clustering
    distance_matrix = adj_matrix.values

    # Perform DBSCAN clustering
    # Set eps (maximum distance between two samples to be considered neighbours) and min_samples
    clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=3)
    labels = clusterer.fit_predict(distance_matrix)

    df['cluster_HDBSCAN'] = labels

    return df

# load frame
def pull_frame(video_path, frame_number):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    # Set the video position to the desired frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the specified frame
    ret, frame = cap.read()

    if ret:
        # The frame is now stored as a variable (NumPy array)
        frame_variable = frame.astype(np.uint8)
        print(f"Frame {frame_number} loaded successfully!")
        print("Frame shape:", frame_variable.shape)
    else:
        print(f"Error: Could not read frame {frame_number}.")
        frame_variable = None

    # Release the video capture object
    cap.release()

    return frame_variable

# Plotting the overlay

def plot_clusters(frame, df, cluster_column):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Display the frame as the background
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), extent=[0, frame.shape[1], frame.shape[0], 0])

    # Overlay the scatter plot on the image
    scatter = sns.scatterplot(x=df.x_body, y=df.y_body, hue=df.loc[:, cluster_column], ax=ax, s=40)

    # Set the aspect ratio to equal
    ax.set_aspect('equal', adjustable='box')

    # Set limits based on the actual data range and image dimensions
    ax.set_xlim(0, frame.shape[1])
    ax.set_ylim(frame.shape[0], 0)  # Invert the y-axis to match image coordinates
    plt.show()

# %%
# comparing a few different frames

video_path = 'data/sideview_data/trim.mp4'

frame0_adj, frame0_edge = edgelist(0, data)
frame100_adj, frame100_edge = edgelist(100, data)
frame1000_adj, frame1000_edge = edgelist(500, data)

frame0_df = DBSCAN_cluster(frame0_adj, frame0_edge, 48, 3)
frame100_df = DBSCAN_cluster(frame100_adj, frame100_edge, 48, 3)
frame1000_df = DBSCAN_cluster(frame1000_adj, frame1000_edge, 48, 3)

frame0_df = HDBSCAN_cluster(frame0_adj, frame0_edge, 48, 3)
frame100_df = HDBSCAN_cluster(frame100_adj, frame100_edge, 48, 3)
frame1000_df = HDBSCAN_cluster(frame1000_adj, frame1000_edge, 48, 3)

frame0 = pull_frame(video_path, 0)
frame100 = pull_frame(video_path, 100)
frame1000 = pull_frame(video_path, 500)

# %%

plot_clusters(frame0, frame0_df,'cluster_DBSCAN')
plot_clusters(frame0, frame0_df,'cluster_HDBSCAN')

plot_clusters(frame100, frame100_df,'cluster_DBSCAN')
plot_clusters(frame100, frame100_df,'cluster_HDBSCAN')

plot_clusters(frame1000, frame1000_df,'cluster_DBSCAN')
plot_clusters(frame1000, frame1000_df,'cluster_HDBSCAN')

# %%
