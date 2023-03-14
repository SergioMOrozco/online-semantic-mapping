import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def kmeans_cluster(objects, viz = False, k_list=[2,3,4,5,6,7,8,9,10]):
    #Objects is assumed to be list of poses ([x,y,z,...])
    #Returns a KMeans model
    X = np.array(objects)

    min_score = math.inf

    best_kmeans = None
    # print("objects array: ", objects)

    # determine best k value
    if viz:
        score_list = []
    for i in k_list:
        # print(i, X)
        kmeans = KMeans(n_clusters=i, n_init="auto").fit(X)
        score = silhouette_score(X, kmeans.labels_)
        # print("score: ", score)

        if viz:
            score_list.append(score)

        if score < min_score:
            best_kmeans = kmeans
            min_score = score

    if viz:
        plt.plot(list(range(1,len(score_list)+1)), score_list)
        plt.title("KMeans score vs number of clusters")
        plt.show()



    return(best_kmeans)


def relabel_clusters(kmeans_models, x, y):
    clusters = {}
    for i, label in enumerate(kmeans_models.labels_):
        cluster_name = "object_" + str(label) + "__" + y[i]  # adding two dashes before label name so that it can be extracted easily

        # Add cluster name to dictionary keys
        if not cluster_name in clusters:
            clusters[cluster_name] = []

        # add object (SE3Pose) to its correct cluster
        clusters[cluster_name].append(x[i])

    return clusters

def visualize_kmeans(k_means_model, data, colors=["red","green","blue","purple","cyan","gold","orange","aquamarine","black","crimson","wheat"]):
    predictions = k_means_model.predict(data)
    print(predictions)

    try:
        color_values = [colors[i] for i in predictions]
    except:
        raise Exception("I ran out of colors :(")

    data_x, data_y = zip(*data)

    plt.scatter(data_x, data_y, color=color_values) #Plots a scatter of the kmeans data

    for idx, v in enumerate(k_means_model.cluster_centers_):
        v_x = v[0]
        v_y = v[1]
        print(idx)
        plt.scatter(v_x,v_y,color=colors[idx], marker="^")

    plt.show()

def viz_raw_data(label_data, position_data, colors=["red","green","blue","purple","cyan","gold","orange","aquamarine","black","crimson","wheat"]):
    num_unique_labels = len(set(label_data))
    used_colors = colors[:num_unique_labels]

    color_label_info = list(zip(set(label_data), used_colors))
    print("Label2color mapping:",color_label_info)

    label_to_idx = [list(set(label_data)).index(l) for l in label_data]
    print(label_to_idx)

    idx_to_color = [colors[idx] for idx in label_to_idx]

    data_x, data_y = zip(*position_data)
    plt.scatter(data_x,data_y,color=idx_to_color)
    plt.show()



def extract_positions(data): #takes in raw_data
    return [d[:2] for d in data]

if __name__ == "__main__":
    import pickle

    data = pickle.load(open("raw_data.pkl","rb")) #get raw data from object identification
    label_data, feature_data = zip(*data)

    position_data = extract_positions(feature_data) #get only position data

    viz_raw_data(label_data, position_data)

    k_means_model = kmeans_cluster(position_data, viz=True, k_list=[4]) #best k-means model
    relabeled_data = relabel_clusters(k_means_model, position_data, label_data)

    visualize_kmeans(k_means_model, position_data)

    import code
    code.interact(local=locals())