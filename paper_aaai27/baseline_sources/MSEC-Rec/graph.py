import pandas as pd
import numpy as np
import torch
import dgl
import math

def construct_graph(path):
    # Load matrices
    train_user_course_matrix = np.load(path)
    user_video_matrix = np.load("./data/train_uv.npy")  # User-video relationship
    course_knowledge_matrix = np.load("./data/ck.npy")  # Course-knowledge relationships
    course_video_matrix = np.load("./data/course_video.npy")  # Course-video relationships
    video_concept_matrix = np.load("./data/video_concept.npy")  # Video-concept relationships

    # Extract user-course relationships
    user_indices, course_indices = np.nonzero(train_user_course_matrix)
    user_indices_tensor = torch.tensor(user_indices, dtype=torch.int64)
    course_indices_tensor = torch.tensor(course_indices, dtype=torch.int64)

    # Extract user-video relationships
    user_indices_uv, video_indices_uv = np.nonzero(user_video_matrix)
    user_indices_uv_tensor = torch.tensor(user_indices_uv, dtype=torch.int64)
    video_indices_uv_tensor = torch.tensor(video_indices_uv, dtype=torch.int64)

    # Extract course-video relationships
    course_indices_cv, video_indices_cv = np.nonzero(course_video_matrix)
    course_indices_cv_tensor = torch.tensor(course_indices_cv, dtype=torch.int64)
    video_indices_cv_tensor = torch.tensor(video_indices_cv, dtype=torch.int64)

    # Extract course-knowledge relationships
    course_indices_ck, knowledge_indices_ck = np.nonzero(course_knowledge_matrix)
    course_indices_ck_tensor = torch.tensor(course_indices_ck, dtype=torch.int64)
    knowledge_indices_ck_tensor = torch.tensor(knowledge_indices_ck, dtype=torch.int64)

    # Extract video-concept relationships
    video_indices_vk, concept_indices_vk = np.nonzero(video_concept_matrix)
    video_indices_vk_tensor = torch.tensor(video_indices_vk, dtype=torch.int64)
    concept_indices_vk_tensor = torch.tensor(concept_indices_vk, dtype=torch.int64)


    # Define the data dictionary with all relationships
    data_dict = {
        ('user', 'uc', 'course'): (user_indices_tensor, course_indices_tensor),
        ('course', 'cu', 'user'): (course_indices_tensor, user_indices_tensor),
        ('user', 'uv', 'video'): (user_indices_uv_tensor, video_indices_uv_tensor),
        ('video', 'vu', 'user'): (video_indices_uv_tensor, user_indices_uv_tensor),
        ('course', 'cv', 'video'): (course_indices_cv_tensor, video_indices_cv_tensor),
        ('video', 'vc', 'course'): (video_indices_cv_tensor, course_indices_cv_tensor),
        ('video', 'vk', 'concept'): (video_indices_vk_tensor, concept_indices_vk_tensor),
        ('concept', 'kv', 'video'): (concept_indices_vk_tensor, video_indices_vk_tensor),
        ('course', 'ck', 'knowledge'): (course_indices_ck_tensor, knowledge_indices_ck_tensor),
        ('knowledge', 'kc', 'course'): (knowledge_indices_ck_tensor, course_indices_ck_tensor)
    }

    # Construct the heterogeneous graph
    graph = dgl.heterograph(data_dict)

    # Move graph to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)
    print("device:", graph.device)

    # Print graph information
    print("path:")
    print(path)
    print("\nNode types:", graph.ntypes)
    print("Edge types:", graph.etypes)
    print("Number of nodes per type:", {ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})
    print("Number of edges per type:", {etype: graph.num_edges(etype) for etype in graph.etypes})
    print("User node index range: 0 to", graph.num_nodes('user') - 1)
    print("Course node index range: 0 to", graph.num_nodes('course') - 1)
    print("Video node index range: 0 to", graph.num_nodes('video') - 1)
    print("Knowledge node index range: 0 to", graph.num_nodes('knowledge') - 1)

    return graph

# Construct the graph
train_graph = construct_graph('./data/train_uc.npy')

# Save the graph (optional)
dgl.save_graphs('./graph/trin_heterograph.bin', [train_graph])
