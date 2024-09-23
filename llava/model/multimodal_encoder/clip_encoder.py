import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np

from transformers import CLIPModel, CLIPVisionModel, CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPVisionConfig, CLIPTextModelWithProjection, AutoTokenizer

import time

# Define the PCA function with autonomous dimension selection
def apply_pca(cur_image_features, variance_threshold=0.99):
    # Convert to float32 for PCA
    cur_image_features_float = cur_image_features.to(torch.float32)

    with torch.cuda.amp.autocast(enabled=False):
        # Transpose to get the shape [1024, 576] for PCA
        cur_image_features_float_t = cur_image_features_float.t()
        # Perform PCA with a larger q value
        U, S, V = torch.pca_lowrank(cur_image_features_float_t, q=576)
    # Compute the explained variance ratio
    explained_variance_ratio = (S ** 2) / (S ** 2).sum()
    cumulative_variance_ratio = torch.cumsum(explained_variance_ratio, dim=0)
    # Select the number of components that explain the desired variance
    desired_dim = torch.searchsorted(cumulative_variance_ratio, variance_threshold).item() + 1
    # Project the data to the desired number of principal components
    pca_features_float_t = torch.matmul(U[:, :desired_dim], torch.diag(S[:desired_dim]))
    # Transpose back to get the shape [desired_dim, 1024]
    pca_features_float = pca_features_float_t.t()
    # Convert back to bfloat16
    pca_features = pca_features_float.to(cur_image_features.dtype)
    return pca_features, desired_dim

def kmeans(X, K, max_iters=5, tolerance=1e-4):
    """
    Perform K-means clustering on X using K clusters.

    X: the data, a tensor of shape (N, D) where N is the number of data points and D is the dimensionality
    K: the number of clusters
    max_iters: the maximum number of iterations to perform
    tolerance: if the change in cluster centers is less than this value, stop iterating

    Returns: a tuple (centroids, labels), where centroids is a tensor of shape (K, D) containing the cluster centers,
             and labels is a tensor of shape (N,) containing the cluster assignments for each data point.

    This is a basic implementation of K-means that might need to be adjusted for your specific application.
    """
    N, D = X.size()
    centroids = X[torch.randperm(N)[:K]]
    for _ in range(max_iters):
        labels = ((X[:, None, :] - centroids[None, :, :]).pow(2).sum(dim=2)).argmin(dim=1)
        new_centroids = torch.stack([X[labels==k].mean(dim=0) for k in range(K)])
        if (new_centroids - centroids).pow(2).sum().sqrt() < tolerance:
            break
        centroids = new_centroids
    return centroids, labels

def compute_total_variation(X, centroids, labels):
    """
    Compute the total within-cluster variation for the given data and cluster assignments.

    X: the data, a tensor of shape (N, D) where N is the number of data points and D is the dimensionality
    centroids: the cluster centers, a tensor of shape (K, D)
    labels: the cluster assignments for each data point, a tensor of shape (N,)

    Returns: the total within-cluster variation, a scalar tensor

    This function is used in conjunction with the kmeans function to implement the Elbow method for determining
    the optimal number of clusters.
    """
    return sum((X[labels==k] - centroids[k]).pow(2).sum() for k in range(len(centroids)))

def compute_cosine_similarity_matrix(features):
    # Normalize the features
    normalized_features = F.normalize(features, p=2, dim=-1)
    # Compute cosine similarity matrix
    cosine_similarity_matrix = torch.matmul(normalized_features, normalized_features.transpose(-1, -2))
    return cosine_similarity_matrix

def generate_cluster_mask(similarity_matrix, init_mask, threshold):
    # Generate candidates based on the similarity matrix and initial mask
    candidates = similarity_matrix * init_mask >= threshold
    # Sum over the patches to find clusters
    cluster_mask = candidates.sum(dim=1) >= 1
    return cluster_mask


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.token_reduce_func = getattr(args, 'mm_vision_token_reduce_func', None) 
        
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)

        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        # NOTE: CLIP text encoder
        if self.token_reduce_func and 'TextSim' in self.token_reduce_func:
            if device_map:  # NOTE: eval
                self.clip_model = CLIPModel.from_pretrained(self.vision_tower_name)
                # self.vision_tower = self.clip_model.vision_model
                self.vision_tower.visual_projection = self.clip_model.visual_projection.to(self.device)#.to(self.vision_tower.weight.dtype)
                self.clip_model.requires_grad_(False)
            else:
                self.vision_tower = CLIPVisionModelWithProjection.from_pretrained(self.vision_tower_name, device_map=device_map)

            self.text_tower = CLIPTextModelWithProjection.from_pretrained(self.vision_tower_name, device_map=device_map)
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.vision_tower_name)
            self.text_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        all_image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = all_image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = all_image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features, all_image_features

    def token_reduction(self, image_features, all_image_features, text_features=None):
        if not self.token_reduce_func:
            return image_features, [image_features.shape[1]]*image_features.shape[0]
        elif 'TextSim' in self.token_reduce_func:

            batch_size, tokens_number, dimension = image_features.shape
            k = float(self.token_reduce_func.replace('TextSim:', '').replace('TextSim+:', ''))  # Set your desired k value here

            # Get image embedding
            with torch.cuda.amp.autocast(enabled=False):
                if image_features.dtype == torch.float16:   # NOTE: eval
                    image_features = image_features.to(torch.float32)
                elif image_features.dtype == torch.float32:   # NOTE: eval milebench
                    image_features = image_features.to(self.vision_tower.vision_model.post_layernorm.weight.dtype)
                # print(self.vision_tower.vision_model.post_layernorm.weight.dtype)
                nomalized_image_features = self.vision_tower.vision_model.post_layernorm(image_features)
                proj_image_features = self.vision_tower.visual_projection(nomalized_image_features)
            
            # Calculate similarity
            similarities = torch.matmul(proj_image_features, text_features.unsqueeze(2)).squeeze(2)  # [batch_size, tokens_number]

            # Apply softmax to similarities
            similarities = -similarities
            similarities = F.softmax(similarities, dim=-1)

            if k == -1:
                # Apply outlier detection to determine ratio
                def outlier_detection(sim):
                    sim_np = sim.to(dtype=torch.float32).cpu().numpy().flatten()
                    Q1 = np.percentile(sim_np, 25)
                    Q3 = np.percentile(sim_np, 75)
                    IQR = Q3 - Q1
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_indices = np.where((sim_np > upper_bound))[0]
                    ratio = len(outlier_indices) / len(sim_np)
                    return ratio

                k = outlier_detection(similarities)


            # Determine the number of tokens to keep
            num_tokens_to_keep = int(tokens_number * k)

            # Select top-k tokens by similarity scores
            topk_values, topk_indices = torch.topk(similarities, num_tokens_to_keep, dim=1, largest=True, sorted=False)

            # Use indices to gather the top-k tokens
            selected_image_features = torch.zeros_like(image_features)
            
            batch_indices = torch.arange(batch_size, device=image_features.device).unsqueeze(1)
            token_mask = torch.zeros(batch_size, tokens_number, device=image_features.device, dtype=torch.bool)

            token_mask[batch_indices, topk_indices] = True
            selected_image_features[:, :num_tokens_to_keep] = image_features[token_mask].view(batch_size, num_tokens_to_keep, -1)

            if 'TextSim+' in self.token_reduce_func:
                remaining_mask = torch.logical_not(token_mask)

                if remaining_mask.sum(dim=1).min().item() > 0:
                    # compute left tokens's mean
                    remaining_mean = (image_features * remaining_mask.unsqueeze(-1)).sum(dim=1) / remaining_mask.sum(dim=1, keepdim=True)
                else:
                    remaining_mean = torch.zeros(batch_size, dimension, device=image_features.device)
            
                # Populates the mean to the specified position
                selected_image_features[:, num_tokens_to_keep] = remaining_mean

            # Update actual_dims
            actual_dims = [num_tokens_to_keep + (1 if 'TextSim+' in self.token_reduce_func else 0)] * batch_size
            image_features = selected_image_features

            if image_features.dtype == torch.float32:   # NOTE: eval
                image_features = image_features.to(torch.float16)
        else:
            raise ValueError(f'Unknown Token Reduction Function::{self.token_reduce_func}')
        return image_features, actual_dims

    @torch.no_grad()
    def forward(self, images, texts=None):

        # Initialize text features
        text_features = None  

        if type(images) is list:  # If multiple images
            image_features = []  
            all_image_features = []  

            image_stream = torch.cuda.Stream()  
            text_stream = torch.cuda.Stream()  

            # Process images in parallel
            with torch.cuda.stream(image_stream):  
                for image in images:
                    # Extract image features
                    image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                    image_feature, all_image_feature = self.feature_select(image_forward_out)
                    image_features = image_features.to(images.dtype)  # Ensure correct dtype
                    all_image_features = all_image_features.to(images.dtype)
                    image_features.append(image_feature)
                    all_image_features.append(all_image_feature)

            # If using text similarity reduction
            if self.token_reduce_func and 'TextSim' in self.token_reduce_func:
                with torch.cuda.stream(text_stream):  # Process text in parallel
                    text_inputs = self.text_tokenizer(text=texts, return_tensors="pt", truncation=True, padding=True)
                    text_inputs = {k: v.to(device=image_features.device) for k, v in text_inputs.items()}  
                    text_features = self.text_tower(**text_inputs, output_hidden_states=False).text_embeds  

            torch.cuda.synchronize()  

        else:  # If single image
            image_stream = torch.cuda.Stream()  
            text_stream = torch.cuda.Stream()  

            # Process the image
            with torch.cuda.stream(image_stream):  
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                image_features, all_image_features = self.feature_select(image_forward_outs)
                image_features = image_features.to(images.dtype)  # Ensure correct dtype
                all_image_features = all_image_features.to(images.dtype)

            # If using text similarity reduction
            if self.token_reduce_func and 'TextSim' in self.token_reduce_func:
                # Process text in parallel
                with torch.cuda.stream(text_stream):  
                    text_inputs = self.text_tokenizer(text=texts, return_tensors="pt", truncation=True, padding=True)
                    text_inputs = {k: v.to(device=image_features.device) for k, v in text_inputs.items()}  
                    text_features = self.text_tower(**text_inputs, output_hidden_states=False).text_embeds  

            torch.cuda.synchronize()  # Synchronize streams to ensure both are done
        
        #NOTE:Perform token reduction on image and text features
        image_features, actual_dims = self.token_reduction(image_features, all_image_features, text_features)

        return image_features, actual_dims

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
