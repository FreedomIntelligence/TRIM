import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np

from transformers import CLIPModel, CLIPVisionModel, CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPVisionConfig, CLIPTextModelWithProjection, AutoTokenizer

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
        # if 'TextSim' not in self.token_reduce_func:
        #     self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        # else:
        #     self.vision_tower = CLIPVisionModelWithProjection.from_pretrained(self.vision_tower_name, device_map=device_map)

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

    def token_reduction(self, image_features, all_image_features, texts=None):
        if not self.token_reduce_func:
            return image_features, [image_features.shape[1]]*image_features.shape[0]
        if 'Pooling' in self.token_reduce_func:
            batch_size, tokens_number, dimension = image_features.shape
            pooling_func_k = self.token_reduce_func.replace('Pooling:', '')
            k = int(pooling_func_k.split('-')[-1])  # Set your desired k value here
            pooling_func = pooling_func_k.split('-')[0] if '-' in pooling_func_k else 'linear'
            actual_dims = []  # To store the actual number of components used for each image
            with torch.cuda.amp.autocast(enabled=False):
                reduced_features = []
                for i in range(batch_size):
                    feature = image_features[i]
                    # Interpolate to reduce features [tokens_number, dimension] -> [k, dimension]
                    # Adjusting input to be (1, dimension, tokens_number) and output size to be (dimension, k)
                    if pooling_func == 'area':
                        old_width = old_height = int(tokens_number**0.5)
                        new_width = new_height = int(k**0.5)
                        assert new_height * new_width == k
                        # Use 2D area interpolate
                        feature = feature.view(1, dimension, old_height, old_width)
                        interpolated_feature = F.interpolate(feature, size=(new_height, new_width), mode='area')
                        feature = interpolated_feature.view(dimension, -1).permute(1, 0)
                        actual_dims.append(k)
                    elif pooling_func == 'HRarea':
                        # k=1,2,3,4,5
                        patch_nums = [1, 4, 16, 64, 256]
                        assert k > 0 and k <= len(patch_nums)
                        interpolated_features = []

                        old_width = old_height = int(tokens_number**0.5)
                        feature = feature.view(1, dimension, old_height, old_width)
                        for i in range(k):
                            patch_num = patch_nums[i]
                            new_width = new_height = int(patch_num**0.5)
                            interpolated_feature = F.interpolate(feature, size=(new_height, new_width), mode='area')
                            interpolated_feature = interpolated_feature.view(dimension, -1).permute(1, 0)
                            # interpolated_features.append(interpolated_feature)
                            interpolated_features.insert(0, interpolated_feature)
                        feature = torch.cat(interpolated_features, dim=0)
                        actual_dims.append(feature.shape[0])
                    elif pooling_func == 'linear':
                        feature = F.interpolate(feature.unsqueeze(0).permute(0, 2, 1), size=(k,), mode='linear', align_corners=False).squeeze(0).permute(1, 0)
                        actual_dims.append(k)
                    else:
                        raise ValueError(f'Unknown pooling function: {pooling_func}.')
                    reduced_features.append(feature)
                image_features = torch.stack(reduced_features).to(image_features.device).to(image_features.dtype)
        elif 'PCA' in self.token_reduce_func:
            # Apply PCA to each image in the batch
            bsz, seq_len, feature_dim = image_features.shape
            variance_threshold = float(self.token_reduce_func.replace('PCA:', ''))  # Set your desired k value here
            pca_image_features = []
            # TODO add actual_dims to llava_arch.py
            actual_dims = []  # To store the actual number of components used for each image
            for i in range(bsz):
                cur_image_features = image_features[i]
                # Apply PCA (note that cur_image_features is [576, 1024])
                cur_image_features_pca, actual_dim = apply_pca(cur_image_features, variance_threshold)
                actual_dims.append(actual_dim)
                # Pad to [576, 1024] if necessary
                pad_size = 576 - cur_image_features_pca.size(0)
                if pad_size > 0:
                    padding = torch.zeros((pad_size, feature_dim), device=cur_image_features_pca.device, dtype=cur_image_features_pca.dtype)
                    cur_image_features_pca = torch.cat((cur_image_features_pca, padding), dim=0)
                pca_image_features.append(cur_image_features_pca)
            # Stack all the padded image features to get the final tensor of shape [bsz, 576, 1024]
            image_features = torch.stack(pca_image_features)
        elif 'Kmeans' in self.token_reduce_func:
            # Apply K-means to each image in the batch
            bsz, seq_len, feature_dim = image_features.shape
            cluster_range = float(self.token_reduce_func.replace('Kmeans:', ''))
            kmeans_image_features = []
            actual_dims = []  # To store the actual number of components used for each image
            for i in range(bsz):
                cur_image_features = image_features[i]
                # Apply K-means
                K_values = list(range(int(seq_len*cluster_range), int(seq_len*(cluster_range+0.05))))  # You may want to adjust the range of K values
                total_variations = []
                last_variation = None
                best_K = None
                max_change = -float('inf')  # Initialize max_change
                for K in K_values:
                    centroids, labels = kmeans(cur_image_features, K)
                    total_variation = compute_total_variation(cur_image_features, centroids, labels)
                    if last_variation is not None and last_variation - total_variation > max_change:
                        max_change = last_variation - total_variation
                        best_K = K
                    last_variation = total_variation
                # Re-run K-means with the best K value
                centroids, labels = kmeans(cur_image_features, best_K)
                # Replace each token with its centroid
                cur_image_features = centroids[labels]
                actual_dims.append(cur_image_features.size(0))  # Save the actual number of components
                # Pad to [seq_len, feature_dim] if necessary
                pad_size = seq_len - cur_image_features.size(0)
                if pad_size > 0:
                    padding = torch.zeros((pad_size, feature_dim), device=cur_image_features.device, dtype=cur_image_features.dtype)
                    cur_image_features = torch.cat((cur_image_features, padding), dim=0)
                kmeans_image_features.append(cur_image_features)
            # Stack all the padded image features to get the final tensor of shape [bsz, seq_len, feature_dim]
            image_features = torch.stack(kmeans_image_features)
        elif 'Cluster' in self.token_reduce_func:
            bsz, seq_len, feature_dim = image_features.shape
            mask_ratio = float(self.token_reduce_func.replace('Cluster:', ''))
            threshold = 0.8  # You can adjust this value based on your requirements

            cluster_masked_features = []
            actual_dims = []  # To store the actual number of components used for each image

            for i in range(bsz):
                cur_image_features = image_features[i]

                # Compute cosine similarity matrix
                similarity_matrix = compute_cosine_similarity_matrix(cur_image_features)

                # Generate initial random mask
                init_mask = torch.rand(seq_len) < mask_ratio
                init_mask = init_mask.to(cur_image_features.device).unsqueeze(0).bool()

                # Generate cluster mask
                cluster_mask = generate_cluster_mask(similarity_matrix, init_mask, threshold)
                cluster_mask = cluster_mask.unsqueeze(0).bool()

                # Combine initial mask and cluster mask
                combined_mask = init_mask | cluster_mask

                # Apply the mask to the features
                masked_features = cur_image_features[combined_mask.squeeze(0)]

                actual_dims.append(masked_features.size(0))  # Save the actual number of components

                # Pad to [seq_len, feature_dim] if necessary
                pad_size = seq_len - masked_features.size(0)
                if pad_size > 0:
                    padding = torch.zeros((pad_size, feature_dim), device=masked_features.device, dtype=masked_features.dtype)
                    masked_features = torch.cat((masked_features, padding), dim=0)

                cluster_masked_features.append(masked_features)

            # Stack all the padded image features to get the final tensor of shape [bsz, seq_len, feature_dim]
            image_features = torch.stack(cluster_masked_features)
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

            # Get text embedding
            text_inputs = self.text_tokenizer(text=texts, return_tensors="pt", truncation=True, padding=True)
            text_inputs = {k: v.to(device=image_features.device) for k, v in text_inputs.items()}
            text_features = self.text_tower(**text_inputs, output_hidden_states=True).text_embeds

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
            for i in range(batch_size):
                selected_image_features[i, :num_tokens_to_keep] = image_features[i, topk_indices[i]]
                if 'TextSim+' in self.token_reduce_func:
                    # Calculate the mean of the remaining tokens
                    remaining_indices = torch.tensor([j for j in range(tokens_number) if j not in topk_indices[i]], device=image_features.device)
                    if remaining_indices.numel() > 0:
                        remaining_mean = image_features[i, remaining_indices].mean(dim=0)
                    else:
                        remaining_mean = torch.zeros(dimension, device=image_features.device)

                    selected_image_features[i, num_tokens_to_keep] = remaining_mean

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
        if type(images) is list:
            image_features = []
            all_image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature, all_image_feature = self.feature_select(image_forward_out)
                image_features = image_features.to(images.dtype)
                all_image_features = all_image_features.to(images.dtype)
                image_features.append(image_feature)
                all_image_features.append(all_image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features, all_image_features = self.feature_select(image_forward_outs)
            image_features = image_features.to(images.dtype)
            all_image_features = all_image_features.to(images.dtype)

        # NOTE: Token Reduction
        image_features, actual_dims = self.token_reduction(image_features, all_image_features, texts)
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
