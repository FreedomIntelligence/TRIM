import torch
import torch.nn as nn
import random
import torch.nn.functional as F

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

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



class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.token_reduce_func = getattr(args, 'mm_vision_token_reduce_func', 'Pooling:2')

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

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def token_reduction(self, image_features):
        if not self.token_reduce_func:
            return image_features
        if 'Pooling' in self.token_reduce_func:
            batch_size, tokens_number, dimension = image_features.shape
            k = int(self.token_reduce_func.replace('Pooling:', ''))  # Set your desired k value here
            actual_dims = []  # To store the actual number of components used for each image
            with torch.cuda.amp.autocast(enabled=False):
                reduced_features = []
                for i in range(batch_size):
                    feature = image_features[i]
                    # Interpolate to reduce features [tokens_number, dimension] -> [k, dimension]
                    # Adjusting input to be (1, dimension, tokens_number) and output size to be (dimension, k)
                    feature = F.interpolate(feature.unsqueeze(0).permute(0, 2, 1), size=(k,), mode='linear', align_corners=False).squeeze(0).permute(1, 0)
                    reduced_features.append(feature)
                    actual_dims.append(k)
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
        # TODO: Tsne
        # TODO: Attention [CLS] [Text]
        else:
            raise ValueError(f'Unknown Token Reduction Function::{self.token_reduce_func}')
        return image_features, actual_dims

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        # NOTE: Token Reduction
        image_features, actual_dims = self.token_reduction(image_features)
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
