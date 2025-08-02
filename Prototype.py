import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.patches import Rectangle
import os
from PIL import Image
import torchvision.transforms as T
from attention import MultiScaleSelfAttentionLayer
    
class PrototypeNet(nn.Module):
    def __init__(self, num_classes=2, num_prototypes_per_class=5, feature_dim=1280):
        super().__init__()
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.num_prototypes = num_classes * num_prototypes_per_class
        self.feature_dim = feature_dim
        self.epsilon = 1e-4
        
        self.backbone = models.mobilenet_v2(pretrained=True).features

        self.mdsa = MultiScaleSelfAttentionLayer(dim=self.feature_dim, num_heads=4)

        # Initialize prototypes
        proto_init = torch.randn(self.num_prototypes, self.feature_dim)
        proto_init = F.normalize(proto_init, p=2, dim=1)
        self.prototype_vectors = nn.Parameter(proto_init)

        self.last_layer = nn.Linear(self.num_prototypes, num_classes, bias=False)

        prototype_class_identity = torch.zeros(self.num_prototypes, num_classes)
        for j in range(self.num_prototypes):
            prototype_class_identity[j, j // num_prototypes_per_class] = 1
        self.register_buffer('prototype_class_identity', prototype_class_identity)

        self.ones = nn.Parameter(torch.ones(1, 2048, 1, 1), requires_grad=False)

        self.attn_map = None

    def forward(self, x, activate_function = "log"):
        x = self.backbone(x)  # (B, C, H, W)
        x = self.mdsa(x)      # (B, C, H, W) 
        self.attn_map = x   
        x = F.normalize(x, p=2, dim=1)  # Normalize channel features
        
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H*W).permute(0, 2, 1)  # (B, N_patches, C)

        # Compute squared sum of x
        x2 = torch.sum(x_flat ** 2, dim=2, keepdim=True)  # (B, N_patches, 1)

        # Squared sum of prototypes, shape (num_prototypes, 1)
        p2 = torch.sum(self.prototype_vectors ** 2, dim=1).unsqueeze(1)  # (num_prototypes, 1)

        # Dot product between x and prototypes (B, N_patches, num_prototypes)
        xp = torch.matmul(x_flat, self.prototype_vectors.t())  # (B, N_patches, num_prototypes)

        # Compute distances using broadcasting (B, N_patches, num_prototypes)
        distances = x2 - 2 * xp + p2.t()  

        distances = F.relu(distances)

        # For each sample, get the minimal distance for each prototype over all patches
        softmin_weights = F.softmin(distances, dim=1)
        min_distances = (softmin_weights * distances).sum(dim=1)

        # Activation transformation
        if activate_function == "log":
            prototype_activations = torch.log((min_distances + 1) / (min_distances + self.epsilon))
        else:
            prototype_activations = -min_distances

        logits = self.last_layer(prototype_activations)  # (B, num_classes)
        probs = F.softmax(logits, dim=-1)
        
        return logits, probs, distances, prototype_activations
        
    @torch.no_grad()
    def attn_extractor(self, x, head_reduction='mean'):
        self.eval()
        _ = self.forward(x)  
 
        attn_map = self.mdsa.attn_map  # (B, num_heads, HW, HW)
        B, heads, HW1, HW2 = attn_map.shape
        S = int(HW1 ** 0.5)

        if head_reduction == 'mean':
            attn_reduced = attn_map.mean(dim=1)  # (B, HW, HW)
        elif head_reduction == 'max':
            attn_reduced = attn_map.max(dim=1)[0]
        else:
            raise ValueError("head_reduction must be 'mean' or 'max'")

        token_importance = attn_reduced.sum(dim=-1)  # (B, HW)
        attn_heatmap = token_importance.view(B, S, S)

        attn_heatmap = (attn_heatmap - attn_heatmap.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]) / \
                       (attn_heatmap.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] - attn_heatmap.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0] + 1e-8)
        return attn_heatmap


def prototype_loss(model, logits, targets, similarity, device, criterion,
                   cfg):
    output_dim = logits.shape[1]
    num_prototypes = model.prototype_vectors.shape[0]
    num_prototypes_per_class = model.num_prototypes_per_class

    # Classification loss
    loss_cls = criterion(logits, targets)

    # similarity shape: (B, N, K)
    max_similarity = similarity.max(dim=1)[0]  # (B, K)

    proto_identity = model.prototype_class_identity.to(device)  # (K, C)
    mask_correct = proto_identity[:, targets].T.bool()  # (B, K)

    # Cluster loss
    cluster_cost = -torch.mean(
        torch.max(max_similarity[mask_correct].reshape(-1, num_prototypes_per_class), dim=1)[0]
    )

    # Separation loss
    mask_wrong = ~mask_correct
    separation_cost = torch.mean(
        torch.max(max_similarity[mask_wrong].reshape(-1, (output_dim - 1) * num_prototypes_per_class), dim=1)[0]
    )

    # L1 sparsity loss and diversity loss
    l1_mask = 1 - model.prototype_class_identity.t().to(device)
    l1 = (model.last_layer.weight * l1_mask).norm(p=1)

    threshold = 0.1
    ld = 0.0 
    for k in range(output_dim):
        start = k * num_prototypes_per_class
        end = (k + 1) * num_prototypes_per_class
        p = model.prototype_vectors[start:end]
        p = F.normalize(p, p=2, dim=1)
        sim_matrix = torch.mm(p, p.t())
        eye = torch.eye(p.shape[0], device=device)
        penalty = sim_matrix - eye - threshold
        ld += torch.sum(torch.clamp(penalty, min=0.0))
    total_loss = loss_cls + cfg.method.clst * cluster_cost + cfg.method.sep * separation_cost + \
                 cfg.method.l1 * l1 + cfg.method.div * ld

    loss_dict = {
        'cls': loss_cls.item(),
        'cluster': cluster_cost.item(),
        'separation': separation_cost.item(),
        'l1': l1.item(),
        'diversity': ld.item()
    }
   
    return total_loss, loss_dict
    

@torch.no_grad()
def project_prototypes(model, data_loader, device,  use_attention = "mul"):
    dataset = data_loader.dataset
    data_indices = list(range(len(dataset)))

    new_prototypes = torch.zeros_like(model.prototype_vectors.data)
    proto_meta = []

    for proto_id in range(model.num_prototypes):
        proto_class = proto_id // model.num_prototypes_per_class
        best_score = -float("inf")
        best_feat = None
        best_match = {
            "sim": best_score,
            "img_path": None,
            "patch_row": -1,
            "patch_col": -1,
            "patch_hw": None
        }

        for j in range(0, len(data_indices)):
            data = dataset[data_indices[j]]

            if len(data) == 3:
                img, label, path = data
            else:
                img, label = data
                path = ""

            if label != proto_class:
                continue

            img = img.unsqueeze(0).to(device)  # (1, C, H, W)
            x = model.backbone(img)            # (1, C, H, W)
            x = F.normalize(x, p=2, dim=1)     # normalized feature
            B, C, H, W = x.shape

            # Compute similarity map between prototype and all positions
            sim_map = F.conv2d(
                x, 
                weight=model.prototype_vectors[proto_id].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, C, 1, 1)
            ).squeeze()  # (H, W)
            sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8) # Normalize

            # Attention fusion
            if use_attention == "add":
                alpha = 0.5 
                attn_map = model.attn_extractor(img)[0]  # (H, W)
                score_map = alpha * sim_map + (1-alpha) * attn_map
            elif use_attention == "mul":
                alpha = 1
                attn_map = model.attn_extractor(img)[0]  # (H, W)
                if attn_map.shape != sim_map.shape:
                    attn_map = F.interpolate(attn_map.unsqueeze(0).unsqueeze(0), size=sim_map.shape, mode='bilinear', align_corners=False).squeeze()
                score_map = sim_map * (attn_map ** alpha)
            else:
                score_map = sim_map

            score_flat = score_map.view(-1)
            max_score, max_idx = score_flat.max(0)

            if max_score.item() > best_score:
                best_score = max_score.item()
                feat_flat = x.view(C, -1)
                best_feat = feat_flat[:, max_idx].clone()

                best_match.update({
                    "sim": best_score,
                    "img_path": os.path.abspath(path) if isinstance(path, str) else str(path),
                    "patch_row": max_idx.item() // W,
                    "patch_col": max_idx.item() % W,
                    "patch_hw": [H, W],
                })

        if best_feat is not None:
            new_prototypes[proto_id] = F.normalize(best_feat, dim=0).view_as(model.prototype_vectors[proto_id])
        else:
            new_prototypes[proto_id] = model.prototype_vectors[proto_id].detach()

        best_match["proto_id"] = proto_id
        best_match["proto_class"] = proto_class
        proto_meta.append(best_match)

    model.prototype_vectors.data.copy_(new_prototypes)
    return proto_meta


# Visualization
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)[0]  # logits
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        score = output[:, class_idx]

        score.backward(retain_graph=True)

        gradients = self.gradients       # (B, C, H, W)
        activations = self.activations   # (B, C, H, W)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # GAP: (B, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        return cam


def explain(model, img_tensor, img_path, label, device, save_path, count):
    save_path = os.path.join(save_path, f"{count}")
    os.makedirs(save_path, exist_ok=True)

    model.eval()
    x = img_tensor.unsqueeze(0).to(device) if img_tensor.dim() == 3 else img_tensor.to(device)
    logits, probs, distances, prototype_activations = model(x)
    pred_label = torch.argmax(probs, dim=1).item()

    proto_meta = getattr(model, 'proto_meta', None)
    if proto_meta is None:
        print("Model does not have proto_meta, cannot explain.")
        return None

    proto_ids_pred_class = [p['proto_id'] for p in proto_meta if p['proto_class'] == pred_label]
    min_distances, _ = distances.min(dim=1)
    min_distances_pred_class = min_distances[0, proto_ids_pred_class]
    proto_idx = proto_ids_pred_class[torch.argmin(min_distances_pred_class).item()]
    proto_info = next((p for p in proto_meta if p['proto_id'] == proto_idx), None)

    H, W = proto_info['patch_hw']

    transform = T.Compose([T.Resize((299, 299)), T.ToTensor()])

    gradcam = GradCAM(model, target_layer=model.mdsa)  

    if pred_label == label:
        # Original image
        img = Image.open(img_path).convert("RGB")
        plt.figure(figsize=(5,5), dpi=300)
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(os.path.join(save_path, f"sample_image_protoClass_{proto_info['proto_class']}_{count}.png"),
                    bbox_inches='tight', pad_inches=0)
        plt.close()

        # Sample heatmap
        sample_tensor = transform(img).to(device)
        sample_cam = gradcam(sample_tensor.unsqueeze(0), class_idx=pred_label)
        sample_cam_tensor = torch.tensor(sample_cam).unsqueeze(0).unsqueeze(0)

        orig_w, orig_h = img.size
        sample_cam_up = F.interpolate(sample_cam_tensor, size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze().numpy()

        with torch.no_grad():
            sample_feat = model.backbone(sample_tensor.unsqueeze(0))
            if hasattr(model, 'mdsa'):
                sample_feat = model.mdsa(sample_feat)
            sample_feat = F.normalize(sample_feat, dim=1)

        proto_vec = F.normalize(model.prototype_vectors[proto_idx].detach(), dim=0)
        B, C, H, W = sample_feat.shape
        sample_feat_flat = sample_feat.view(C, H * W)
        sim_scores = torch.matmul(proto_vec.unsqueeze(0), sample_feat_flat).view(H, W).cpu().numpy()
        r_sample, c_sample = np.unravel_index(np.argmax(sim_scores), sim_scores.shape)

        patch_h_sample = orig_h / H
        patch_w_sample = orig_w / W
        x1_s = c_sample * patch_w_sample
        y1_s = r_sample * patch_h_sample
        x2_s = (c_sample + 1) * patch_w_sample
        y2_s = (r_sample + 1) * patch_h_sample

        plt.figure(figsize=(5, 5), dpi=300)
        plt.imshow(img)
        plt.imshow(sample_cam_up, cmap='jet', alpha=0.5)
        plt.gca().add_patch(Rectangle((x1_s, y1_s), x2_s - x1_s, y2_s - y1_s,
                                      edgecolor='black', fill=False, linewidth=1))
        plt.axis('off')
        plt.savefig(os.path.join(save_path, f"sample_overlay{count}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()

    print("Explanation completed.")
