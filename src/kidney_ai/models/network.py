"""
Clinical Vision Architecture & Grad-CAM Explainability Module (EfficientNet-B0 CPU-Optimized):
Provides lightweight EfficientNet-B0 backbone with clinical classification head and Grad-CAM heatmap generation
so doctors can visually verify where the model detects kidney stones.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class KidneyStoneNet(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        in_features = self.backbone.classifier[1].in_features
        
        # Use inplace=False to prevent autograd in-place modification conflicts during training
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout_rate / 2, inplace=False),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class GradCAMExplainer:
    """Generates Grad-CAM visual attention overlays for clinical interpretability on EfficientNet."""
    def __init__(self, model: KidneyStoneNet):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        target_layer = self.model.backbone.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """Returns normalized 2D heatmap [0, 1] highlighting regions driving the prediction."""
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, target_class]
        score.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = F.relu(cam).cpu().numpy()

        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        cam_resized = cv2.resize(cam, (224, 224))
        return cam_resized

    @staticmethod
    def overlay_heatmap(original_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """Blends Grad-CAM heatmap onto the original medical scan."""
        resized_orig = cv2.resize(original_rgb, (224, 224))
        heatmap_uint8 = np.uint8(255 * heatmap)
        colored_cam = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colored_cam = cv2.cvtColor(colored_cam, cv2.COLOR_BGR2RGB)
        blended = np.uint8((1 - alpha) * resized_orig + alpha * colored_cam)
        return blended