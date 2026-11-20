"""
Explainable AI (XAI) - Grad-CAM Implementation for Crop Disease Detection
Generates heatmaps highlighting the exact regions of leaves triggering disease classifications.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm


class GradCAM:
    """
    Grad-CAM class for extracting feature maps and calculating gradients
    with respect to target class.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module = None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        if self.target_layer is None:
            # Automatically find the last convolutional layer
            for module in reversed(list(self.model.modules())):
                if isinstance(module, torch.nn.Conv2d):
                    self.target_layer = module
                    break

        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        if self.target_layer is not None:
            self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
            self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Generates Grad-CAM heatmap for a given input tensor.
        Returns:
            heatmap (np.ndarray): 2D float array normalized between 0 and 1.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        loss = output[0, target_class]
        loss.backward()

        if self.gradients is None or self.activations is None:
            # Fallback if hooks didn't capture
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        # Global average pooling on gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        # Apply ReLU to retain only positive influence
        cam = F.relu(cam)
        
        # Upsample to input tensor resolution
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize between 0 and 1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_heatmap(self, original_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5, colormap: str = 'jet') -> Image.Image:
        """
        Superimposes the heatmap onto the original PIL image.
        """
        heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(original_image.size, Image.BILINEAR)
        try:
            import matplotlib as mpl
            cmap = mpl.colormaps[colormap]
        except (AttributeError, KeyError):
            import matplotlib.cm as cm
            cmap = cm.get_cmap(colormap)
        colored_heatmap = cmap(np.array(heatmap_resized) / 255.0)  # RGBA
        colored_heatmap = np.uint8(colored_heatmap[:, :, :3] * 255)
        
        heatmap_img = Image.fromarray(colored_heatmap)
        blended = Image.blend(original_image.convert("RGB"), heatmap_img, alpha=alpha)
        return blended

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
