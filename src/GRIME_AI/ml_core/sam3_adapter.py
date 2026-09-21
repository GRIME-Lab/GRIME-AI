# sam3_adapter.py
"""
SAM3 Adapter - Wraps SAM3 model to provide SAM2-compatible interface for training

This adapter allows SAM3 to be trained using the same trainer code as SAM2.
It implements the required interface methods that the trainer expects.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from omegaconf import OmegaConf
from hydra.utils import instantiate


class SAM3Adapter(torch.nn.Module):
    """
    Adapter that wraps SAM3 model to provide SAM2-compatible interface.

    The trainer expects these methods:
    - encode_image(): Encode image and store features
    - prompt_embeddings(): Get sparse/dense embeddings
    - decode_masks(): Decode masks from features + embeddings
    - postprocess_masks(): Upsample masks to original resolution
    """

    def __init__(self, model_cfg, checkpoint_path, device):
        super().__init__()

        # Clean config
        raw_cfg = OmegaConf.to_container(model_cfg, resolve=True)
        for key in ["device", "no_obj_embed_spatial", "use_signed_tpos_enc_to_obj_ptrs"]:
            raw_cfg.pop(key, None)

        new_cfg = OmegaConf.create(raw_cfg)
        self.model = instantiate(new_cfg, _recursive_=True)

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt["model"] if "model" in ckpt else ckpt
        self.model.load_state_dict(state, strict=False)

        self.model.to(device)
        self.model.train()

        self.device = device
        self._features = {}
        self._orig_hw = []

        print(f"✓ SAM3 model loaded from {checkpoint_path}")

    def encode_image(self, image_np: np.ndarray) -> None:
        """
        Encode image and store features for later mask decoding.

        Args:
            image_np: RGB image as numpy array [H, W, 3]

        This method should:
        1. Preprocess image (resize, normalize, convert to tensor)
        2. Run through image encoder
        3. Store features in self._features
        4. Store original image size in self._orig_hw
        """
        # Store original size
        h, w = image_np.shape[:2]
        self._orig_hw = [(h, w)]

        # Preprocess image
        # SAM models typically expect 1024x1024 input
        image_tensor = self._preprocess_image(image_np)

        # Encode image through SAM3's image encoder
        # Adjust this based on actual SAM3 architecture
        with torch.no_grad():
            # Example structure - adjust to match SAM3's actual interface
            if hasattr(self.model, 'image_encoder'):
                image_embed = self.model.image_encoder(image_tensor)
            elif hasattr(self.model, 'vision_encoder'):
                image_embed = self.model.vision_encoder(image_tensor)
            else:
                # Fallback: try to find encoder
                raise AttributeError("Cannot find image encoder in SAM3 model. "
                                     "Please update encode_image() to match SAM3 architecture.")

        # Store features
        # Adjust structure to match what SAM3 actually returns
        self._features = {
            "image_embed": [image_embed],  # Main image embedding
            "high_res_feats": [],  # High-res features if available
        }

        # If SAM3 has multi-scale features, extract them here
        # Example:
        # if hasattr(self.model, 'get_high_res_features'):
        #     self._features["high_res_feats"] = self.model.get_high_res_features()

    def _preprocess_image(self, image_np: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for SAM3 input.

        Args:
            image_np: RGB image [H, W, 3] in range [0, 255]

        Returns:
            Preprocessed tensor [1, 3, H', W']
        """
        # Convert to tensor
        image = torch.from_numpy(image_np).permute(2, 0, 1).float()  # [3, H, W]

        # Normalize (SAM uses ImageNet normalization)
        mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        image = (image - mean) / std

        # Resize to SAM3's expected input size (typically 1024x1024)
        target_size = 1024
        h, w = image.shape[1:]

        # Pad to square
        if h != w:
            max_dim = max(h, w)
            pad_h = max_dim - h
            pad_w = max_dim - w
            image = F.pad(image, (0, pad_w, 0, pad_h), value=0)

        # Resize to target
        if image.shape[1] != target_size or image.shape[2] != target_size:
            image = F.interpolate(
                image.unsqueeze(0),
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # Add batch dimension and move to device
        image = image.unsqueeze(0).to(self.device)  # [1, 3, 1024, 1024]

        return image

    def prompt_embeddings(
            self,
            points: Optional[torch.Tensor] = None,
            boxes: Optional[torch.Tensor] = None,
            masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get sparse and dense prompt embeddings.

        For training without prompts (like your use case), this returns empty embeddings.

        Args:
            points: Point prompts [B, N, 2]
            boxes: Box prompts [B, 4]
            masks: Mask prompts [B, 1, H, W]

        Returns:
            sparse_embeddings: Sparse prompt embeddings
            dense_embeddings: Dense prompt embeddings
        """
        # For training without prompts, return appropriate empty embeddings
        # Adjust dimensions based on SAM3's actual requirements

        batch_size = 1

        if hasattr(self.model, 'prompt_encoder'):
            # Use SAM3's prompt encoder
            return self.model.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=masks
            )
        else:
            # Fallback: create empty embeddings
            # Adjust dimensions to match SAM3's decoder expectations
            embed_dim = 256  # Typical SAM embedding dimension

            # Sparse embeddings (empty for no prompts)
            sparse_embeddings = torch.zeros(
                (batch_size, 0, embed_dim),
                device=self.device
            )

            # Dense embeddings (zeros for no prompt mask)
            dense_embeddings = torch.zeros(
                (batch_size, embed_dim, 64, 64),  # Typical dense embedding size
                device=self.device
            )

            return sparse_embeddings, dense_embeddings

    def decode_masks(
            self,
            multimask_output: bool = False,
            repeat_image: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode masks from stored features and prompt embeddings.

        Args:
            multimask_output: If True, return 3 masks. If False, return 1 mask.
            repeat_image: Whether to repeat image features for batch processing

        Returns:
            low_res_masks: Low-resolution mask logits [B, N, H, W]
            scores: Predicted IoU scores [B, N]
        """
        # Get prompt embeddings (empty for training without prompts)
        sparse_embeddings, dense_embeddings = self.prompt_embeddings()

        # Get stored image features
        if "image_embed" not in self._features or len(self._features["image_embed"]) == 0:
            raise RuntimeError("Image features not found. Call encode_image() first.")

        image_embed = self._features["image_embed"][-1]  # Most recent embedding

        # Decode masks using SAM3's mask decoder
        # Adjust this based on actual SAM3 architecture
        if hasattr(self.model, 'mask_decoder'):
            # Use SAM3's mask decoder
            # Get image positional encoding if needed
            if hasattr(self.model, 'prompt_encoder') and hasattr(self.model.prompt_encoder, 'get_dense_pe'):
                image_pe = self.model.prompt_encoder.get_dense_pe()
            else:
                image_pe = None

            # Get high-res features if available
            high_res_features = self._features.get("high_res_feats", None)

            # Decode masks
            low_res_masks, scores, _, _ = self.model.mask_decoder(
                image_embeddings=image_embed.unsqueeze(0) if image_embed.dim() == 3 else image_embed,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                repeat_image=repeat_image,
                high_res_features=high_res_features,
            )
        else:
            # Fallback: try to find decoder
            raise AttributeError("Cannot find mask decoder in SAM3 model. "
                                 "Please update decode_masks() to match SAM3 architecture.")

        return low_res_masks, scores

    def postprocess_masks(
            self,
            low_res_masks: torch.Tensor,
            orig_hw: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Upsample low-resolution masks to original image size.

        Args:
            low_res_masks: Low-res mask logits [B, N, H_lr, W_lr]
            orig_hw: Original image size (height, width)

        Returns:
            High-res mask logits [B, N, H, W]
        """
        return F.interpolate(
            low_res_masks,
            size=orig_hw,
            mode="bilinear",
            align_corners=False
        )


class SAM3ImagePredictor:
    """
    Predictor interface that matches SAM2ImagePredictor API.

    This is used during validation to generate predictions.
    """

    def __init__(self, sam3_adapter: SAM3Adapter):
        self.model = sam3_adapter
        self._features = {}
        self._orig_hw = []

    def set_image(self, image_np: np.ndarray) -> None:
        """
        Encode image and prepare for prediction.

        Args:
            image_np: RGB image as numpy array [H, W, 3]
        """
        self.model.encode_image(image_np)
        self._features = self.model._features
        self._orig_hw = self.model._orig_hw

    def predict(
            self,
            point_coords: Optional[np.ndarray] = None,
            point_labels: Optional[np.ndarray] = None,
            multimask_output: bool = False
    ) -> Tuple[list, np.ndarray, list]:
        """
        Generate mask predictions.

        Args:
            point_coords: Point prompts [N, 2]
            point_labels: Point labels [N] (1=foreground, 0=background)
            multimask_output: If True, return 3 masks. If False, return 1 mask.

        Returns:
            masks: List of binary masks (after sigmoid)
            scores: IoU scores for each mask
            logits: List of raw logits (before sigmoid)
        """
        # Decode masks
        low_res_masks, scores = self.model.decode_masks(
            multimask_output=multimask_output,
            repeat_image=True
        )

        # Postprocess to original size
        high_res_masks = self.model.postprocess_masks(
            low_res_masks,
            self._orig_hw[-1]
        )

        # Convert to numpy
        # Apply sigmoid to get probabilities
        masks = torch.sigmoid(high_res_masks).squeeze(0).cpu().numpy()

        # Get logits (before sigmoid)
        logits = low_res_masks.squeeze(0).cpu().numpy()

        # Get scores
        scores_np = torch.sigmoid(scores).cpu().numpy()

        # Return as lists (matches SAM2 interface)
        return [masks], scores_np, [logits]


# ======================================================================================================================
# USAGE NOTES FOR YOUR IMPLEMENTATION
# ======================================================================================================================
"""
TO COMPLETE THIS ADAPTER, YOU NEED TO:

1. **Update encode_image()** (line 66):
   - Find SAM3's actual image encoder (might be named differently)
   - Extract any high-resolution features if SAM3 uses them
   - Adjust preprocessing if SAM3 uses different normalization

2. **Update prompt_embeddings()** (line 126):
   - Find SAM3's prompt encoder (or equivalent)
   - Return correct embedding dimensions

3. **Update decode_masks()** (line 170):
   - Find SAM3's mask decoder
   - Match the exact interface (parameters might differ)
   - Handle any SAM3-specific features

4. **Test the adapter**:
   ```python
   # Test code:
   import numpy as np
   from omegaconf import OmegaConf

   # Load config
   cfg = OmegaConf.load("path/to/sam3_config.yaml")

   # Create adapter
   adapter = SAM3Adapter(
       model_cfg=cfg.model,
       checkpoint_path="path/to/checkpoint.pt",
       device=torch.device("cuda")
   )

   # Test encoding
   test_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
   adapter.encode_image(test_image)

   # Test decoding
   masks, scores = adapter.decode_masks(multimask_output=False)
   print(f"Mask shape: {masks.shape}")
   print(f"Score shape: {scores.shape}")
   ```

5. **Common SAM3 architecture differences**:
   - Different encoder name: `vision_encoder`, `image_encoder`, `backbone`
   - Different decoder name: `mask_decoder`, `decoder`, `sam_decoder`
   - Different prompt encoder: `prompt_encoder`, `prompt_embed`
   - Different feature extraction: Some models use FPN, some don't

6. **Debugging tips**:
   - Print `dir(self.model)` to see available attributes
   - Check SAM3's forward pass to understand data flow
   - Compare with SAM2's architecture if available
   - Add print statements to track tensor shapes
"""