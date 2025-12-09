import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class BioCLIPClassifier(nn.Module):
    def __init__(self, num_classes, model_name='imageomics/bioclip'):
        """
        Initializes BioCLIP model using Hugging Face Transformers.
        """
        super().__init__()
        print(f"Loading BioCLIP model: {model_name}...")
        try:
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            print("Falling back to openai/clip-vit-base-patch16 for testing...")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
        # Freeze the visual encoder
        for param in self.model.vision_model.parameters():
            param.requires_grad = False
            
        self.visual_encoder = self.model.vision_model
        
        # Get embedding dimension
        self.embed_dim = self.model.config.vision_config.hidden_size
        
        # Linear Probe Head
        # We use the raw hidden state from the vision model (before projection) 
        # or the projected one?
        # BioCLIP usually uses the projected embedding for retrieval.
        # But for linear probe on "frozen encoder", usually we take the output of the encoder.
        # CLIPVisionModel output has pooler_output (projected) and last_hidden_state.
        # Let's use pooler_output which matches the contrastive space.
        self.embed_dim = self.model.config.projection_dim
        
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        
    def forward(self, x):
        # x: [batch, 3, 224, 224] (pixel_values)
        
        # Use get_image_features to get the projected embeddings (e.g. 512 dim)
        features = self.model.get_image_features(pixel_values=x)
        
        # Normalize
        features = features / features.norm(dim=-1, keepdim=True)
        
        logits = self.classifier(features)
        return logits

    def get_preprocess(self):
        # Return a transform compatible with torchvision datasets
        # CLIPProcessor is not a torchvision transform.
        # We need to wrap it.
        from torchvision import transforms
        
        # HF Processor usually handles resizing, normalization etc.
        # But for DataLoader we need a callable that takes PIL image and returns tensor.
        
        def transform(image):
            # processor returns a dict with 'pixel_values'
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs['pixel_values'].squeeze(0)
            
        return transform

