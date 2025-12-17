import torch.nn as nn
import open_clip

class BioCLIPClassifier(nn.Module):
    def __init__(self, num_classes, model_name='hf-hub:imageomics/bioclip-2'):
        """
        Initializes BioCLIP model using OpenCLIP.
        """
        super().__init__()
        if not model_name.startswith('hf-hub:') and 'bioclip' in model_name:
             model_name = f'hf-hub:{model_name}'
             
        print(f"Loading BioCLIP model: {model_name}...")
        try:
            self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(model_name)
            self.tokenizer = open_clip.get_tokenizer(model_name)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            print("Falling back to ViT-B-32 for testing...")
            self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        
        # Freeze the visual encoder
        for param in self.model.visual.parameters():
            param.requires_grad = False
            
        self.visual_encoder = self.model.visual
        self.embed_dim = self.model.visual.output_dim
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        
    def forward(self, x):        
        # encode_image returns the projected embeddings
        features = self.encode_image(x)
        
        # Normalize
        features = features / features.norm(dim=-1, keepdim=True)
        
        logits = self.classifier(features)
        return logits

    def encode_image(self, x):
        return self.model.encode_image(x)

    def get_preprocess(self):
        return self.preprocess_val

