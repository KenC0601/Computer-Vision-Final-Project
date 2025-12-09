
import torch
import torch.nn as nn
import math

class FlyLoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r=16, lora_alpha=16, dropout=0.1, k=4):
        """
        FlyLoRA Layer: Implicit Rank-Wise Mixture-of-Experts.
        
        Args:
            k (int): Number of active experts (ranks) per input.
        """
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.dropout = nn.Dropout(p=dropout)
        self.k = k
        
        if k > r:
            raise ValueError(f"k ({k}) must be <= r ({r})")
        
        # Down projection (A)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        
        # Up projection (B)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        
        # Sparse Random Projection Router
        # Projects low-rank features to a space (here size r for simplicity/rank-wise)
        # to determine which ranks to activate.
        # Frozen random weights.
        self.router = nn.Linear(r, r, bias=False)
        self.router.weight.requires_grad = False
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        # Initialize router with random weights (e.g. Gaussian)
        nn.init.normal_(self.router.weight, mean=0.0, std=1.0)

    def forward(self, x):
        # x: [batch, tokens, in_features]
        
        # Down project
        h = self.lora_A(self.dropout(x)) # [batch, tokens, r]
        
        # Router (FlyHash-like)
        # 1. Project
        router_logits = self.router(h) # [batch, tokens, r]
        
        # 2. k-Winners-Take-All (Top-k)
        topk_values, topk_indices = torch.topk(router_logits, self.k, dim=-1)
        
        # Create mask
        mask = torch.zeros_like(router_logits)
        mask.scatter_(-1, topk_indices, 1.0)
        
        # Apply mask to h
        # This effectively selects which ranks (columns of B) are used.
        h_sparse = h * mask
        
        # Up project
        output = self.lora_B(h_sparse) # [batch, tokens, out_features]
        
        return output * self.scaling

class FlyLoRAWrapper(nn.Module):
    def __init__(self, original_layer, fly_layer):
        super().__init__()
        self.original_layer = original_layer
        self.fly_layer = fly_layer
        
        # Freeze original
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
            
    def forward(self, x):
        return self.original_layer(x) + self.fly_layer(x)

def apply_flylora(model, r=16, k=8, target_modules=["c_fc", "out_proj"]):
    """
    Applies FlyLoRA to the model by replacing target Linear layers.
    """
    print(f"Applying FlyLoRA with r={r}, k={k}, targets={target_modules}...")
    
    # Helper to replace modules
    def replace_module(module, name_path=""):
        for name, child in module.named_children():
            full_name = f"{name_path}.{name}" if name_path else name
            
            if isinstance(child, nn.Linear) and any(t in name for t in target_modules):
                # Replace
                print(f"Replacing {full_name} with FlyLoRA")
                fly_layer = FlyLoRALayer(
                    child.in_features, 
                    child.out_features, 
                    r=r, 
                    k=k
                )
                
                wrapper = FlyLoRAWrapper(child, fly_layer)
                setattr(module, name, wrapper)
                
            else:
                replace_module(child, full_name)

    replace_module(model)
    
    # Ensure classifier is trainable
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        elif "fly_layer" in name:
            if "router" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        else:
            param.requires_grad = False
            
    return model
