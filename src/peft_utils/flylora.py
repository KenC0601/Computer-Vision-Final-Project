import torch
import torch.nn as nn
import torch.nn.functional as F

class FlyLoRALayer(nn.Module):
    """
    FlyLoRA Linear Layer
    Implements the FlyLoRA method with implicit routing via fixed sparse random projection.
    """
    def __init__(self, in_features, out_features, r=32, k=8, lora_alpha=None, dropout=0.1, sparsity_ratio=None, bias_lr=1e-3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r  # total rank
        self.k = k  # activated ranks
        self.alpha = lora_alpha or (2.0 * r)  # scaling factor, default 2r as in LoRA
        self.bias_lr = bias_lr  # learning rate for bias update
        self.sparsity_ratio = sparsity_ratio or (k / r)  # sparsity ratio ρ = p/n
        self.dropout = nn.Dropout(p=dropout)

        # Fixed sparse random projection A ∈ R^{r×n}
        # Each row has exactly p non-zero entries sampled from N(0, 1/r^2)
        A = torch.zeros(r, in_features)
        p = max(1, int(in_features * self.sparsity_ratio))  # number of non-zero entries per row
        
        for i in range(r):
            # Randomly select p indices for non-zero entries
            indices = torch.randperm(in_features)[:p]
            # Initialize selected entries with normal distribution
            A[i, indices] = torch.randn(p) * (1.0 / r)
        
        self.register_buffer("A", A)  # frozen during training

        # Trainable up-projection B ∈ R^{m×r}
        self.B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.zeros_(self.B)

        # Expert-wise bias term for load balancing d ∈ R^r
        self.d = nn.Parameter(torch.zeros(r), requires_grad=False)

    def forward(self, x):
        """
        Forward pass of FlyLoRA:
        1. Project input through frozen sparse A: y = A x
        2. Add expert bias for routing: y' = y + d
        3. Select top-k experts based on |y'|
        4. Compute output using only activated experts in B
        """
        x = self.dropout(x)

        # Project input through frozen sparse A
        y = F.linear(x, self.A)  # (batch_size, r)
        
        # Add expert bias for routing
        y_biased = y + self.d  # (batch_size, r)
        
        # Select top-k experts based on magnitude
        _, selected_experts = torch.topk(y_biased.abs(), self.k, dim=-1)  # (batch_size, k)
        
        # Create mask for activated experts
        mask = torch.zeros_like(y_biased)  # (batch_size, r)
        mask.scatter_(-1, selected_experts, 1.0)  # set top-k positions to 1
        
        # Update assignment counts for load balancing
        if self.training:
            # Flatten batch dimension to count usage across all samples in batch
            # selected_experts shape: [batch, k]
            ci = torch.bincount(selected_experts.flatten(), minlength=self.r).float()
            delta_bias = (ci.mean() - ci).sign()
            self.d.data = self.d.data + self.bias_lr * delta_bias
            
        # Compute output using only activated experts
        activated_y = y * mask
        output = F.linear(activated_y, self.B) * (self.alpha / self.r)
        
        return output

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

    @property
    def weight(self):
        return self.original_layer.weight

    @property
    def bias(self):
        return self.original_layer.bias

def apply_flylora(model, r=16, k=8, dropout=0.1, target_modules=["c_fc", "out_proj"]):

    print(f"Applying FlyLoRA with r={r}, k={k}, dropout={dropout}, targets={target_modules}...")
    
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
                    k=k,
                    dropout=dropout
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
