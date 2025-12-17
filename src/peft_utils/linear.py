def apply_linear_probe(model):
    
    print("Applying Linear Probe (Freezing encoder, unfreezing classifier)...")
    
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze classifier
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
            
    return model
