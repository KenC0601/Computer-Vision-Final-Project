import torch

def get_classifier(model):
    """Recursively find the classifier layer."""
    if hasattr(model, "classifier"):
        return model.classifier
    if hasattr(model, "base_model"):
        return get_classifier(model.base_model)
    # Fallback for wrappers that might use .model
    if hasattr(model, "model"):
        return get_classifier(model.model)
    raise AttributeError("Classifier not found in model hierarchy")

def get_encoder(model):
    """Recursively find the object that has encode_image."""
    if hasattr(model, "encode_image"):
        return model
    if hasattr(model, "base_model"):
        return get_encoder(model.base_model)
    if hasattr(model, "model"):
        return get_encoder(model.model)
    raise AttributeError("Encoder (encode_image) not found in model hierarchy")

def precompute_features(model, loader, device):
    encoder = get_encoder(model)
    features_list = []
    labels_list = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            # Encode
            features = encoder.encode_image(images)
            # Normalize (important for CLIP/BioCLIP)
            features = features / features.norm(dim=-1, keepdim=True)
            
            features_list.append(features)
            labels_list.append(labels.to(device))
            
    return torch.cat(features_list), torch.cat(labels_list)

def train_linear_probe(model, loader, criterion, optimizer, device, epochs=50):
    train_features, train_labels = precompute_features(model, loader, device)
    
    classifier = get_classifier(model)
    classifier.train()
    
    batch_size = loader.batch_size
    num_samples = train_features.size(0)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        # Shuffle indices
        indices = torch.randperm(num_samples, device=device)
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_idx = indices[start_idx:end_idx]
            
            batch_features = train_features[batch_idx]
            batch_labels = train_labels[batch_idx]
            
            optimizer.zero_grad()
            outputs = classifier(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total
