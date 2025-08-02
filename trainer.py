import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from models.Prototype import prototype_loss, project_prototypes
import pickle

# Training function
def train_model(model, cfg,  train_loader,  model_path, device):
    criterion = nn.CrossEntropyLoss()
    
    # Phase 1: Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr1, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=3, verbose=True)
    
    for epoch in range(cfg.epochs1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, probs, distances, prototype_activations = model(inputs)

            similarity_map = -distances  # (B, N, K)
            B, N, K = similarity_map.shape
            similarity = similarity_map  

            loss, loss_items = prototype_loss(
                model, outputs, labels, similarity, device, criterion,
                cfg
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        print(f'Epoch {epoch+1}/{cfg.epochs1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc*100:.2f}%')
        scheduler.step(epoch_loss)

    # Phase 2: Unfreeze backbone
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr2, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=3, verbose=True)

    for epoch in range(cfg.epochs2):
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_probs = [] 

        # Prototype update
        if cfg.method.method_name == "Prototype" and epoch > cfg.method.start_proj and epoch % cfg.method.inter_proj == 0:
            model.eval()
            model.proto_meta = project_prototypes(model, train_loader, device)
            print("Prototype projection finished.")

        for inputs, labels, _ in train_loader:
            model.train()
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs, probs, distances, prototype_activations = model(inputs)

            similarity_map = -distances  # (B, N, K)
            B, N, K = similarity_map.shape
            similarity = similarity_map  

            loss, loss_items = prototype_loss(
                model, outputs, labels, similarity, device, criterion,
                cfg
                )
            probs = probs[:, 1] 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        epoch_auc = roc_auc_score(all_labels, all_probs)

        print(f'Epoch {epoch+1}/{cfg.epochs2}, Loss: {epoch_loss:.4f}, ACC: {epoch_acc*100:.2f}%, AUC: {epoch_auc*100:.2f}%')
        scheduler.step(epoch_loss)

    torch.save(model.state_dict(), model_path)
    print("Model saved!")

    if cfg.method.method_name == "Prototype":
        model.proto_meta = project_prototypes(model, train_loader, device)
        with open(f"./proto_meta/{cfg.dataset.dataset_name}_{cfg.method.method_name}.pkl", "wb") as f:
            pickle.dump(model.proto_meta, f)
            print("Final proto_meta saved.")

def test_model(model, cfg, test_loader, model_path, device):
    # Evaluation
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []  
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, probs, distances, prototype_activations = model(inputs)

            similarity_map = -distances  # (B, N, K)
            B, N, K = similarity_map.shape
            similarity = similarity_map  

            loss, loss_items = prototype_loss(
                model, outputs, labels, similarity, device, criterion,
                cfg
                )

            test_loss += loss.item()
            probs = torch.softmax(outputs, dim=1) 
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_probs.extend(probs[:, 1].cpu().numpy()) 
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss /= len(test_loader)
    test_acc = correct / total
    auc = roc_auc_score(all_labels, all_probs)
    precision = precision_score(all_labels, all_preds) 
    recall = recall_score(all_labels, all_preds)        
    f1 = f1_score(all_labels, all_preds)  

    print(f'Test Loss: {test_loss:.4f}, Test ACC: {test_acc*100:.2f}%, Test AUC: {auc*100:.2f}%, Precision: {precision*100:.2f}, Recall: {recall*100:.2f}, F1-score: {f1*100:.2f}')
