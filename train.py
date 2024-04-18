import torch

def train_loop(num_epochs, device, model, traning_data, batch_size, learning_rate):
    # Initialize your model, optimizer, and other necessary components
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    max_score = -1
    best_epoch = 0
    best_score = None
    
    for epoch in range(num_epochs):
        
        data = data.to(device)
        
        model.train()
        
        model.zero_grad()
        
        pred = model(data)
        
        loss = criterion(pred, data.y)
        
        loss.backward()
        
        optimizer.step()
        
        for batch in batch_size:
            
            
        # Evaluate your model on validation data (optional)