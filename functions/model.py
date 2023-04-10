import time
import copy
       
        
def train_dev_model(model, dataloaders, criterion, optimizer, metrics, scheduler=None, epochs=5):
    datasloader_sizes= {'train': len(train_dl), 'dev': len(dev_dl)}
    start = time.time()
    
    # Copy state dict to have the alternative of the best accuracy weights
    best_wts = copy.deepcopy(model.state_dict())
    
    # Declare loss & accuracy
    losses = []
    accuracies = []
    
    # Main loop
    for e in range(epoch):
        # Declare loss and accuracy for the current epoch 
        epoch_loss = 0.0
        epoch_acc = 0
        best_acc = 0
        
        # Loop between the train and dev sets 
        for phase in ['train','dev']:
            # Declare loss and accuracy for the current train or dev set
            running_loss = 0.0
            running_acc = 0
            
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            # Phase Dataloader's Loop
            progress = tqdm(enumerate(dataloaders[phase]), desc=f\"Epoch: {e}, R_Loss: {running_loss}, R_Acc: {running_acc}\", total=datasloader_sizes[phase])
            for i, data in progress:
                # Map the images and labels of the current batch 
                Y = data[0].type(torch.LongTensor)
                Y = Y.to(device)
                X = data[1].to(device)

                # Put the optimizer's gradients to zero 
                optimizer.zero_grad()
                
                # Activate the automatic gradient calculation depending on the phase (If train = Activate grad)  
                with torch.set_grad_enabled(phase =='train'):
                    # Predict output from the current images batch
                    output = model(X)
                    _, Yhat = torch.max(output, 1)
                    
                    # Calculate Loss from Y and the prediction
                    loss = criterion(output, Y)
                    
                    # If in training phase, 
                    if phase =='train':
                        loss.backward()
                        optimizer.step()
                
                       
                        
                # Compute Loss and accuracy
                running_loss += loss.item()
                a = accuracy_func(Yhat, Y)
                running_acc += a
 
                r_loss =loss.item()
                r_acc = a
        
                # Updater tqdm
                progress.set_description(f\"Epoch: {e+1}, R_Loss: {r_loss}, R_Acc: {r_acc}\")
                
        if phase == 'train' and scheduler!=None:
                    scheduler.step() 
        # Compute epoch's Loss and accuracy
        epoch_loss = running_loss / datasloader_sizes[phase]
        epoch_acc = running_acc / datasloader_sizes[phase]
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        print(f\"Epoch Loss: {epoch_loss}\")
        print(f\"Epoch Accuracy: {epoch_acc}\")
        
        if epoch_acc > best_acc:
            best_wts_model = copy.deepcopy(model.state_dict())
    # Comopute time and displayo it
    time_elapsed_s = np.round(time.time() - start,2) 
    print(f\"Training completed in: {time_elapsed_s}\")
    
    # Copy state dict to have the alternative of the last accuracy weights
    last_epoch_wts = copy.deepcopy(model.state_dict())
    return model, best_wts, last_epoch_wts, losses, accuracies
          
def train(model, dataloader, criterion, optimizer, metrics, scheduler=None, epoch=5, device, weights=None):
    dl_size = len(dataloader)
    # This function is to train only and quickly (from a specific state_dict or continuing with the current model state_dict)
    if weights is not None:
          model.load(weights)
          
    losses = []
    metrics = []
    for e in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        best_acc = 0.0
        model.train()
        progress = tqdm(enumerate(dataloader), desc="Epoch: {e}, R_Loss: {running_loss}, R_Acc: {running_acc}", total=dl_size)
        for Y, X from progress:
          running_acc = 0.0
          running_loss = 0.0
          
          X = X.to(device)
          Y = Y.to(device)
          
          optimizer.zero_grad()
          
          output= model(X)
          _, Yhat = torch.max(output, 1)
          
          loss = criterion(output, Y)
          loss.backward()
          optimizer.step()
          
          running_loss += loss.item()
          running_acc += metrics(Yhat, Y)
        if scheduler is not None:
          scheduler.step()
        epoch_loss = running_loss / dl_size
        epoch_acc = running_acc / dl_size
        losses.append(epoch_loss)
        metrics.append(epoch_acc)
        print(f\"Epoch Loss: {epoch_loss}\")
        print(f\"Epoch Accuracy: {epoch_acc}\")
              
    return model, losses, metrics
    
    
    
    
    