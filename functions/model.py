import time
import copy
import torch
import numpy as np
from torch.optim import *
from tqdm import tqdm
        
def train_dev_model(model, dataloaders, criterion, optimizer, device, metrics, scheduler=None, epochs=5):
    dataloader_sizes= {'train': len(dataloaders['train']), 'dev': len(dataloaders['dev'])}
    start = time.time()
    
    # Copy state dict to have the alternative of the best accuracy weights
    best_wts = copy.deepcopy(model.state_dict())
    
    # Declare loss & accuracy
    losses = []
    accuracies = []
    
    # Main loop
    for e in range(epochs):
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
            progress = tqdm(enumerate(dataloaders[phase]), desc=f"Epoch: {e}", total=dataloader_sizes[phase])
            for i, (X,Y) in progress:
                # Map the images and labels of the current batch 
                X = X.to(device)
                Y = Y.type(torch.LongTensor)
                Y = Y.to(device)
                
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
                a = metrics(Yhat, Y)
                running_acc += a
 
                r_loss =loss.item()
                r_acc = a
        
                # Updater tqdm
                progress.set_description(f"Epoch: {e+1}")
                
        if phase == 'train' and scheduler!=None:
                    scheduler.step() 
        # Compute epoch's Loss and accuracy
        epoch_loss = running_loss / dataloader_sizes[phase]
        epoch_acc = running_acc / dataloader_sizes[phase]
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        print(f"Epoch Loss: {epoch_loss}")
        print(f"Epoch Accuracy: {epoch_acc}")
        
        if epoch_acc > best_acc:
            best_wts_model = copy.deepcopy(model.state_dict())
    # Comopute time and displayo it
    time_elapsed_s = np.round(time.time() - start,2) 
    print(f"Training completed in: {time_elapsed_s}")
    
    # Copy state dict to have the alternative of the last accuracy weights
    last_epoch_wts = copy.deepcopy(model.state_dict())
    return model, best_wts, last_epoch_wts, losses, accuracies
          
def train(model, dataloader, criterion, optimizer, metrics, device, scheduler=None, epoch=5, weights=None):
    dl_size = len(dataloader)
    start = time.time()
    # This function is to train only and quickly (from a specific state_dict or continuing with the current model state_dict)
    if weights is not None:
          model.load(weights)
          
    losses = []
    for e in range(epochs):
        epoch_loss = 0.0
        model.train()
        progress = tqdm(enumerate(dataloader), desc=f"Epoch: {e}, R_Loss: {running_loss}", total=dl_size)
        for Y, X in progress:
            X = X.to(device)
            Y = Y.to(device)
          
            optimizer.zero_grad()

            output= model(X)
            _, Yhat = torch.max(output, 1)

            loss = criterion(output, Y)
            loss.backward()
            optimizer.step()
          
            epoch_loss += loss.item()
            progress.set_description(f"Epoch: {e+1}, R_Loss: {r_loss}")
        if scheduler is not None:
            scheduler.step()
        epoch_loss = epoch_loss / dl_size
        losses.append(epoch_loss)
        print(f"Training Epoch Loss: {epoch_loss}")
        time_elapsed_s = np.round(time.time() - start,2) 
        print(f"Training completed in: {time_elapsed_s}")   
    return model, losses
    
    
def evaluation(model, metrics, device, weights=None):
    dl_size = len(dataloader)
    start = time.time()          
    if weights is not None:
          model.load(weights)
          
    losses = []
    accuracies = []
    for e in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        best_acc = 0.0
        model.eval()
        progress = tqdm(enumerate(dataloader), desc="Epoch: {e}, R_Loss: {running_loss}, R_Acc: {running_acc}", total=dl_size)
        for Y, X in progress:
          running_acc = 0.0
          running_loss = 0.0
          with torch.no_grad():
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
              progress.set_description(f"Epoch: {e+1}, R_Loss: {r_loss}, R_Acc: {r_acc}")

        epoch_loss = running_loss / dl_size
        epoch_acc = running_acc / dl_size
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        print(f"Evaluation Epoch Loss: {epoch_loss}")
        print(f"Evaluation Epoch Accuracy: {epoch_acc}")
        time_elapsed_s = np.round(time.time() - start,2) 
        print(f"Evaluation completed in: {time_elapsed_s}")      
    return model, losses, accuracies
    
def test(model, metrics, device, weights=None):
    dl_size = len(dataloader)
    start = time.time()          
    if weights is not None:
          model.load(weights)
          
    losses = []
    accuracies = []
    for e in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        best_acc = 0.0
        progress = tqdm(enumerate(dataloader), desc="Epoch: {e}, R_Loss: {running_loss}, R_Acc: {running_acc}", total=dl_size)
        for Y, X in progress:
          running_acc = 0.0
          running_loss = 0.0
          with torch.no_grad():
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
              progress.set_description(f"Epoch: {e+1}, R_Loss: {r_loss}, R_Acc: {r_acc}")

        epoch_loss = running_loss / dl_size
        epoch_acc = running_acc / dl_size
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        print(f"Test Epoch Loss: {epoch_loss}")
        print(f"Test Epoch Accuracy: {epoch_acc}")
        time_elapsed_s = np.round(time.time() - start,2) 
        print(f"Test completed in: {time_elapsed_s}")      
    return model, losses, accuracies

    
    