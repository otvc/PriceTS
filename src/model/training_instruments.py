
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm


def dict_to_input(batch_dict, feature_keys, cat_feature, target = 'target_price', device = 'cpu'):
    feat_iter = iter(feature_keys)
    
    batch = batch_dict[next(feat_iter)]
    batch = torch.cat(list(map(lambda x: x.view(1,x.shape[0], 1),batch)), dim = 0).permute(1, 0, 2)
    
    batch = batch.view(batch.shape[0], batch.shape[1], 1) # (bs, ls, 1)
    
    for key_data in feat_iter:
        temp_batch = batch_dict[key_data]
        temp_batch = torch.cat(list(map(lambda x: x.view(1,x.shape[0], 1), temp_batch)), dim = 0).permute(1, 0, 2)
        batch = torch.cat([batch, temp_batch], dim = -1)
    
    batch_cat = torch.cat(list(map(lambda x: x.view(1,x.shape[0], 1), batch_dict[cat_feature])), dim = 0).permute(1, 0, 2)
    
    target_val = batch_dict[target]
    
    return (batch.to(torch.float32).to(device), batch_cat.to(device)), target_val.to(torch.float32).to(device)

def plot_train_process(train_loss, val_loss, title_suffix='', path = 'artefacts/', name = 'CatEmbLoss'):
    fig, axes = plt.subplots(1, 1, figsize=(15, 5))

    axes.set_title(' '.join(['Loss', title_suffix]))
    axes.plot(train_loss, label='train')
    axes.plot(val_loss, label='validation')
    axes.legend()

    plt.savefig(path + name + '.png')
    

def train_loop(model, train_dataloader, optimizer, criterion, batch_transform, device = 'cpu'):
    losses = []
    for batch in train_dataloader:
        X_batch, y_batch = batch_transform(batch, device = device)
        
        output = model(X_batch)
        loss = criterion(output, y_batch)
        losses.append(loss.detach().cpu())
        loss.backward()
        
        optimizer.step()
    return torch.Tensor(losses).mean()

def val_loop(model, dataloader, criterion, batch_transform, device = 'cpu'):
    losses = []
    for batch in dataloader:
        X_batch, y_batch = batch_transform(batch, device = device)
        output = model(X_batch)
        loss = criterion(output, y_batch)
        losses.append(loss.detach().cpu())
    return torch.Tensor(losses).mean()

def test_loop(model, dataloader, batch_transform, device = 'cpu'):
    y_pred = []
    y_gt = []
    for batch in dataloader:
        X_batch, y_batch = batch_transform(batch, device = device)
        output = model(X_batch)
        y_pred.append(output)
        y_gt.append(y_batch)
    return y_pred, y_gt


def save_stage(model:torch.nn.Module, 
               optim:torch.optim.Optimizer, 
               path_to_save:str, 
               model_name:str, 
               stage_num:int):
    torch.save(model, path_to_save + model_name + '_optimizer' + '_' + str(stage_num) + '.pt')
    torch.save(optim, path_to_save + model_name + '_' + str(stage_num) + '.pt')


'''
`params`:
    `plot_loss`:bool: value regulating plotting losses;
    `every_epoch`: every  `every_epoch` we should calculate metrics on val dataset.
'''
def train(model, 
          train_dataloader, 
          val_dataloader, 
          optimizer, 
          criterion, 
          batch_transform, 
          epochs:int = 100, 
          plot_loss:bool = True, 
          every_epoch:int = 5, 
          device = 'cpu', 
          path_to_stages = '../../models/stages/',
          model_name = 'CatEmbLoss'):
    train_loss_per_epoch = []
    val_loss_per_epoch = []
    for e in tqdm(range(1, epochs+1)):
        loss = train_loop(model, train_dataloader, optimizer, criterion, batch_transform, device = device)
        if e % every_epoch == 0:
            save_stage(model, optimizer, path_to_stages, model_name, e)

            with torch.no_grad():
                val_loss = val_loop(model, val_dataloader, criterion, batch_transform, device = device)

            train_loss_per_epoch.append(loss)
            val_loss_per_epoch.append(val_loss)

        if plot_loss and e % every_epoch == 0:
            plot_train_process(train_loss_per_epoch, val_loss_per_epoch)
            
    output = {}
    output['model'] = model
    output['train_loss'] = train_loss_per_epoch
    output['val_loss'] = val_loss_per_epoch
    return output

def unpack_CatEmbLSTM(batch,
                      numeric_features = ['StoreInventory', 'sales_cost_x', 'sales_value', 'price', 'cost', 'sales_cost_y'],
                      cat_features = ['price_zone_&_class_name'],
                      device = 'cpu'):
    return dict_to_input(batch, numeric_features, cat_features[0], target = 'target_price', device = device)

def train_CatEmbLSTM(model,
                     train_dataloader,
                     val_dataloader,
                     optimizer, criterion, 
                     epochs = 100,
                     every_epoch = 1,
                     plot_loss = True,
                     device = 'cpu',
                     path_to_stages = '../../models/stages/',
                     model_name = 'CatEmbLoss'):
    return train(model, train_dataloader, val_dataloader, optimizer, criterion, unpack_CatEmbLSTM, 
                 epochs = epochs, every_epoch = every_epoch, plot_loss = plot_loss, device = device,
                 path_to_stages = path_to_stages, model_name = model_name)

def inference_CatEmbLSTM(model, dataloader, device = 'cpu'):
    return test_loop(model, dataloader, unpack_CatEmbLSTM, device = device)
