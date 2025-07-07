import torch
import copy
import logging
from metrics import get_cindex, get_rm2
import os
import datetime
import pandas as pd


def test(data_loader, model, loss_fn):
    with torch.no_grad():
        model.eval()
        y_true = []
        y_pred = []
        score_list = []
        label_list = []
        smiles_list = []
        fasta_list = []

        running_loss = 0.0
        
        for smiles, graph_data, fasta, label in data_loader:
            
            x = graph_data.x
            edge_index = graph_data.edge_index
            score = model((smiles, x, edge_index), fasta).view(-1)
            loss = loss_fn(score, label)
            running_loss += loss.item()
            y_pred += score.detach().cpu().tolist()
            y_true += label.detach().cpu().tolist()
            score_list.append(score)
            label_list.append(label)
            smiles_list.append(smiles)
            fasta_list.append(fasta)
        with open( "prediction.txt", 'a') as f:
            for i in range(len(score_list)):
                f.write(str(smiles_list[i]) + " " + str(fasta_list[i]) + " " + str(label_list[i]) + " " + str(score_list[i]) +'\n')
        
        ci = get_cindex(y_true, y_pred)
        rm2 = get_rm2(y_true, y_pred)
        model.train()
    return running_loss/len(data_loader), ci, rm2


def train(model, train_loader, val_loader, test_loader, writer, NAME, DATASET, lr=0.0001, epoch=300):

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = 'pth'
    log_dir = 'log'
    test_dir = 'test'

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    val_csv_path = os.path.join(log_dir, f'{DATASET}_training_log_{current_time}.csv')
    test_csv_path = os.path.join(test_dir, f'{DATASET}_test_log_{current_time}.csv')

    training_log = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'val_ci', 'val_rm2'])
    
    opt = torch.optim.Adam(model.parameters(), lr = lr)
    loss_fn = torch.nn.MSELoss()
    model_best = copy.deepcopy(model)
    min_loss = 1000
    test_loss_min = 1000

    resume_path = None
    if resume_path:
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        min_loss = checkpoint['test_loss']
        test_ci = checkpoint['test_ci']
        test_rm2 = checkpoint['test_rm2']  
        print(f"Resuming training from epoch {start_epoch}...")
    else:
        start_epoch = 0
    for start_epoch in range(epoch):
        epo = start_epoch
        print(f"Starting epoch {epo+1}/{epoch} on {DATASET} dataset...")
        model.train()
        running_loss = 0.0
        
        for data in train_loader:                                       
            smiles, graph_data, fasta, label = data
            x = graph_data.x
            edge_index = graph_data.edge_index
            score = model((smiles, x, edge_index), fasta).view(-1)
            loss = loss_fn(score, label)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()

        avg_train_loss = running_loss/len(train_loader)    
        writer.add_scalar(f'Loss/train_{NAME}', avg_train_loss, epo)
        logging.info(f'Training at Epoch {epo + 1} with loss {avg_train_loss:.4f}')
        
        val_loss, val_ci, val_rm2 = test(val_loader, model, loss_fn)
        writer.add_scalar(f'Loss/valid_{NAME}', val_loss, epo)
        logging.info(f'Validation at Epoch {epo+1} with loss {val_loss:.4f}, ci {val_ci}, rm2 {val_rm2}')

        if val_loader is not None:
            val_loss, val_ci, val_rm2 = test(val_loader, model, loss_fn)
            writer.add_scalar(f'Loss/valid_{NAME}', val_loss, epo)
            logging.info(f'Validation at Epoch {epo+1} with loss {val_loss:.4f}, ci {val_ci}, rm2 {val_rm2}')

        test_loss, test_ci, test_rm2 = test(test_loader, model, loss_fn)
        new_test_row = pd.DataFrame({
            'epoch': [epo + 1],
            'test_loss': [test_loss],
            'test_ci': [test_ci],
            'test_rm2': [test_rm2]
        })
        if epo == 0:
            new_test_row.to_csv(test_csv_path, index=False, mode='w')
        else:
            new_test_row.to_csv(test_csv_path, index=False, mode='a', header=False)
        if test_loss < test_loss_min:
            test_loss_min = test_loss
            model_best = copy.deepcopy(model)
            save_path = os.path.join(test_dir, f'{DATASET}_best_test_model_{current_time}.pth')
            torch.save({
                'epoch': epo,
                'model_state_dict': model_best.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'test_loss': test_loss_min,
                'test_ci': test_ci,
                'test_rm2': test_rm2   
            }, save_path)
            
    

