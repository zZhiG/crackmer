import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import DataLoader
from src.Net import Net
from utils.dataloader import Datases_loader as dataloader
from utils.loss import Loss


def compute_confusion_matrix(precited,expected):
    # part = precited ^ expected
    part = np.logical_xor(precited, expected)
    pcount = np.bincount(part)
    # tp_list = list(precited & expected)
    # fp_list = list(precited & ~expected)
    tp_list = list(np.logical_and(precited, expected))
    fp_list = list(np.logical_and(precited, np.logical_not(expected)))
    tp = tp_list.count(1)
    fp = fp_list.count(1)                  
    tn = pcount[0] - tp
    fn = pcount[1] - fp
    return tp, fp, tn, fn


def compute_indexes(tp, fp, tn, fn):
    accuracy = (tp+tn) / (tp+tn+fp+fn)
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    F1 = (2*precision*recall) / (precision+recall)
    return accuracy, precision, recall, F1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batchsz = 1
lr = 1
items = 1

model = Net().to(device)

criterion = Loss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=1, T_mult=1)

savedir = r'save_path_xxxx'

imgpath = r'xxxxx'
labpath = r'xxxxx'

imgsz = 512

dataset = dataloader(imgpath, labpath, imgsz, imgsz)
trainsets = DataLoader(dataset, batch_size=batchsz, shuffle=True)

lossx = 0
tp, tn, fp, fn = 0, 0, 0, 0
accuracy, precision, recall, F1, ls_loss = [],[],[],[],[]

def train():
    for epoch in range(items):
        lossx = 0
        tp, tn, fp, fn = 0, 0, 0, 0
        for idx, samples in enumerate(trainsets):
            img, lab = samples['image'], samples['mask']
            img, lab = img.to(device), lab.to(device)

            optimizer.zero_grad()
            pred = model(img)

            loss = criterion(pred, lab)
            loss.backward()
            optimizer.step()

            lossx = lossx + loss

            p = pred.reshape(-1)
            p[p >= 0.] = 1
            p[p < 0.] = 0
            t = lab.reshape(-1)
            tp_, fp_, tn_, fn_ = compute_confusion_matrix(p.detach().cpu().numpy(), t.detach().cpu().numpy())
            tp = tp + tp_
            fp = fp + fp_
            tn = tn + tn_
            fn = fn + fn_

        accuracy_, precision_, recall_, F1_ = compute_indexes(tp, fp, tn, fn)
        accuracy.append(accuracy_)
        precision.append(precision_)
        recall.append(recall_)
        F1.append(F1_)

        scheduler.step()
        lossx = lossx / dataset.num_of_samples()
        ls_loss.append(lossx.item())

    torch.save(model.state_dict(), savedir)

if __name__ == '__main__':
    train()
    str = 'accuracy:' + str(accuracy) + '\nprecision:' + str(precision) + '\nrecall:' + str(recall) + '\nF1:' + str(F1) + '\nloss:' + str(ls_loss)
    filename = r'xxxx'
    with open(filename, mode='w', newline='') as f:
        f.writelines(str)