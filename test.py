import torch
import numpy as np
from torch.utils.data import DataLoader
import os
from src.Net import Net
from utils.dataloader import Datases_loader as dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchsz = 1
model = Net().to(device)
savedir = r'Saved_Weight_Path'
imgdir = r'Test_Set_Image_Path'
labdir = r'Test_Set_Label_Path'
imgsz = 512
resultsdir = r'xxxxx'

dataset = dataloader(imgdir, labdir, imgsz, imgsz)
testsets = DataLoader(dataset, batch_size=batchsz, shuffle=False)

def test():
    model.load_state_dict(torch.load(savedir))
    exist = os.path.exists(resultsdir)
    if not exist:
        os.makedirs(resultsdir)

    for idx, samples in enumerate(testsets):
        img, lab = samples['image'], samples['mask']
        img, lab = img.to(device), lab.to(device)

        pred = model(img)

        np.save(resultsdir + r'/img' + str(idx+1) + '.npy', img.detach().cpu().numpy()) # original images
        np.save(resultsdir + r'/pred' + str(idx+1) + '.npy', pred.detach().cpu().numpy()) # predicted results
        np.save(resultsdir + r'/label' + str(idx+1) + '.npy', lab.detach().cpu().numpy()) # labels

if __name__ == '__main__':
    test()
