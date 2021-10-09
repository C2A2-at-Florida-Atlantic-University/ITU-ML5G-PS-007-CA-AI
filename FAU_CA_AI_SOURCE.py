import h5py 
import numpy as np 
import random 
import matplotlib 
matplotlib.use('Agg')  
import matplotlib.pyplot as plt 
import time 
import sys 
from math import ceil 
from sklearn.decomposition import PCA 
from torch.utils.data import Dataset, DataLoader 
import torch 
from torch import nn 
from sklearn.metrics import accuracy_score 
from tqdm.notebook import tqdm as tqdm 
from brevitas.export.onnx.generic.manager import BrevitasONNXManager 
from finn.util.inference_cost import inference_cost 
import json 
import netron 
import tensorly as tl
import os 
from IPython.display import IFrame
 
dataset_path = "./GOLD_XYZ_OSC.0001_1024.hdf5" 
 
gpu = 0 
if torch.cuda.is_available(): 
    torch.cuda.device(gpu) 
    print("Using GPU %d" % gpu) 
else: 
    gpu = None 
    print("Using CPU only") 
 
class radioml_18_dataset(Dataset): 
    def __init__(self, dataset_path): 
        super(radioml_18_dataset, self).__init__() 
        h5_file = h5py.File(dataset_path,'r') 
        self.data = h5_file['X'] 
        self.mod = np.argmax(h5_file['Y'], axis=1) # comes in one-hot encoding 
        self.snr = h5_file['Z'][:,0] 
        self.len = self.data.shape[0] 
 
        self.mod_classes= ['OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK',
                '16APSK','32APSK','64APSK','128APSK','16QAM','32QAM','64QAM','128QAM','256QAM',
                        'AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM','GMSK','OQPSK']
 
        self.snr_classes = np.arange(-20., 32., 2) # -20dB to 30dB 
 
        # do not touch this seed to ensure the prescribed train/test split! 
        np.random.seed(2018) 
        train_indices = [] 
        test_indices = [] 
        for mod in range(0, 24): # all modulations (0 to 23) 
            for snr_idx in range(0, 26): # all SNRs (0 to 25 = -20dB to +30dB) 
                # 'X' holds frames strictly ordered by modulation and SNR 
                start_idx = 26*4096*mod + 4096*snr_idx 
                indices_subclass = list(range(start_idx, start_idx+4096)) 
                 
                # 90%/10% training/test split, applied evenly for each mod-SNR p
                split = int(np.ceil(0.1 * 4096))  
                np.random.shuffle(indices_subclass)

                train_indices_subclass = indices_subclass[split:] 
                test_indices_subclass = indices_subclass[:split]
                 
                # you could train on a subset of the data, e.g. based on the SNR
                # here we use all available training samples 
                if snr_idx >= 0: 
                    train_indices.extend(train_indices_subclass) 
                test_indices.extend(test_indices_subclass) 

        torch.manual_seed(2016) #this seed has provided best performance
        self.test_indices = test_indices         
        self.train_indices = train_indices 
        self.train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        self.test_sampler = torch.utils.data.SubsetRandomSampler(test_indices) 
 
    def __getitem__(self, idx): 
        # transpose frame into Pytorch channels-first format (NCL = -1,2,1024) 
        return self.data[idx].transpose(), self.mod[idx], self.snr[idx] 
 
    def __len__(self): 
        return self.len 
 
dataset = radioml_18_dataset(dataset_path) 
 
def init_weights(m): 
    if isinstance(m, nn.Linear): 
        torch.nn.init.xavier_uniform_(m.weight) 
        m.bias.data.fill_(0.01) 

#Create original baseline network
dr = 0.5 
model = nn.Sequential( 
    nn.Conv1d(2, 64, kernel_size=8), 
    nn.ReLU(), 
    nn.MaxPool1d(2, stride=2), 
    nn.Conv1d(64, 64, kernel_size=8), 
    nn.ReLU(), 
    nn.MaxPool1d(2, stride=2), 
    nn.Conv1d(64, 64, kernel_size=8), 
    nn.ReLU(), 
    nn.MaxPool1d(2, stride=2), 
    nn.Conv1d(64, 64, kernel_size=8), 
    nn.ReLU(), 
    nn.MaxPool1d(2, stride=2), 
    nn.Conv1d(64, 64, kernel_size=8), 
    nn.ReLU(), 
    nn.MaxPool1d(2, stride=2), 
    nn.Conv1d(64, 64, kernel_size=8), 
    nn.ReLU(), 
    nn.MaxPool1d(2, stride=2), 
    nn.Conv1d(64, 64, kernel_size=8), 
    nn.ReLU(), 
    nn.MaxPool1d(2,2), 
    nn.Flatten(), 
    nn.Linear(64, 128), 
    nn.SELU(), 
    nn.Dropout(dr), 
    nn.Linear(128, 128), 
    nn.SELU(), 
    nn.Dropout(dr), 
    nn.Linear(128, 24), 
)
model.apply(init_weights) 
print(model) 
 
def train(model, train_loader, optimizer, criterion): 
    losses = [] 
    # ensure model is in training mode 
    model.train()     
 
    for (inputs, target, snr) in tqdm(train_loader, desc="Batches", leave=False):
        if gpu is not None: 
            inputs = inputs.cuda() 
            target = target.cuda() 
        #print(inputs.shape)         
        # forward pass 
        output = model(inputs) 
        loss = criterion(output, target) 
        #print(output.shape) 
        # backward pass + run optimizer to update weights 
        optimizer.zero_grad()  
        loss.backward() 
        optimizer.step() 
         
        # keep track of loss value 
        losses.append(loss.cpu().detach().numpy()) 
            
    return losses 
 
def test(model, test_loader):     
    # ensure model is in eval mode 
    model.eval()  
    y_true = [] 
    y_pred = [] 
    with torch.no_grad(): 
        for (inputs, target, snr) in test_loader: 
            if gpu is not None: 
                inputs = inputs.cuda() 
                target = target.cuda() 
            output = model(inputs) 
            #print(output.shape)
            pred = output.argmax(dim=1, keepdim=True)
            #print(pred.shape)
            y_true.extend(target.tolist())  
            y_pred.extend(pred.reshape(-1).tolist())
            #print(len(y_true), len(y_pred))
    print(y_true[:100], y_pred[:100])
    return accuracy_score(y_true, y_pred) 
 
batch_size = 500 
num_epochs = 20

# uncomment to train original baseline network
"""
data_loader_train = DataLoader(dataset, batch_size=batch_size, sampler=dataset.train_sampler)
data_loader_test = DataLoader(dataset, batch_size=batch_size, sampler=dataset.test_sampler)
 
if gpu is not None: 
    model = model.cuda() 
 
# loss criterion and optimizer 
criterion = nn.CrossEntropyLoss() 
if gpu is not None: 
    criterion = criterion.cuda() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.999)) 
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
 
running_loss = [] 
running_test_acc = [] 
start = 0 
end = 0 
 
for epoch in tqdm(range(num_epochs), desc="Epochs"): 
        start = time.time() 
        loss_epoch = train(model, data_loader_train, optimizer, criterion) 
        test_acc = test(model, data_loader_test) 
        print("Epoch %d: Training loss = %f, test accuracy = %f" % (epoch, np.mean(loss_epoch), test_acc))
        running_loss.append(loss_epoch) 
        running_test_acc.append(test_acc) 
        lr_scheduler.step() 
        end = time.time() 
        print("Epoch training time: ", end - start) 
 
        torch.save(model.state_dict(), "FAU_CA_AI_baseline.pth") 
"""

# Load original baseline trained parameters 
savefile = "FAU_CA_AI_baseline.pth" 
saved_state = torch.load(savefile, map_location=torch.device("cpu")) 
model.load_state_dict(saved_state) 
if gpu is not None: 
    model = model.cuda() 

# uncomment to get test accuracy of original baseline network
"""
# Set up a fresh test data loader 
dataset = radioml_18_dataset(dataset_path) 
data_loader_test = DataLoader(dataset, batch_size=batch_size, sampler=dataset.test_sampler)
 
# Run inference on validation data 
y_exp = np.empty((0)) 
y_snr = np.empty((0)) 
y_pred = np.empty((0,len(dataset.mod_classes))) 
model.eval() 
with torch.no_grad(): 
    for data in tqdm(data_loader_test, desc="Batches"): 
        inputs, target, snr = data 
        if gpu is not None: 
            inputs = inputs.cuda() 
        output = model(inputs) 
        y_pred = np.concatenate((y_pred,output.cpu())) 
        y_exp = np.concatenate((y_exp,target)) 
        y_snr = np.concatenate((y_snr,snr)) 
 
conf = np.zeros([len(dataset.mod_classes),len(dataset.mod_classes)]) 
confnorm = np.zeros([len(dataset.mod_classes),len(dataset.mod_classes)]) 
for i in range(len(y_exp)): 
    j = int(y_exp[i]) 
    k = int(np.argmax(y_pred[i,:])) 
    conf[j,k] = conf[j,k] + 1 
    for i in range(0,len(dataset.mod_classes)): 
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:]) 
        cor = np.sum(np.diag(conf)) 
ncor = np.sum(conf) - cor 
print("Overall Accuracy across all SNRs of the original CNN: %f"%(cor / (cor+ncor)))

acc = [] 
for snr in dataset.snr_classes: 
    # extract classes @ SNR 
    indices_snr = (y_snr == snr).nonzero() 
    y_exp_i = y_exp[indices_snr] 
    y_pred_i = y_pred[indices_snr] 
    conf = np.zeros([len(dataset.mod_classes),len(dataset.mod_classes)]) 
    confnorm = np.zeros([len(dataset.mod_classes),len(dataset.mod_classes)])
    for i in range(len(y_exp_i)): 
        j = int(y_exp_i[i]) 
        k = int(np.argmax(y_pred_i[i,:])) 
        conf[j,k] = conf[j,k] + 1 
    for i in range(0,len(dataset.mod_classes)): 
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
     
    cor = np.sum(np.diag(conf)) 
    ncor = np.sum(conf) - cor 
    acc.append(cor/(cor+ncor)) 
index = np.argmax(acc) 
print("Highest accuracy for snr ", dataset.snr_classes[index], acc[index]) 
print("Accuracy @ highest SNR (+30 dB): %f"%(acc[-1])) 
print("Accuracy overall: %f"%(np.mean(acc))) 
"""

#start compression algorithm 

def give_p(d):

        #sum of all eigen values
        sum = np.sum(d)
        sum_999 = 0.9998 * sum
        temp = 0
        p = 0
        while temp < sum_999:
            temp += d[p]
            p += 1
        return p

def explained_variance(d):
     # Get variance explained by singular values
    explained_variance_ = (d ** 2) / (len(d) - 1)
    total_var = explained_variance_.sum()
    explained_variance_ratio_ = explained_variance_ / total_var

    return explained_variance_ratio_

N = 200
s_list = []
rank =64
num_modes=2
print("staring to compute mpca...") 
print(*list(model.children()))

def compute_modek_total_scatter(mode, factors):
        scatter = 0

        for m in range(X.shape[0]):
            proj_but_k = tl.unfold(tl.tenalg.multi_mode_dot(X[m], factors, transpose=True, skip=mode), mode)
            #print(proj_but_k.shape)
            scatter += tl.dot(proj_but_k, proj_but_k.T)

        return scatter


data_loader_train = DataLoader(dataset, batch_size=batch_size*4, sampler=dataset.train_sampler)
data_train = next(iter(data_loader_train))[0] 
print("shape train data", len(data_train), len(data_train[0]))

plot_matrix = [[0 for x in range(65)] for y in range(7)]
for layer in range(1, 20, 3):  
    out_model = nn.Sequential(*list(model.children())[:layer]) 
    act_final = [] 
    inputs = data_train[:N] 
    if gpu is not None: 
        inputs = inputs.cuda() 
    act_output = out_model(inputs) 
    act_output = act_output.permute(0, 2, 1) 
    print("printing output shape of N samples of ", layer, "layer: ", act_output.shape)
    num_batches = ceil(100*rank / (act_output.shape[1]*N)) 
    print("num batches for layer", layer, "is : ", num_batches) 
    inputs = data_train[:(num_batches*N)] 
    
    if gpu is not None: 
        inputs = inputs.cuda() 

    act_output = out_model(inputs) 
    act_output = act_output.permute(0, 2, 1) 
    print("shape of act space: ", act_output.shape, type(act_output)) 
    act_output = act_output.cpu().detach().numpy() 
    print("shape of act output in numpy:", act_output.shape) 

    mean_tensor = tl.mean(act_output, axis=0) #the mean of the input training samples TX
    act_output = act_output - mean_tensor
    print("act output shape after centering", act_output.shape)
    X = act_output
    factors = [tl.ones((dim, rank)) for i, dim in enumerate(list(act_output.shape)[1:])]
    for k in range(num_modes):
        scatter = compute_modek_total_scatter(k, factors)
        print("shape scatter", scatter.shape)
        U, S, _ = tl.partial_svd(scatter, n_eigenvecs = rank)
        factors[k] = U
        print("eigenvalues shape for layer", layer, S.shape)
        el = give_p(S)
        print(el)
        if (k==1):
            s_list.append(el)

        var = explained_variance(S)
        print(var)
        y_layer = np.cumsum(var)
        print(y_layer)
        print(y_layer.shape, type(y_layer), y_layer[0].shape)
        #y_layer = np.insert(y_layer, 0, 0, axis = 0)
        #plot_matrix[layer] = y_layer
    
print(s_list) 

filters = [s_list[0]]
for i in range(1, len(s_list)):
    last_elem = filters[-1]
    if (s_list[i] > last_elem):
        filters.append(s_list[i])

print(filters)

#uncomment below for saved filters
#filters = [11, 23, 32, 36, 39, 43] 

#Building the network with a variable number of convolutional layers every time
num_layers = len(filters)

k= [508,250,121,57, 25, 9, 1]
mult = k[num_layers-1]
class CNNModuleVar(nn.Module):
    def __init__(self, input_dim, output_dim, filters_array = []):
        super().__init__()
        self.layers = []
        self.cnn_layers = len(filters_array)
        for l in range(self.cnn_layers):
            self.layers.append(nn.Conv1d(input_dim, filters_array[l], kernel_size=8))
            input_dim = filters_array[l]
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool1d(2, stride=2))
        self.layers = nn.ModuleList(self.layers)
        self.flatten=nn.Flatten()
        self.fc1 = nn.Linear(input_dim*mult, 128)
        self.selu1 = nn.SELU()
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 128)
        self.selu2 = nn.SELU()
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        out = x
        for i in range(self.cnn_layers*3):
            out = self.layers[i](out)
            #print(out.shape)
        out = self.flatten(out)
        #print(out.shape)
        out = self.fc1(out)
        out = self.selu1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.selu2(out)
        out = self.drop2(out)
        #print(out.shape)
        out = self.fc3(out)
        return out

# model2 is optimized network
model2 = CNNModuleVar(2, 24, filters)
model2.apply(init_weights)
print(model2)

data_loader_train = DataLoader(dataset, batch_size=batch_size, sampler=dataset.train_sampler)
data_loader_test = DataLoader(dataset, batch_size=batch_size, sampler=dataset.test_sampler)

if gpu is not None:
    model2 = model2.cuda()
    
# uncomment to train optimized network
"""
# loss criterion and optimizer
criterion = nn.CrossEntropyLoss()
if gpu is not None:
    criterion = criterion.cuda()

optimizer = torch.optim.Adam(model2.parameters(), lr=0.001, betas=(0.9,0.999))
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5, T_mult=1)

running_loss = []
running_test_acc = []
start = 0
end = 0

for epoch in tqdm(range(num_epochs), desc="Epochs"):
    start = time.time()
    loss_epoch = train(model2, data_loader_train, optimizer, criterion)
    test_acc = test(model2, data_loader_test)
    print("Epoch %d: Training loss = %f, test accuracy = %f" % (epoch, np.mean(loss_epoch), test_acc))
    running_loss.append(loss_epoch)
    running_test_acc.append(test_acc)
    lr_scheduler.step()
    end = time.time()
    print(end - start)

torch.save(model2.state_dict(), "FAU_CA_AI_final_model.pth")

batch_size=500
savefile = "FAU_CA_AI_final_model.pth"
saved_state = torch.load(savefile, map_location=torch.device("cpu"))
model2.load_state_dict(saved_state)
if gpu is not None:
    model2 = model2.cuda()

"""

# uncomment to get test accuracy of optimized network
"""
dataset = radioml_18_dataset(dataset_path)
data_loader_test = DataLoader(dataset, batch_size=batch_size, sampler=dataset.test_sampler)

y_exp = np.empty((0))
y_snr = np.empty((0))
y_pred = np.empty((0,len(dataset.mod_classes)))
model2.eval()
with torch.no_grad():
    for data in tqdm(data_loader_test, desc="Batches"):
        inputs, target, snr = data
        if gpu is not None:
            inputs = inputs.cuda()
        output = model2(inputs)
        y_pred = np.concatenate((y_pred,output.cpu()))
        y_exp = np.concatenate((y_exp,target))
        y_snr = np.concatenate((y_snr,snr))
conf = np.zeros([len(dataset.mod_classes),len(dataset.mod_classes)])
confnorm = np.zeros([len(dataset.mod_classes),len(dataset.mod_classes)])
for i in range(len(y_exp)):
    j = int(y_exp[i])
    k = int(np.argmax(y_pred[i,:]))
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(dataset.mod_classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])

cor = np.sum(np.diag(conf))
ncor = np.sum(conf) - cor
print("Overall Accuracy across all SNRs of the optimized network: %f"%(cor / (cor+ncor)))
"""

#Computing inference cost for the original baseline and the optimized network

#Original baseline network
export_onnx_path = "FAU_CA_AI_baseline_export.onnx"
final_onnx_path = "FAU_CA_AI_baseline_final.onnx"
cost_dict_path = "FAU_CA_AI_baseline_cost.json"

BrevitasONNXManager.export(model.cpu(), input_t=torch.randn(1, 2, 1024), export_path=export_onnx_path);
inference_cost(export_onnx_path, output_json=cost_dict_path, output_onnx=final_onnx_path, preprocess=True, discount_sparsity=True)

def showInNetron(model_filename):
    localhost_url = os.getenv("LOCALHOST_URL")
    netron_port = os.getenv("NETRON_PORT")
    netron.start(model_filename, address=("0.0.0.0", int(netron_port)))
    return IFrame(src="http://%s:%s/" % (localhost_url, netron_port), width="100%", height=400)

showInNetron(final_onnx_path)

with open(cost_dict_path, 'r') as f:
    inference_cost_dict = json.load(f)

bops = int(inference_cost_dict["total_bops"])
w_bits = int(inference_cost_dict["total_mem_w_bits"])

bops_baseline = 807699904
w_bits_baseline = 1244936

score_original_model = 0.5*(bops/bops_baseline) + 0.5*(w_bits/w_bits_baseline)
print("Normalized inference cost score for the original network: %f" % score_original_model)

# Optimized network

export_onnx_path = "FAU_CA_AI_final_model_export.onnx" 
final_onnx_path = "FAU_CA_AI_final_model_final.onnx" 
cost_dict_path = "FAU_CA_AI_final_model_cost.json"

BrevitasONNXManager.export(model2.cpu(), input_t=torch.randn(1, 2, 1024), export_path=export_onnx_path)
inference_cost(export_onnx_path, output_json=cost_dict_path, output_onnx=final_onnx_path)
 
showInNetron(final_onnx_path) 
 
with open(cost_dict_path, 'r') as f: 
    inference_cost_dict = json.load(f)

bops = int(inference_cost_dict["total_bops"]) 
w_bits = int(inference_cost_dict["total_mem_w_bits"]) 
 
bops_baseline = 807699904 
w_bits_baseline = 1244936

score_optimized_model = 0.5*(bops/bops_baseline) + 0.5*(w_bits/w_bits_baseline) 
print("Normalized inference cost score for the optimized network: %f" % score_optimized_model)
