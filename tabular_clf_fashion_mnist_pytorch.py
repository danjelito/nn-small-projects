import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

RANDOM_STATE= 8
torch.manual_seed(RANDOM_STATE)
plt.rcParams["font.family"] = "serif"

# load df
path_train= 'dataset/fashion-mnist_train.csv'
path_test= 'dataset/fashion-mnist_test.csv'
df_train= pd.read_csv(path_train)
df_test= pd.read_csv(path_test)

# train val test split
X_train= df_train.drop(columns= 'label')
y_train= df_train.loc[:, 'label'].values
X_test= df_test.drop(columns= 'label')
y_test= df_test.loc[:, 'label'].values

X_train, X_val, y_train, y_val= train_test_split(
    X_train, 
    y_train, 
    test_size= 0.25, random_state= RANDOM_STATE
)

# preprocessing
scaler= StandardScaler()
scaler.fit(X_train)
X_train= scaler.transform(X_train)
X_val= scaler.transform(X_val)
X_test= scaler.transform(X_test)

# create dataset
X_train_tensor= torch.Tensor(X_train).float()
X_val_tensor= torch.Tensor(X_val).float()
X_test_tensor= torch.Tensor(X_test).float()

y_train_tensor= torch.Tensor(y_train).long()
y_val_tensor= torch.Tensor(y_val).long()
y_test_tensor= torch.Tensor(y_test).long()

train_ds= TensorDataset(X_train_tensor, y_train_tensor)
train_dl= DataLoader(train_ds, batch_size= 64, shuffle= True)

# create model
class MyModel(nn.Module):

    def __init__(self, n_input, n_neuron, n_output):
        super().__init__()
        l1= nn.Linear(n_input, n_neuron)
        a1= nn.ReLU()
        l2= nn.Linear(n_neuron, n_output)
        a2= nn.ReLU()
        l3= nn.Linear(n_neuron, n_output)
        a3= nn.Softmax(dim= 1)
        self.module_list= nn.ModuleList([l1, a1, l2, a2, a3])

    def forward(self, X):
        for f in self.module_list:
            X= f(X)
        return X
    
n_input= X_train_tensor.shape[1]
n_neuron= 32
n_output= len(np.unique(y_train_tensor))
learning_rate= 0.001
n_train_samples= X_train_tensor.shape[0]

# model with shape n_input - 32 - 32 - n_output
model= MyModel(n_input, n_neuron, n_output)
loss_fn= nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(model.parameters(), lr= learning_rate)

# training and validation
n_epochs= 10
log_print = 2 # print acc and loss every this epoch

acc_hist_train= [0] * n_epochs
acc_hist_val= [0] * n_epochs

loss_hist_train= [0] * n_epochs
loss_hist_val= [0] * n_epochs

for epoch in range(n_epochs):
    
    # training
    for X_batch, y_batch in train_dl:
        
        y_pred_proba= model(X_batch)
        y_pred= torch.argmax(y_pred_proba, dim= 1)

        loss= loss_fn(y_pred_proba, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        is_correct= (y_pred == y_batch).float()
        n_correct= is_correct.sum()

        acc_hist_train[epoch] += n_correct.item()
        loss_hist_train[epoch] += loss.detach().item()
    
    acc_hist_train[epoch] /= n_train_samples
    loss_hist_train[epoch] /= n_train_samples

    # validation
    y_pred_proba= model(X_val_tensor)
    y_pred= torch.argmax(y_pred_proba, axis= 1)

    loss= loss_fn(y_pred_proba, y_val_tensor)
    is_correct= (y_pred == y_val_tensor).float()
    acc= is_correct.mean()

    acc_hist_val[epoch] += acc
    loss_hist_val[epoch] += loss.detach().item()

    if epoch % log_print == 0:
    
        print(f'epoch {epoch}\n'
              f'train acc: {acc_hist_train[epoch]: .3f}, val acc: {acc_hist_val[epoch]: .3f}\n'
              f'train loss: {loss_hist_train[epoch]: .3f}, val loss: {loss_hist_val[epoch]: .3f}\n')
        
# test
y_pred_proba= model(X_test_tensor)
y_pred= torch.argmax(y_pred_proba, axis= 1)
acc= (y_test_tensor == y_pred).float().mean()

print(f'test\ntest acc: {acc: .3f}')

# visualize prediction
last_10= X_test_tensor[-10:].numpy()
reshaped_10 = last_10.reshape(-1, 28, 28) # reshape to 28*28

y_true= y_test_tensor[-10:].numpy()
y_pred= y_pred[-10:].numpy()
labels= {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot',
}
y_true_label= [labels[y] for y in y_true]
y_pred_label= [labels[y] for y in y_pred]

# plot
fig, axs = plt.subplots(2, 5, figsize= (12, 8))

for i, ax in enumerate(axs.flatten()):
    
    ax.imshow(reshaped_10[i], cmap='gray_r')
    
    # set title, color it green if true else red
    title= f'True label: {y_true_label[i]}\nPredicted label: {y_pred_label[i]}'
    if y_true_label[i] == y_pred_label[i]:
        ax.set_title(title, fontweight= 'bold', color= 'green')
    else:
        ax.set_title(title, fontweight= 'bold', color= 'red')
    
    ax.axis('off')

plt.suptitle('Categorizing Fashion Items with Neural Network', 
             fontsize= 30, fontweight= 'bold')
plt.subplots_adjust(hspace= 0.075)
plt.show()