# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:28:13 2023

@author: student
"""
from collections import OrderedDict 
import torch
import torch.nn as nn #Redes neuronales
import torch.optim as optim #Optimizadores
import matplotlib.pyplot as plt #Visualizacion
import torchvision.transforms as transforms # Normalizacion



class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        
        
        self.layers = nn.Sequential( OrderedDict([ 
                     ('hidden_linear', nn.Linear(input_size,hidden_size)),
                     ('hidden_activation', nn.Tanh()),#Tanh or ReLU
                     ('output_linear', nn.Linear(hidden_size,output_size)),
                     ('output_activation', nn.Identity()),
                     ]))
        

    def forward(self, inputs):
        out = self.layers(inputs)
        return out

class Training(nn.Module):
    
     def __init__(self, Model, n_epochs=1000,learning_rate=0.001, optimizer='RMSprop', loss='MSE', weight_decay=0, patient=5):
         super(Training, self).__init__()
         self.criterion = nn.MSELoss() if loss=='MSE' else nn.L1Loss()#MAE fc_loss
         self.n_epochs=n_epochs
         self.model=Model
         self.model_dtype = next(Model.parameters()).dtype
         self.patient=patient  # Número máximo de épocas sin mejora permitidas
         if optimizer == 'SGD':
            self.optimizer = optim.SGD(Model.parameters(), lr=learning_rate, weight_decay=weight_decay)
         elif optimizer == 'Adam':
            self.optimizer = optim.Adam(Model.parameters(), lr=learning_rate, weight_decay=weight_decay)
         elif optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(Model.parameters(), lr=learning_rate, weight_decay=weight_decay)
         else:
            raise ValueError("Invalid optimizer. Allowed options are 'SGD', 'Adam', and 'RMSprop'.")
       
     def train(self, train_loader, val_loader,show=1):
        
        train_loss_plot=[]
        val_loss_plot=[]
        
        #patient maximum in validation
        
        # Inicializar variables para la parada temprana
        best_score = float('inf')  # Puntuación inicialmente muy alta para asegurar la actualización en la primera época
        best_model = None
        patient = 0
        max_patient = self.patient 
        
        
        for epoch in range(self.n_epochs):
             train_loss = 0.0
             val_loss = 0.0
             
             # Entrenar en los datos de entrenamiento
             for batch in train_loader:
                 inputs, targets = batch
                 inputs = inputs.to(dtype=self.model_dtype)
                 self.optimizer.zero_grad()
                 outputs = self.model(inputs)
                 loss = self.criterion(outputs, targets)
                 loss.backward()#backpropragation
                 self.optimizer.step()
                 train_loss += loss.item() * inputs.size(0)
                 
                
                 # Evaluar en los datos de validación
             with torch.no_grad():
                  for batch in val_loader:
                      inputs, targets = batch
                      inputs = inputs.to(dtype=self.model_dtype)
                      outputs = self.model(inputs)
                      loss = self.criterion(outputs, targets)
                      val_loss += loss.item() * inputs.size(0)
                  
             # Calcular las pérdidas promedio
             train_loss /= len(train_loader.dataset)
             val_loss /= len(val_loader.dataset)
             
             if val_loss < best_score and self.patient != 0:
                 best_score = val_loss
                 best_model = self.model.state_dict()  # Guardar el estado del modelo
                 patient = 0  # Reiniciar contador de paciencia
             else:
                 patient += 1

              # Comprobar si se ha alcanzado el máximo de paciencia permitido
             if patient >= self.patient and self.patient != 0:
                  print('Early stopping due to reached validation. Epochs without improvement.:', patient)
                  break
             
             
             
             
             # Imprimir el desempeño en cada época
             print(f'Epoch {epoch+1}/{self.n_epochs}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')
             
             
             if show == 1:# grafico en tiempo real
                 train_loss_plot.append(train_loss)
                 val_loss_plot.append(val_loss)
    
                 # Actualizar el gráfico en cada época
                 plt.plot(train_loss_plot, label='Training')
                 plt.plot(val_loss_plot, label='Validation')
                 plt.xlabel('Epoch')
                 plt.ylabel('loss')
                 plt.legend()
                 plt.title('Training and Validation Performance')
                 plt.show(block=False)  # Usar block=False para que el gráfico se muestre en tiempo real
                 plt.pause(0.1) # Pausar la ejecución para permitir la actualización del gráfico
        if show == 1:
          # Mostrar el gráfico final después de la finalización del entrenamiento
             plt.show()
             
     def test(self, test_loader):
          pass
    
class Normalization:
    
    def __init__(self, kindof='meanstd',value1=0, value2=1):
        
        self.value1=value1# value1=mean or min
        self.value2=value2# value1=std or max
        
        if kindof  == 'meanstd' or kindof == 'minmax':
           self.kindof=kindof
        else:
             raise ValueError("Invalid parameter. Allowed options are 'meanstd' and 'minmax'.")
        
        if kindof == 'minmax' and value1>value2:
            self.value1=value2
            self.value2=value1
       
    def minmax(self,tensor):
        return (tensor - self.value1) / (self.value2 - self.value1)
    
    def transform(self, data):
        
        if self.kindof=='meanstd':
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),  # Convierte los datos a tensores
                    transforms.Normalize(self.value1.unsqueeze(0).expand(data.shape[0], -1), #expande la misma media por todas las filas
                                         self.value2.unsqueeze(0).expand(data.shape[0], -1)),  # Normaliza los datos
                    ]
                )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),  # Convierte los datos a tensores
                    transforms.Lambda(self.minmax),  # Normaliza los datos entre valor mínimo y máximo
                    ]
                )
        return transform(data)


    