import torch.nn as nn



class NNForecater(nn.Module):
    def __init__(self, type, hyperparameters):
        super(NNForecater, self).__init__()
        if type=="wind":
            self.MLP=WindMLP(hyperparameters["input_size"],hyperparameters["hidden_size"],hyperparameters["output_size"])
        elif type=="solar":
            self.MLP=SolarMLP(hyperparameters["input_size"],hyperparameters["hidden_size"],hyperparameters["output_size"])

    def fit(self,):
        pass

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out




class SolarMLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(SolarMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.relu = nn.ReLU()  
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out
    
class WindMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(WindMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.relu = nn.ReLU()  
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out