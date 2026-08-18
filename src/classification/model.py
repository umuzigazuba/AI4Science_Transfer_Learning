import torch
import lightning


class CNNModel_Deep(lightning.LightningModule):
    def __init__(self, dropout = 0.3):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(4, 32, kernel_size=7, padding=3),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.Conv1d(32, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.Conv1d(32, 62, kernel_size=5, padding=2),
            torch.nn.BatchNorm1d(62),
            torch.nn.ReLU(),

            torch.nn.Conv1d(62, 62, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(62),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1)               
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(16, 3),
        )
        self.save_hyperparameters()

    def forward(self, x, meta):

        x = self.cnn(x)             
        x = x.flatten(start_dim = 1)  
        x = torch.cat([x, meta], dim = 1)
        return self.mlp(x)


class CNNModel_Shallow(lightning.LightningModule):
    def __init__(self, dropout = 0.3):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(4, 32, kernel_size=7, padding=3),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.Conv1d(32, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.Conv1d(32, 62, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(62),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1)               
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, 3),
        )
        self.save_hyperparameters()

    def forward(self, x, meta):

        x = self.cnn(x)             
        x = x.flatten(start_dim = 1)  
        x = torch.cat([x, meta], dim = 1)
        return self.mlp(x)


class GRUModel(lightning.LightningModule):
    def __init__(
        self, input_size, hidden_size = 256, mlp_hidden_size = 64, dropout = 0.3):

        super().__init__()
 
        self.gru_layer_1 = torch.nn.GRU(
            input_size, hidden_size, num_layers = 1, batch_first = True, bidirectional = True,
        )
        self.gru_layer_2 = torch.nn.GRU(
            hidden_size * 2, hidden_size, num_layers = 1, batch_first = True, bidirectional = True,
        ) # *2 because bidirectional = True
        self.hidden_bn = torch.nn.BatchNorm1d(hidden_size * 2)
        self.dropout = torch.nn.Dropout(dropout)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear((hidden_size * 2) + 2, mlp_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(mlp_hidden_size, 3),
        )
        self.save_hyperparameters()


    def forward(self, x, meta):
        x = x.transpose(1, 2)

        out1, _ = self.gru_layer_1(x)
        out1 = self.dropout(out1)
        out1 = self.hidden_bn(out1.transpose(1, 2)).transpose(1, 2)
 
        out2, _ = self.gru_layer_2(out1)
        out2 = self.dropout(out2)
        out2 = self.hidden_bn(out2.transpose(1, 2)).transpose(1, 2)
 
        last_hidden = out2[:, -1, :]

        last_hidden = torch.cat([last_hidden, meta], dim = 1) 
        output = self.mlp(last_hidden)

        return output

