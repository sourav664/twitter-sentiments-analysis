import torch
import torch.nn as nn
import torch.nn.functional as F
import json

with open('./data/vocab.json', 'r') as f:
    vocab = json.load(f)

device = torch.device("cpu")

model_path = './models/modelV3.pth'

class LSTMModelV1(nn.Module):

    def __init__(self, input_size, embedding_dim, hidden_size, layer_dim, drop_out, bi_directional , output_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_dim = layer_dim
        self.embedding = nn.Embedding(input_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, num_layers=layer_dim, bidirectional=bi_directional, dropout=drop_out)
        self.dropout = nn.Dropout(drop_out)
        self.linear = nn.Linear(hidden_size * 2 if bi_directional else hidden_size, output_dim) # Adjust linear layer size based on bidirectionality
        self.bi_directional = bi_directional


    def forward(self, x):

        num_directions = 2 if self.bi_directional else 1
        h0 = torch.zeros(self.layer_dim * num_directions, x.size(0), self.hidden_size , device=device).requires_grad_()
        c0 = torch.zeros(self.layer_dim * num_directions, x.size(0), self.hidden_size, device=device).requires_grad_()
        embedding = self.embedding(x)
        output ,(hidden, cell) = self.lstm(embedding, (h0.detach(), c0.detach()))

        if self.bi_directional:
             # hidden[-2,:,:] corresponds to the last layer's forward hidden state
            # hidden[-1,:,:] corresponds to the last layer's backward hidden state
            hidden_combined = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden_combined = self.dropout(hidden[-1,:,:]) # Use only the last layer's hidden state for unidirectional LSTM

        # hidden_combined: (batch_size, hidden_dim * 2)

        # Pass through fully connected layer to get the final prediction
        out =  self.linear(hidden_combined)

        return out
    


# if __name__ == "__main__":
    
    
model = LSTMModelV1(input_size=len(vocab),
                embedding_dim=100,
                hidden_size=64,
                layer_dim=4,
                drop_out = 0.2,
                bi_directional = True,
                output_dim=4
                ).to(device)

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict=state_dict)
model.to(device)



