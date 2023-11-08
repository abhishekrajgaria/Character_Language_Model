import torch
import torch.nn as nn
from torch.autograd import Variable

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Pytorch version: {torch.__version__}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU available.")


class LMModel(nn.Module):
    def __init__(self, vocab_dim, emb_dim=50, hidden_dim=200, num_layers=1):
        super(LMModel, self).__init__()
        self.vocab_dim = vocab_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.projection_layer = nn.Embedding(vocab_dim, emb_dim)

        self.lstm = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(hidden_dim, 300, bias=True),
            nn.ReLU(),
            nn.Linear(300, self.vocab_dim, bias=True),
        )

    def forward(self, input):
        self.batch_size = input.shape[0]
        input_emb = self.projection_layer(input)

        h_0 = Variable(
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(device)
        )
        c_0 = Variable(
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(device)
        )

        lstm_output, _ = self.lstm(input_emb, (h_0, c_0))
        output = self.fc_layer(lstm_output)

        return output


if __name__ == "__main__":
    model = LMModel(386)
