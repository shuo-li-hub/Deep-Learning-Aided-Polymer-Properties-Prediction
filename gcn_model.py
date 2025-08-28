class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels=100, output_dim=1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, output_dim)
            )
    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        return self.regressor(x)
