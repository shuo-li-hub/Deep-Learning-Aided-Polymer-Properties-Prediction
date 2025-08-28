from sklearn.model_selection import KFold
def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = F.l1_loss(pred, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch)
            loss = F.l1_loss(pred, batch.y)
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for property in targets:
    data_list=load_data(train_list, property=property, is_test=False)
    input_dim=data_list[0].x.shape[1]
    print(f'{property} prediction')
    for fold, (train_idx, val_idx) in enumerate(kf.split(data_list)):
        print(f"\n🌀 Fold {fold+1}/5")
        train_set = [data_list[i] for i in train_idx]
        val_set = [data_list[i] for i in val_idx]
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=64)
        model = GCN(in_channels=input_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

        best_val = float('inf')
        patience, trigger = 40, 0
        for epoch in range(300):
            loss = train(model, train_loader, optimizer)
            val_loss = evaluate(model, val_loader)
            if epoch % 10 == 0:
                print(f'Epoch {epoch:03d} | Train: {loss:.4f} | Val: {val_loss:.4f}')
                if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), f'model_{property}_fold{fold}.pt')
                trigger = 0
            else:
                trigger += 1
                if trigger >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
    print(f'✅ Best val MAE for {property}: {best_val:.4f}')
