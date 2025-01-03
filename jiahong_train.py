import torch
import torch.optim as optim
from torch.nn import MSELoss
from tqdm import tqdm
from BPP import BinPackingDataset
from model import BPPModel
from torch.utils.data import DataLoader
from BPP import train_loader, test_loader



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BPPModel().to(device)
criterion = MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training parameters
num_epochs = 50
clip = 1.0  # Gradient clipping

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        dimensions, rotations, positions, masks, lengths = batch
        dimensions = dimensions.to(device)  # [batch_size, seq_len, 3]
        rotations = rotations.to(device)    # [batch_size, seq_len, 3]
        positions = positions.to(device)    # [batch_size, seq_len, 3]
        masks = masks.to(device)            # [batch_size, seq_len]

        # Prepare input and target
        src = torch.cat([dimensions, rotations], dim=2)  # [batch_size, seq_len, 6]
        tgt = torch.cat([positions, rotations], dim=2)   # [batch_size, seq_len, 6]

        # Shift tgt for decoder input (Teacher Forcing)
        tgt_input = torch.cat([torch.zeros(tgt.size(0),1,6).to(device), tgt[:,:-1,:]], dim=1)

        # Create masks
        src_mask = masks  # [batch_size, src_seq_len]
        tgt_mask_flag = masks  # Assuming tgt has the same mask

        optimizer.zero_grad()
        output = model(src, tgt_input, src_mask, tgt_mask_flag)  # [batch_size, seq_len, 6]
        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            dimensions, rotations, positions, masks, lengths = batch
            dimensions = dimensions.to(device)
            rotations = rotations.to(device)
            positions = positions.to(device)
            masks = masks.to(device)

            # Prepare input and target
            src = torch.cat([dimensions, rotations], dim=2)
            tgt = torch.cat([positions, rotations], dim=2)

            # Shift tgt for decoder input
            tgt_input = torch.cat([torch.zeros(tgt.size(0),1,6).to(device), tgt[:,:-1,:]], dim=1)

            # Create masks
            src_mask = masks
            tgt_mask_flag = masks

            output = model(src, tgt_input, src_mask, tgt_mask_flag)
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# Training Loop
best_test_loss = float('inf')
for epoch in range(1, num_epochs + 1):
    print(f"Epoch {epoch}/{num_epochs}")
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    # Save the model if it has the best test loss so far
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), "best_bpp_model.pth")
        print("Model saved.")