import os, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import Linear
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    PROJECT_ROOT,
    ARTIFACT_DIR,
)
from src.data_loader import load_zFC_df
from src.functional_connectivity import fisher_z2r


class FCMLP(torch.nn.Module):
    def __init__(
            self, 
            input_dim,
            hidden_dim: int = 64, 
            output_dim: int = 2,
        ):
        super(FCMLP, self).__init__()
        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def predict(self, x, return_probabilities = True):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            if return_probabilities:
                return probs.argmax(dim=1), probs
            return probs.argmax(dim=1)
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self.forward(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            if X_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    val_outputs = self.forward(X_val)
                    val_loss = criterion(val_outputs, y_val)
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            else:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device=None):
        if device is None:
            device = next(self.parameters()).device

        state_dict = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state_dict)
        return self

    def evaluate(self, X_test, y_test):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(X_test)
            predicted = outputs.argmax(dim=1)
            accuracy = (predicted == y_test).float().mean().item()
        print(f'Accuracy: {accuracy:.4f}')