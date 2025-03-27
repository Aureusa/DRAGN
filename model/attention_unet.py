from monai.networks.nets import AttentionUnet
import torch
from tqdm import tqdm
from copy import deepcopy

class AttentionUNET(AttentionUnet):
    def __init__(
            self,
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(64, 128, 256, 512),
            strides=(2, 2, 2),
            kernel_size=3,
            up_kernel_size=3,
            dropout=0.1,
            *args,
            **kwargs
        ) -> None:
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            up_kernel_size=up_kernel_size,
            dropout=dropout,
            *args,
            **kwargs
        )

        # Define the training and validation loss history
        self._train_loss = []
        self._val_loss = []

    @property
    def train_loss(self):
        return deepcopy(self._train_loss)
    
    @property
    def val_loss(self):
        return deepcopy(self._val_loss)

    def forward(self, x):
        return super().forward(x)
    
    def train_model(self, train_loader, val_loader, lr, loss_function, num_epochs):
        # Define the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        epoch_loss = 1e9
        val_loss = 1e9
        # Iterate over epochs
        for _ in tqdm(
            range(num_epochs),
            desc=f'Training. Train Loss: {epoch_loss/len(train_loader)}; Val Loss: {val_loss/len(val_loader)}'
        ):
            self.train()
            epoch_loss = 0
            for inputs, targets, psf in train_loader:
                # Move the data to the device
                inputs = inputs.to(device)
                targets = targets.to(device)
                psf = psf.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Generate predictions
                outputs = self.forward(inputs)

                # Calculate the loss
                loss = loss_function(inputs, outputs, targets, psf)

                # Perform backpropagation
                loss.backward()
                optimizer.step()

                # Update the loss
                epoch_loss += loss.item()

            # Evaluate the model
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets, psf in val_loader:
                    # Move the data to the device
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    psf = psf.to(device)

                    # Generate predictions
                    outputs = self.forward(inputs)

                    # Calculate the loss
                    loss = loss_function(inputs, outputs, targets, psf)

                    # Update the loss
                    val_loss += loss.item()

            # Save the training and validation loss
            self._train_loss.append(epoch_loss / len(train_loader))
            self._val_loss.append(val_loss / len(val_loader))
