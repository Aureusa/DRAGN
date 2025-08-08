import torch
from monai.networks.nets import BasicUNet

from networks.models._base_model import BaseModel
from data_pipeline.transforms import PerImageAsinhNormalize


def load_cut_generator_only(model_path, device='cuda:0'):
    """
    Load only the generator from a trained CUT model
    
    Args:
        model_path: Path to the generator checkpoint (.pth file)
        device: Device to load the model on
        
    Returns:
        generator: Loaded generator model
    """
    import sys
    sys.path.append('/home4/s4683099/CUT')
    from models.networks import define_G
    
    # Define generator with same architecture as training
    generator = define_G(
        input_nc=1,
        output_nc=1,
        ngf=64,
        netG='resnet_9blocks',
        norm='instance',
        use_dropout=False,
        init_type='xavier',
        init_gain=0.02,
        gpu_ids=[0] if 'cuda' in device else []
    )
    
    # Load the state dict
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint)
    generator.eval()
    
    print(f"✓ Loaded generator from: {model_path}")

    sys.path.remove('/home4/s4683099/CUT')

    # Move to device
    generator.to(device)

    return generator


# TODO: Impliment the train_model, save_model, and save_train_val_loss methods
class UNet_CUT(BasicUNet, BaseModel):
    def __init__(
            self,
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            features=(32, 32, 64, 128, 256, 32),
            act=('LeakyReLU', {'inplace': True, 'negative_slope': 0.1}),
            norm=('instance', {'affine': True}),
            bias=True,
            dropout=0.1,
            upsample='deconv'
        ) -> None:
        """
        Initialize the UNet model.

        :param spatial_dims: Number of spatial dimensions.
        :type spatial_dims: int
        :param in_channels: Number of input channels.
        :type in_channels: int
        :param out_channels: Number of output channels.
        :type out_channels: int
        :param features: Number of features in each layer.
        :type features: tuple
        :param act: Activation function.
        :type act: tuple
        :param norm: Normalization layer.
        :type norm: tuple
        :param bias: Whether to use bias in the convolutional layers.
        :type bias: bool
        :param dropout: Dropout rate.
        :type dropout: float
        :param upsample: Upsampling method.
        :type upsample: str
        """
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout,
            upsample=upsample
        )

        # Load the CUT generator model and freeze its parameters
        self.CUT_G = load_cut_generator_only(
            model_path='/home4/s4683099/CUT/checkpoints/mock2real_galaxies/latest_net_G.pth',
            device='cuda:0'
        )
        for param in self.CUT_G.parameters():
            param.requires_grad = False

        # Use robust normalization instead of standard normalization
        # This preserves fine details while handling extreme spikes
        self.normalize = PerImageAsinhNormalize()
    
    def from_config(cls, config):
        """
        Config construction - this should be abstract
        """
        raise NotImplementedError("from_config method is not implemented for AttentionUNET")

    def forward(self, x):
        """
        Forward pass through the UNet model.
        
        :param x: Input tensor.
        :return: Output tensor.
        """
        return super().forward(x)

    def _preprocess_with_cut(self, tensor):
        """
        Use CUT model to transform input tensor while preserving flux
        """
        with torch.no_grad():
            # Store original flux
            original_flux = tensor.sum(dim=(2, 3), keepdim=True)  # Sum over spatial dimensions
            
            # Apply CUT transformation
            transformed = self.CUT_G(tensor)
            
            # Calculate flux after transformation
            transformed_flux = transformed.sum(dim=(2, 3), keepdim=True)
            
            # Rescale to preserve original flux
            # Avoid division by zero
            scale_factor = torch.where(
                transformed_flux > 1e-8,
                original_flux / transformed_flux,
                torch.ones_like(transformed_flux)
            )
            
            flux_conserved_output = transformed * scale_factor
            
        return flux_conserved_output

    def preprocess_with_cut(self, input_tensor, target_tensor):
        psf = input_tensor - target_tensor

        del input_tensor
        torch.cuda.empty_cache()

        target_tensor = self._preprocess_with_cut(target_tensor)
        input_tensor = target_tensor + psf  # Reconstruct targets from PSF
        return input_tensor, target_tensor

    
    def train_model(
            self,
            train_loader: torch.utils.data.DataLoader,
            loss_function: callable,
            optimizer: torch.optim.Optimizer,
            device,
        ) -> None:
        self.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            # Move the data to the device
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Compute PSF
            psf = inputs - targets

            # Remove inputs from memory
            del inputs
            torch.cuda.empty_cache()

            # Preprocess targets using CUT model and generate new inputs
            targets = self._preprocess_with_cut(targets)
            inputs = targets + psf  # Reconstruct targets from PSF

            # Normalize the inputs and targets
            inputs_normalized, _ = self.normalize(inputs)
            targets_normalized, _ = self.normalize(targets)

            # Zero the gradients
            optimizer.zero_grad()

            # Generate predictions
            outputs = self.forward(inputs_normalized)

            # Calculate the loss (in normalized space)
            loss = loss_function(inputs_normalized, outputs, targets_normalized, psf)

            # Perform backpropagation
            loss.backward()
            optimizer.step()

            # Update the loss
            epoch_loss += loss.item()
            
            # Clear cache and delete variables to free memory
            del inputs, targets, outputs, psf, inputs_normalized, targets_normalized
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        return epoch_loss / len(train_loader)

    def validate_model(
            self,
            val_loader: torch.utils.data.DataLoader,
            loss_function: callable,
            device
        ) -> float:
        """
        Validate the model.

        :param val_loader: DataLoader for validation data.
        :val_loader type: torch.utils.data.DataLoader
        :param loss_function: Loss function to use.
        :loss_function type: callable
        :return: Validation loss.
        :rtype: float
        """
        self.eval()
        val_loss = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move the data to the device
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                psf = inputs - targets

                # Remove inputs from memory to save GPU memory  
                del inputs
                torch.cuda.empty_cache()

                # Preprocess targets using CUT model and generate new inputs
                targets = self._preprocess_with_cut(targets)
                inputs = targets + psf  # Reconstruct inputs from PSF

                # Normalize the inputs and targets
                inputs_normalized, _ = self.normalize(inputs)
                targets_normalized, _ = self.normalize(targets)

                # Generate predictions
                outputs = self.forward(inputs_normalized)

                # Calculate the loss (in normalized space)
                loss = loss_function(inputs_normalized, outputs, targets_normalized, psf)

                # Update the loss
                val_loss += loss.item()
                
                # Clear cache and delete variables to free memory
                del inputs, targets, outputs, psf, inputs_normalized, targets_normalized
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        return val_loss / len(val_loader)

    def predict_denormalized(self, inputs, targets=None):
        """
        Make predictions and return them in the original (denormalized) scale.
        
        :param inputs: Input tensors
        :param targets: Target tensors (optional, for getting normalization params)
        :return: Denormalized predictions
        """
        self.eval()
        with torch.no_grad():
            # If targets are provided, use them for normalization parameters
            if targets is not None:
                # Apply CUT preprocessing
                psf = inputs - targets
                targets = self._preprocess_with_cut(targets)
                inputs = targets + psf
                
                # Normalize using target statistics
                inputs_normalized, input_params = self.normalize(inputs)
                targets_normalized, target_params = self.normalize(targets)
                
                # Make prediction
                outputs_normalized = self.forward(inputs_normalized)
                
                # Denormalize using target parameters (since we want output in target scale)
                outputs_denormalized = self.normalize.inverse(outputs_normalized, target_params)
                
                return outputs_denormalized, targets
            else:
                # If no targets, normalize inputs and denormalize using input parameters
                inputs_normalized, input_params = self.normalize(inputs)
                outputs_normalized = self.forward(inputs_normalized)
                outputs_denormalized = self.normalize.inverse(outputs_normalized, input_params)
                
                return outputs_denormalized

    def save_model(self, filename: str,  dir_: str):
        """
        Save the model to a file.
        
        :param filename: Name of the file to save the model to.
        :type name: str
        :param dir_: Directory to save the model to.
        :type dir_: str
        """
        super().save_model(filename, dir_)

    def load_model(self, filename: str,  dir_: str):
        """
        Load the model from a file.
        
        :param filename: Name of the file to load the model from.
        :type name: str
        :param dir_: Directory to load the model from.
        :type dir_: str
        """
        super().load_model(filename, dir_)
