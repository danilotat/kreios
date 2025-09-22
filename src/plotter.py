import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

fontPath = '/CTGlab/home/danilo/.fonts/HelveticaNeue-Medium.otf'
font_prop = fm.FontProperties(fname=fontPath, size=14)
fm.fontManager.addfont(fontPath)
# Set the default font size
mpl.rcParams['font.size'] = 16
mpl.rcParams['font.family'] = font_prop.get_name()
mpl.rcParams['font.sans-serif'] = font_prop.get_name()

class TensorVisualizer:
    """
    A class to visualize the multi-channel tensors produced by the ReadFetcher.
    It dynamically creates grayscale subplots for each tensor in the input dictionary.
    """

    def __init__(self):
        """Initializes the visualizer."""
        # No color setup needed for simple grayscale plotting.
        pass

    def _get_plot_params(self, channel_name: str) -> dict:
        """
        Returns suggested plotting parameters (like vmin, vmax) for known channels.
        This helps normalize the visual contrast for quality scores.
        """
        name = channel_name.lower()
        if 'base_qualities' in name:
            return {'vmin': 0, 'vmax': 45, 'label': 'Phred Quality Score'}
        elif 'alignment_qualities' in name:
            return {'vmin': 0, 'vmax': 60, 'label': 'Mapping Quality (MAPQ)'}
        elif 'sequence' in name:
            # For uint8 encoded sequences like DeepVariant's
            return {'vmin': 0, 'vmax': 255, 'label': 'Encoded Base Value'}
        elif 'frequency' in name:
             # For uint8 encoded allele frequency
            return {'vmin': 0, 'vmax': 255, 'label': 'Encoded Allele Freq.'}
        else:
            # Default for unknown channels
            return {'vmin': None, 'vmax': None, 'label': 'Value'}

    def plot(self, data_tensors: dict, chromosome: str, start: int, end: int):
        """
        Generates and displays a dynamic, multi-panel grayscale plot for the given tensors.

        Args:
            data_tensors (dict): The dictionary of tensors from ReadFetcher.
                                 e.g., {'sequence_color': tensor, 'base_qualities': tensor}
            chromosome (str): The chromosome name for the title.
            start (int): The start coordinate for the title and x-axis.
            end (int): The end coordinate for the title and x-axis.
        """
        # Filter out empty or non-tensor entries
        valid_tensors = {
            k: v for k, v in data_tensors.items() 
            if isinstance(v, torch.Tensor) and v.nelement() > 0
        }

        if not valid_tensors:
            print(f"No valid tensor data found for {chromosome}:{start}-{end}. Nothing to plot.")
            return

        num_channels = len(valid_tensors)
        
        # --- Create the plot ---
        fig, axes = plt.subplots(
            num_channels, 1, 
            figsize=(20, 4 * num_channels), # Adjust height based on number of channels
            sharex=True, 
            sharey=True
        )
        fig.suptitle(f"Read Pileup Visualization for {chromosome}:{start}-{end}", fontsize=18)
        
        # If there's only one channel, axes is not a list, so we make it one
        if num_channels == 1:
            axes = [axes]

        for i, (channel_name, tensor) in enumerate(valid_tensors.items()):
            ax = axes[i]
            
            # Move tensor to CPU and convert to NumPy for plotting
            numpy_tensor = tensor.cpu().numpy()
            
            # Get plotting parameters for this channel
            params = self._get_plot_params(channel_name)

            # Display the tensor as an image
            im = ax.imshow(
                numpy_tensor,
                cmap='gray',
                aspect='auto',
                interpolation='none',
                extent=[start, end, numpy_tensor.shape[0], 0],
                vmin=params['vmin'],
                vmax=params['vmax']
            )
            
            # Set titles and labels
            # Clean up the channel name for a nice title
            clean_title = channel_name.replace('_', ' ').title()
            ax.set_title(clean_title)
            ax.set_ylabel("Read Index (Centered)")
            
            # Add a color bar for the current subplot
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(params['label'])

        # Set the x-axis label only on the very last subplot
        axes[-1].set_xlabel("Genomic Position")
        
        plt.tight_layout(rect=[0, 0.01, 1, 0.96]) # Adjust for suptitle and labels
        return fig