import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm
import seaborn as sns

nature_palette = {
    "Shakespeare": "#4DBBD5",
    "Cinnabar" :"#E64B35",
    "PersianGreen" :"#00A087", 
    "Chambray" : "#3C5488",
    "Apricot" : "#F39B7F",
    "WildBlueYonder": "#8491B4",
    "MonteCarlo" : "#91D1C2",
    "Monza" :"#DC0000",
    "RomanCoffee" :"#7E6148",
    "Sandrift": "#B09C85"
}

# Set the seaborn whitegrid theme
# sns.set_theme(style="ticks")
sns.set_palette(nature_palette.values())
# Set the default font
fontPath = '../../font/HelveticaNeue-Medium.otf'
font_prop = fm.FontProperties(fname=fontPath)
fm.fontManager.addfont(fontPath)
# Set the default font size
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.family'] = font_prop.get_name()
mpl.rcParams['font.sans-serif'] = font_prop.get_name()
# Set the font size for labels and titles
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['legend.title_fontsize'] = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
# Set the black spines with width 1.5
mpl.rcParams['axes.spines.right'] = True
mpl.rcParams['axes.spines.top'] = True
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['lines.linewidth'] = 1.5
# Set normal ticks on axis
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'

def plot_riboseq_profile(
    triplet_counts, 
    positions=None, 
    variant_pos=None, 
    predicted_mean=None, 
    title="Riboseq Profile",
    figsize=(12, 10)
):
    """
    General-purpose visualizer for Riboseq codon-level data.
    Works for both Simulated ground truth and Real observed data.

    Parameters
    ----------
    triplet_counts : np.ndarray
        Shape (M, 3). Counts per codon for Frame 0, 1, 2.
    positions : np.ndarray, optional
        Shape (M,). Codon indices/coordinates. If None, uses 0..M-1.
    variant_pos : float/int, optional
        The x-coordinate to draw a vertical line (e.g., variant site).
    predicted_mean : np.ndarray, optional
        Shape (M,). A line to overlay on the abundance plot (e.g., True Mu 
        from simulation or Posterior Mean from model).
    title : str
        Main title of the figure.
    figsize : tuple
        Figure dimensions.
    """
    
    # --- Input Sanitization ---
    triplet_counts = np.asarray(triplet_counts)
    M = triplet_counts.shape[0]
    
    if positions is None:
        positions = np.arange(M)
    else:
        positions = np.asarray(positions)
        
    # Calculate Total Counts (Abundance)
    total_counts = triplet_counts.sum(axis=1)

    # Prepare Figure
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # =======================================================
    # PLOT 1: Abundance (Total Counts vs Model)
    # =======================================================
    ax = axes[0]
    ax.bar(positions, total_counts, color='gray', alpha=0.6, width=1.0, label='Observed Counts')
    
    if predicted_mean is not None:
        ax.plot(positions, predicted_mean, color='red', lw=2, linestyle='--', label='Model/True Mean')
    
    if variant_pos is not None:
        ax.axvline(variant_pos, color='black', linestyle='-', lw=2, label='Variant')
        
    ax.set_ylabel("Total Ribosome Counts")
    ax.set_title(f"{title}: Abundance & Ramp")
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    # =======================================================
    # PLOT 2: Frame Periodicity (Stacked Fractions)
    # =======================================================
    ax = axes[1]
    
    # Normalize counts to fractions
    # Handle division by zero for empty codons
    row_sums = triplet_counts.sum(axis=1)
    safe_sums = row_sums.copy()
    safe_sums[safe_sums == 0] = 1 
    fracs = triplet_counts / safe_sums[:, None]
    
    # If a row was 0, fractions are 0, which is fine for stackplot
    
    pal = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green
    ax.stackplot(positions, fracs[:,0], fracs[:,1], fracs[:,2], 
                 labels=['Frame 0', 'Frame 1', 'Frame 2'],
                 colors=pal, alpha=0.8)
    
    if variant_pos is not None:
        ax.axvline(variant_pos, color='black', linestyle='-', lw=2)
        
    ax.set_ylabel("Frame Fraction")
    ax.set_title("Reading Frame Fidelity")
    ax.set_ylim(0, 1)
    ax.legend(loc='lower left', frameon=True, framealpha=0.9)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    # =======================================================
    # PLOT 3: Detailed Nucleotide Counts (Grouped Bar)
    # =======================================================
    ax = axes[2]
    
    width = 0.3
    # Shift bars slightly so they sit side-by-side on the codon integer tick
    ax.bar(positions - width, triplet_counts[:,0], width=width, color=pal[0], label='Frame 0')
    ax.bar(positions,        triplet_counts[:,1], width=width, color=pal[1], label='Frame 1')
    ax.bar(positions + width, triplet_counts[:,2], width=width, color=pal[2], label='Frame 2')
    
    if variant_pos is not None:
        ax.axvline(variant_pos, color='black', linestyle='-', lw=2)

    ax.set_ylabel("Counts per Frame")
    ax.set_xlabel("Codon Position")
    ax.set_title("Nucleotide-Level Counts (Log Scale)")
    
    # Log scale is usually necessary for Riboseq to see UTRs vs CDS
    ax.set_yscale('symlog', linthresh=1.0) 
    
    # Format x-axis
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))

    plt.tight_layout()
    return fig, axes


def plot_frameshift_signal(
    pre_profile: np.ndarray,
    after_profile: np.ndarray,
    variant_rel_pos: int,
    features: list = None,
    ax=None,
    title: str = None,
    figsize: tuple = (10, 4),
    show_frames: bool = False,
):
    """
    Plot ribosome signal around a frameshift variant with features annotation.

    Parameters
    ----------
    pre_profile : np.ndarray
        Signal profile before the variant.
    after_profile : np.ndarray
        Signal profile after the variant.
    variant_rel_pos : int
        Relative position of the variant (used for the dashed line).
    features : list, optional
        List of (feature_type, rel_start, rel_end) tuples from TranscriptCollector.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    title : str, optional
        Title for the plot.
    figsize : tuple, optional
        Figure size if creating a new figure.
    show_frames : bool, optional
        If True, plot signal for all three reading frames (frame 0, 1, 2)
        starting from the start codon position. Requires features to contain
        a 'start_codon' entry.

    Returns
    -------
    tuple : (ax_features, ax_signal) or ax_signal if no features
        The axes with the plot.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Build coordinates and concatenate profiles
    pre_coords = np.arange(len(pre_profile))
    after_coords = np.arange(variant_rel_pos, variant_rel_pos + len(after_profile))

    full_coords = np.concatenate([pre_coords, after_coords])
    full_signal = np.concatenate([pre_profile, after_profile])

    # Calculate xlim for both axes
    xlim = (full_coords.min(), full_coords.max())

    # Hide the parent axes (used only as container)
    try:
        ax.set_axis_off()
    except AttributeError:
        pass

    if features:
        top_height = 0.2
        gap = 0
        bottom_height = 1.0 - top_height - gap

        ax_features = inset_axes(
            ax, width="100%", height=f"{top_height*100:.0f}%",
            loc='upper left', bbox_to_anchor=(0, 0, 1, 1),
            bbox_transform=ax.transAxes, borderpad=0
        )
        ax_signal = inset_axes(
            ax, width="100%", height=f"{bottom_height*100:.0f}%",
            loc='lower left', bbox_to_anchor=(0, 0, 1, bottom_height),
            bbox_transform=ax.transAxes, borderpad=0
        )

        # --- Plot features on top axis ---
        feature_colors = {
            'start_codon': nature_palette['Monza'],
            'CDS': nature_palette['Sandrift'],
        }

        for feat_type, start, end in features:
            color = feature_colors.get(feat_type, 'gray')
            ax_features.plot([start, end], [0.5, 0.5], color=color, lw=5, solid_capstyle='butt', label=feat_type)

        # Variant line on features axis
        ax_features.axvline(variant_rel_pos, color='red', linestyle='--', lw=1, alpha=0.7)

        ax_features.set_xlim(xlim)
        ax_features.set_ylim(0, 1)
        ax_features.set_yticks([])
        # ax_features.set_xticks([])
        for direction in ['top', 'left', 'right',]:
            ax_features.spines[direction].set_visible(False)

        # Deduplicate legend
        handles, labels = ax_features.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_features.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8, frameon=False)

        if title:
            ax_features.set_title(title)
    else:
        # No features, use entire area for signal
        ax_signal = inset_axes(
            ax, width="100%", height="100%",
            loc='center', bbox_to_anchor=(0, 0, 1, 1),
            bbox_transform=ax.transAxes, borderpad=0
        )
        ax_features = None
        if title:
            ax_signal.set_title(title)

    # --- Plot signal on bottom axis ---
    if show_frames and features:
        # Find start codon position from features
        start_codon_pos = None
        for feat_type, start, end in features:
            if feat_type == 'start_codon':
                start_codon_pos = start
                break

        if start_codon_pos is not None:
            # Slice signal from start codon position
            mask = full_coords >= start_codon_pos
            sliced_coords = full_coords[mask]
            sliced_signal = full_signal[mask]

            # Determine frame for each position based on distance from start codon
            rel_coords = sliced_coords - start_codon_pos
            frames = rel_coords % 3

            # Color each bar by its frame
            frame_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
            bar_colors = [frame_colors[f] for f in frames]

            ax_signal.bar(sliced_coords, sliced_signal, width=1, color=bar_colors,
                         edgecolor='none', alpha=0.8)

            # Track that we need custom legend with frame colors
            _show_frame_legend = True
        else:
            # No start codon found, fall back to default plotting
            ax_signal.bar(full_coords, full_signal, width=1, color='steelblue',
                         edgecolor='none', alpha=0.8)
            _show_frame_legend = False
    else:
        ax_signal.bar(full_coords, full_signal, width=1, color='steelblue',
                     edgecolor='none', alpha=0.8)
        _show_frame_legend = False

    ax_signal.axvline(variant_rel_pos, color='red', linestyle='--', lw=1.5)
    ax_signal.set_xlim(xlim)
    ax_signal.set_xlabel("Relative coordinate")
    ax_signal.set_ylabel("Signal")
    ax_signal.spines['top'].set_visible(False)
    ax_signal.spines['right'].set_visible(False)

    # Build legend with frame colors if needed
    if _show_frame_legend:
        frame_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        handles = [Patch(facecolor=c, label=f'Frame {i}') for i, c in enumerate(frame_colors)]
        handles.append(Line2D([0], [0], color='red', linestyle='--', lw=1.5, label='Frameshift'))
        ax_signal.legend(handles=handles, loc='upper right', fontsize=8, frameon=False)
    else:
        ax_signal.legend(['Frameshift'], loc='upper right', fontsize=8, frameon=False)

    if ax_features is not None:
        return ax_features, ax_signal
    return ax_signal