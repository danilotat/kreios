import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_exons_and_signal(
    ax,
    exons: dict,
    signal: dict,
    position: int,
    *,
    pos_bw=None,
    neg_bw=None,
    flanking=300,
    top_height=0.35,
    gap=0.0,
    exon_y=0.6,
    exon_linewidth=12,
    exon_color="#808180",
    signal_color="C0",
    signal_linewidth=1.5,
    signal_label="RiboProfile",
    positionLabel="Frameshift"
):
    """
    Plot exons as arrows and signal as a line on separate axes within a given matplotlib Axes.
    This function creates two inset axes within the provided parent Axes: 
    one for plotting exons as arrows (top) and another for plotting signal data as a line (bottom). 
    It also supports optional additional signal tracks (e.g., forward and reverse signals) 
    and highlights a specific genomic position with a vertical dashed line.
        Mapping of exon_id to a tuple (start, end, strand). 
        Strand should be "+" for forward or "-" for reverse.
        Mapping of genomic position to signal value.
    
    Parameters:
        position : int
            The genomic position to highlight with a vertical dashed line.
        pos_bw : list[float], optional
            Additional signal data for the forward strand to overlay on the bottom axis.
        neg_bw : list[float], optional
            Additional signal data for the reverse strand to overlay on the bottom axis.
        flanking : int, default=300
            The number of bases to display on either side of the highlighted position.
        top_height : float, default=0.35
            The height of the top axis (exons) as a fraction of the parent Axes height.
        gap : float, default=0.0
            The vertical gap between the top and bottom axes as a fraction of the parent Axes height.
        exon_y : float, default=0.6
            The vertical position of the exon arrows within the top axis.
        exon_linewidth : int, default=12
            The line width of the exon arrows.
        exon_color : str, default="#808180"
            The color of the exon arrows.
        signal_color : str, default="C0"
            The color of the signal line.
        signal_linewidth : float, default=1.5
            The line width of the signal line.
        signal_label : str, default="RiboProfile"
            The label for the signal axis.
        positionLabel : str, default="Frameshift"
            The label for the highlighted position.
    
    Notes
        - The function automatically adjusts the x-axis limits to encompass all exons and signal data.
        - The top axis (exons) includes a legend to indicate strand direction (forward or reverse).
        - The bottom axis (signal) includes optional overlays for additional signal tracks (e.g., pos_bw, neg_bw).
        - The function ensures synchronized x-axis limits between the top and bottom axes.

    Returns:
        A dictionary containing the two inset Axes:
        - "top_ax": The Axes for plotting exons.
        - "bottom_ax": The Axes for plotting the signal.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from matplotlib.lines import Line2D
    
    # Compute x-limits from both exons and signal
    x_candidates = []
    if exons:
        for s, e, _ in exons.values():
            x_candidates.extend([s, e])
    if signal:
        x_candidates.extend(signal.keys())
    if x_candidates:
        xmin, xmax = min(x_candidates), max(x_candidates)
        if xmin == xmax:  # avoid zero-width xlim
            xmin -= 0.5
            xmax += 0.5
    else:
        xmin, xmax = 0.0, 1.0

    # Hide parent ax spines and ticks
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Create two inset axes within the given Axes: top for exons, bottom for signal
    # bounds = [x0, y0, width, height] in Axes-relative coordinates
    top_ax = ax.inset_axes([0.0, 1.0 - top_height, 1.0, top_height], zorder=2)
    bottom_ax = ax.inset_axes([0.0, 0.0, 1.0, 1.0 - top_height - gap], zorder=1)  # Removed the /2
    
    # Plot bottom signal
    if signal:
        xs = np.array(sorted(signal.keys()), dtype=float)
        ys = np.array([signal[x] for x in xs], dtype=float)
        bottom_ax.plot(xs, ys, color=signal_color, linewidth=signal_linewidth, label='riboP')
        
        # Plot pos_bw if provided
        if pos_bw is not None:
            # Ensure pos_bw matches xs length or handle accordingly
            if len(pos_bw) >= len(xs):
                bottom_ax.plot(xs, pos_bw[:len(xs)], color='green', label='fw')
            else:
                bottom_ax.plot(xs[:len(pos_bw)], pos_bw, color='green', label='fw')
        
        # Plot neg_bw if provided
        if neg_bw is not None:
            # Ensure neg_bw matches xs length or handle accordingly
            if len(neg_bw) >= len(xs):
                bottom_ax.plot(xs, neg_bw[:len(xs)], color='grey', label='rev')
            else:
                bottom_ax.plot(xs[:len(neg_bw)], neg_bw, color='grey', label='rev')
    bottom_ax.set_xlim(xmin, xmax)
    bottom_ax.set_ylabel(signal_label)
    
    # Plot top exons as arrows based on strand
    top_ax.set_xlim(xmin, xmax)
    top_ax.set_ylim(0.0, 1.0)
    
    # display first an axhline behind the arrows
    top_ax.axhline(y=exon_y, lw=1, color='black', zorder=1)
    
    for exon_id, (start, end, strand) in sorted(exons.items(), key=lambda kv: kv[1][0]):
        # Draw simple line
        line = Line2D(
            [start, end], [exon_y, exon_y],
            linewidth=5,
            color="grey" if strand == "-" else "green",
            solid_capstyle='butt',  # Square ends instead of rounded
            zorder=2
        )
        top_ax.add_line(line)
        
    
    # Clean top axis visuals - drop all spines
    for spine in top_ax.spines.values():
        spine.set_visible(False)
    top_ax.set_ylabel("Exons", rotation=0, labelpad=20)
    top_ax.set_xticks([])
    top_ax.set_yticks([])

    # Add legend for strand colors
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=5, label='fw'),
        Line2D([0], [0], color='grey', linewidth=5, label='rev')
    ]
    top_ax.legend(
        handles=legend_elements, 
        loc='lower center', 
        bbox_to_anchor=(1, -0.2), 
        frameon=False, 
        fontsize=9
    )

    # Clean bottom axis visuals - drop top and right spines
    bottom_ax.spines['top'].set_visible(False)
    bottom_ax.spines['right'].set_visible(False)

    # Keep bottom axis with x ticks; ensure zoom/pan sync by mirroring x-lims
    def _sync_xlims(event_ax):
        if event_ax is bottom_ax:
            top_ax.set_xlim(bottom_ax.get_xlim())
    bottom_ax.callbacks.connect("xlim_changed", _sync_xlims)
    bottom_ax.grid(axis="y")
    
    # Center on the position
    for ax_block in (top_ax, bottom_ax):
        ax_block.set_xlim(position - flanking, position + flanking)
    
    # Add marker on bottom_ax (where it will be visible)
    bottom_ax.axvline(x=position, color="black", lw=2, linestyle='--', zorder=10)
    
    # Add marker label on top_ax
    max_y = bottom_ax.get_ylim()[1]
    bottom_ax.text(x=position+(flanking // 60), y=0.75 * max_y, s=positionLabel, ha='left', 
                   fontsize=11, color="black", zorder=10)
    
    return {"top_ax": top_ax, "bottom_ax": bottom_ax}

def plot_samplewise_riboprofile(rb, df, exon_collector, pos_bw=None, neg_bw=None, flanking=200, **kwargs):
    """
    Plots sample-wise ribosomal profiles along with exon structures and optional signal tracks.

    Parameters:
        rb (RibotishReader): An object containing ribosomal profiles.
        df (pd.DataFrame): A DataFrame containing information about the samples to plot. 
                           It must include columns 'tid' (transcript ID) and 'varID' (variant ID in "chrom:pos" format).
        exon_collector: An object that provides exon structures for the given transcript IDs.
        pos_bw: (Optional) A pyBigWig object for positive strand signal data.
        neg_bw: (Optional) A pyBigWig object for negative strand signal data.
        flanking (int, optional): The number of bases to include upstream and downstream of the position. Default is 200.
        **kwargs: Additional keyword arguments passed to the `plot_exons_and_signal` function.

    Notes:
        - The function creates subplots for each sample in the DataFrame.
        - If a ribosomal profile is not available for a given transcript ID, the corresponding subplot will be empty.
        - Signal data from `pos_bw` and `neg_bw` is queried for the specified genomic region and plotted if provided.
        - Exon structures are retrieved from the `exon_collector` object.

    Returns:
        None: The function generates a matplotlib figure with subplots for each sample.
    """
    # Determine the number of rows and columns for subplots
    n_rows = (len(df) + 3) // 4  # Ensure enough rows for all plots
    fig, axs = plt.subplots(n_rows, 4, figsize=(7 * 4, n_rows * 3))
    axs = axs.flatten()  # Flatten the axes array for easy indexing

    for row, ax in zip(df.itertuples(), axs):
        rprofile = rb.ribo_profiles.get(row.tid, None)
        # we need to query the bw files using coordinates here
        chrom = row.varID.split(":")[0]
        pos = int(row.varID.split(":")[1])
        range_start, range_end = pos-flanking, pos+flanking
        if pos_bw:
            pos_signal = np.array(pos_bw.values(chrom, range_start, range_end))
            pos_signal = np.where(np.isnan(pos_signal), 0, pos_signal)
        if neg_bw:
            neg_signal = neg_bw.values(chrom, range_start, range_end)
            neg_signal = np.where(np.isnan(neg_signal), 0, neg_signal)
        if rprofile:
            plot_exons_and_signal(
                ax=ax,
                exons=exon_collector._features[row.tid],
                signal=rprofile,
                position=int(row.varID.split(":")[1]),
                flanking=flanking,
                pos_bw=pos_signal,
                neg_bw=neg_signal,
                **kwargs
            )
            # plot_riboprofile(rprofile, int(row.varID.split(":")[1]), ax)
            ax.set_title(row.varID)
        else:
            ax.set_title(f"No profile for {row.tid}")
            ax.axis("off")  # Hide the axis if no profile is available

    # Hide any unused axes
    for ax in axs[len(df):]:
        ax.axis("off")

    fig.tight_layout()