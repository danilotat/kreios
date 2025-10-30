import numpy as np
import matplotlib
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
    gap=0.0,  # Changed default to 0.0
    exon_y=0.6,
    exon_linewidth=12,
    exon_color="#808180",
    signal_color="C0",
    signal_linewidth=1.5,
    signal_label="RiboProfile",
    positionLabel="Frameshift"
):
    """
    Plot exons (top) as arrows based on strand and signal (bottom) as a line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The parent subplot/axes to host two inset axes (top and bottom).
    exons : dict[str, tuple[int, int, str]]
        Mapping exon_id -> (start, end, strand). Strand should be "+" or "-".
    signal : dict[int, float]
        Mapping genomic position -> signal value.

    Returns
    -------
    dict
        {"top_ax": Axes, "bottom_ax": Axes}
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