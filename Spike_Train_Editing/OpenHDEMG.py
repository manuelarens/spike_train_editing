import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import seaborn as sns
import pandas as pd
import numpy as np
from openhdemg.library.mathtools import compute_sil



def plot_ipts(
    emgfile,
    addrefsig=False,
    timeinseconds=True,
    figsize=[20, 15],
    showimmediately=True,
    tight_layout=True,
):
    """
    Plot the IPTS (decomposed source) with optional reference signal.
    
    Parameters
    ----------
    emgfile : dict
        The dictionary containing the emgfile.
    addrefsig : bool, default False
        If True, overlay the reference signal.
    timeinseconds : bool, default True
        Whether to show the time on the x-axis in seconds (True) or in samples (False).
    figsize : list, default [20, 15]
        Size of the figure in centimeters [width, height].
    showimmediately : bool, default True
        If True (default), plt.show() is called and the figure showed to the user.
    tight_layout : bool, default True
        If True (default), plt.tight_layout() is called and the figure's layout is improved.
        
    Returns
    -------
    fig : pyplot `~.figure.Figure`
    """
    
    # Check to have the IPTS in a pandas dataframe
    if isinstance(emgfile["IPTS"], pd.DataFrame):
        ipts = emgfile["IPTS"]
    else:
        raise TypeError(
            "IPTS is probably absent or it is not contained in a dataframe"
        )
    

    # Check to have the MUPULSES in a pandas dataframe
    if isinstance(emgfile["MUPULSES"], list):
        mu_pulses = []
        for pulses in emgfile["MUPULSES"]:
            # Convert each list of firing times to seconds
            converted_pulses = [pulse / emgfile["FSAMP"] for pulse in pulses]
            mu_pulses.append(converted_pulses)
    else:
        raise TypeError(
            "MUPULSES is probably absent or it is not contained in a dataframe"
        )

    # Here we produce an x axis in seconds or samples
    if timeinseconds:
        x_axis = ipts.index / emgfile["FSAMP"]
    else:
        x_axis = ipts.index

    # Create a single subplot
    fig, ax1 = plt.subplots(
        figsize=(figsize[0] / 2.54, figsize[1] / 2.54),
        num="IPTS",
    )
    

    # Initialize the current MU index
    current_index = 0

    def plot_current_mu(mu_index):
        ax1.clear()  # Clear the previous plot
        if isinstance(mu_index, int):
            ax1.plot(x_axis, ipts[mu_index])
            ax1.set_ylabel(f"MU {mu_index +1}")

            pulses = mu_pulses[mu_index]
            
            # Plot red circles at peaks (x = pulses, y = ipts value at pulse time)
            for pulse in pulses:
                # Find the corresponding y-value at this time (interpolate if necessary)
                closest_idx = (np.abs(x_axis - pulse)).argmin()
                y_value = ipts.iloc[closest_idx, mu_index]  # Get the corresponding y-value
                
                # Add a red circle at the pulse and the corresponding y-value
                ax1.plot(pulse, y_value, 'ro', markersize=2, label='Peak') 
            sil_value = compute_sil(ipts[mu_index], emgfile["MUPULSES"][mu_index])

            # Display the SIL value at the top center of the plot
            ax1.text(
                0.5, 1.05,  # (x, y) position in axes coordinates (0.5 is center, 1.05 is above the plot)
                f"SIL = {sil_value:.6f}",  # Format the SIL value to 2 decimal places
                ha='center',  # Horizontal alignment
                va='center',  # Vertical alignment
                transform=ax1.transAxes,  # Use axes coordinates (0 to 1 for both x and y)
                fontsize=10,  # You can adjust the font size if needed
                fontweight='bold'  # Bold text
            )



    # Plot the initial MU
    plot_current_mu(current_index)
    ax1.set_xlabel("Time (Sec)" if timeinseconds else "Samples")

    # Optional reference signal
    if addrefsig:
        ax2 = ax1.twinx()
        xref = (
            emgfile["REF_SIGNAL"].index / emgfile["FSAMP"]
            if timeinseconds
            else emgfile["REF_SIGNAL"].index
        )
        sns.lineplot(
            x=xref,
            y=emgfile["REF_SIGNAL"][0],
            color="0.4",
            ax=ax2,
        )
        ax2.set_ylabel("MVC")

    showgoodlayout(tight_layout, despined="2yaxes" if addrefsig else False)

    def zoom(event):
        current_xlim = ax1.get_xlim()
        zoom_factor = 0.85 if event.button == 'up' else 1.15
        midpoint = (current_xlim[0] + current_xlim[1]) / 2
        delta = (current_xlim[1] - current_xlim[0]) * zoom_factor / 2

        new_xlim = (midpoint - delta, midpoint + delta)

        if min(new_xlim) > 0 and max(new_xlim) < max(x_axis):
        
            ax1.set_xlim(new_xlim)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', zoom)

        # Connect the key press event
    def on_key(event):
        if event.key == 'left':
            scroll_left()
        elif event.key == 'right':
            scroll_right()

    def scroll_left():
        current_xlim = ax1.get_xlim()
        
        delta = (current_xlim[1] - current_xlim[0]) * 0.1  # Fine-tune the scrolling amount
        if current_xlim[0] - delta > 0:
            new_xlim = [current_xlim[0] - delta, current_xlim[1] - delta]
        
            ax1.set_xlim(new_xlim)
            fig.canvas.draw_idle()

    def scroll_right():
        current_xlim = ax1.get_xlim()
        delta = (current_xlim[1] - current_xlim[0]) * 0.1  # Fine-tune the scrolling amount
        if current_xlim[1] + delta < max(x_axis):
            new_xlim = [current_xlim[0] + delta, current_xlim[1] + delta]
            
            ax1.set_xlim(new_xlim)
            fig.canvas.draw_idle()
        

    # Ensure the connection of the key press event to the function
    fig.canvas.mpl_connect('key_press_event', on_key)
    # Add instructions to the plot
    instructions = (
        "Mouse Wheel: Zoom in/out\n"
        "Left/Right Arrow Keys: Scroll left/right"
    )

    # Add text annotation
    ax1.text(
        -0.1, 1.08, instructions, transform=ax1.transAxes,
        fontsize=10, verticalalignment='top'
    )


    def previous_mu(event):
        nonlocal current_index
        if current_index > 0:
            current_index -= 1
            plot_current_mu(current_index)
            update_slider()
            fig.canvas.draw_idle()

    def next_mu(event):
        nonlocal current_index
        if current_index < emgfile["NUMBER_OF_MUS"] - 1:
            current_index += 1
            plot_current_mu(current_index)
            update_slider()
            fig.canvas.draw_idle()

    # Create buttons for navigation
    ax_prev = plt.axes([0.12, 0.85, 0.12, 0.04], facecolor='lightgoldenrodyellow')
    btn_prev = Button(ax_prev, 'Previous')
    btn_prev.on_clicked(previous_mu)

    ax_next = plt.axes([0.87, 0.85, 0.12, 0.04], facecolor='lightgoldenrodyellow')
    btn_next = Button(ax_next, 'Next')
    btn_next.on_clicked(next_mu)

    ax1.set_xlim([min(x_axis),max(x_axis)])

    if showimmediately:
        plt.show()

    return fig