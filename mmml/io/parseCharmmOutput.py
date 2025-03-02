import os
import matplotlib.pyplot as plt
import polars as pl
import numpy as np

kinetic_color = "#B80000"
spatial_color = "#005384"
potential_color = "#004D3D"
state_color = "#A94A8C"
observable_color = "#003B5C"
count_color = "#B67B00"
#DYNA DYN: Step         Time      TOTEner        TOTKe       ENERgy  TEMPerature
#DYNA PROP:             GRMS      HFCTote        HFCKe       EHFCor        VIRKe
#DYNA INTERN:          BONDs       ANGLes       UREY-b    DIHEdrals    IMPRopers
#DYNA EXTERN:        VDWaals         ELEC       HBONds          ASP         USER
#DYNA IMAGES:        IMNBvdw       IMELec       IMHBnd       RXNField    EXTElec
#DYNA PRESS:            VIRE         VIRI       PRESSE       PRESSI       VOLUme
#DYNA>        0      0.00000 -61671.83101    283.74974 -61955.58076    124.11071
#DYNA PRESS>       418.64436   -413.41713  -3099.63460  -1660.34713   9261.00000

def read_dyna_line(line: str):
    dyna = None
    step = None
    time = None
    total_energy = None
    total_kinetic_energy = None
    energy = None
    temperature = None
    
    if line.startswith("DYNA"):
        dyna = True
    else:
        raise ValueError(f"Line {line} does not start with DYNA")

    step = int(line[5:18])
    time = float(line[19:28])
    total_energy = float(line[28:44])
    total_kinetic_energy = float(line[44:54])
    energy = float(line[54:64])
    temperature = float(line[69:])
    output = {
        "dyna": dyna,
        "step": step,
        "time": time,
        "total_energy": total_energy,
        "total_kinetic_energy": total_kinetic_energy,
        "energy": energy,
        "temperature": temperature
    }
    return output 

def read_open_line(line: str):
    pass


def read_press_line(line: str):
    vire = None
    viri = None
    press_e = None
    press_i = None
    volume = None
    if line.startswith("DYNA PRESS"):
        vire = float(line[11:28])
        viri = float(line[30:40])
        press_e = float(line[30:42])
        press_i = float(line[42:55])
        volume = float(line[68:80])
    output = {
        "vire": vire,
        "viri": viri,
        "press_e": press_e,
        "press_i": press_i,
        "volume": volume
    }
    return output




import matplotlib.pyplot as plt
def plot_simulation_overview(dyna_df, press_df, subfig=None):
    # plot of temperature, virials and volume over time

    if subfig is None:
        subfig, axs = plt.subplots(7, 1, figsize=(10, 10), sharex=True)
    else:
        # Create a 7x1 grid of axes within the subfigure
        axs = subfig.subplots(7, 1, sharex=True)
        subfig.subplots_adjust(hspace=0.1)

    axs[0].plot(dyna_df["temperature"], alpha=0.5, color=kinetic_color)
    axs[0].axhline(dyna_df["temperature"].mean(), color=kinetic_color, linestyle="--")
    
    axs[1].plot(press_df["viri"], alpha=0.25, color=state_color)
    axs[1].axhline(press_df["viri"].mean(), color=state_color, linestyle="--")
    axs[1].plot(press_df["vire"], alpha=0.25, color=spatial_color)
    axs[1].axhline(press_df["vire"].mean(), color=spatial_color, linestyle="--")
    axs[1].plot(press_df["vire"] + press_df["viri"], alpha=0.5, color="k", linestyle="--")
    axs[1].axhline((press_df["vire"] + press_df["viri"]).mean(), color="k", linestyle="--")   
    
    axs[2].plot(press_df["press_e"], alpha=0.5, color=spatial_color)
    axs[2].axhline(press_df["press_e"].mean(), color=spatial_color, linestyle="--")
    axs[2].plot(press_df["press_i"], alpha=0.5, color=state_color)
    axs[2].axhline(press_df["press_i"].mean(), color=state_color, linestyle="--")
    axs[2].plot(press_df["press_e"] + press_df["press_i"], alpha=0.5, color=observable_color, linestyle="--")
    axs[2].axhline((press_df["press_e"] + press_df["press_i"]).mean(), color=observable_color, linestyle="--")
    
    axs[3].plot(press_df["volume"], alpha=0.5, color=spatial_color)
    axs[3].axhline(press_df["volume"].mean(), color=spatial_color, linestyle="--")
    
    axs[4].plot(dyna_df["total_energy"], alpha=0.5, color="k")
    axs[4].axhline(dyna_df["total_energy"].mean(), color="k", linestyle="--")
    
    axs[5].plot(dyna_df["total_kinetic_energy"], alpha=0.5, color=kinetic_color)
    axs[5].axhline(dyna_df["total_kinetic_energy"].mean(), color=kinetic_color, linestyle="--")
    
    axs[6].plot(dyna_df["energy"], alpha=0.5, color=potential_color)
    axs[6].axhline(dyna_df["energy"].mean(), color=potential_color, linestyle="--")
    
    axs[0].set_ylabel("$T$")
    axs[1].set_ylabel("Virial")
    axs[2].set_ylabel("Pressure")
    axs[3].set_ylabel("Volume")
    axs[4].set_ylabel("$E$")
    axs[5].set_ylabel("$E_{\\rm kinetic}$")
    axs[6].set_ylabel("$E_{\\rm potential}$")

    max_time = dyna_df["time"].abs().max()
    current_xticks = axs[-1].get_xticks()
    print(current_xticks)
    max_x_tick = max(current_xticks)
    new_xticks = ["{}".format(int(max_time * x/max_x_tick)) for x in current_xticks]
    print(new_xticks)
    axs[-1].set_xticklabels(new_xticks)

    if fig is None:
        plt.savefig("simulation_overview.png", bbox_inches="tight")
    return fig

def plot_distribution(data, column, ax, color, shift=0.0):
    # If data is a DataFrame, get the column, otherwise use the data directly
    values = data[column] if isinstance(data, pl.DataFrame) else data
    
    ax.hist(values, bins=100, density=True, alpha=0.2, linewidth=1, edgecolor="k", color=color)
    # fit a gaussian to the data
    from scipy.stats import norm
    mu, std = norm.fit(values)
    x = np.linspace(values.min(), values.max(), 1000)
    p = norm.pdf(x, mu, std)
    ax.set_title(column)
    ax.plot(x, p, 'k', linewidth=1, linestyle="--", alpha=0.5)
    ax.text(0.01, 0.95+shift, f"$\mu = {mu:.0f}$\n $\sigma = {std:.0f}$", transform=ax.transAxes, ha="left", va="top")
    ax.set_xlabel(column)
    ax.set_ylabel("Density")

def plot_simulation_distribution_overview(dyna_df, press_df, subfig=None):
    if subfig is None:
        fig, axs = plt.subplots(3, 3, figsize=(10, 10), sharex=False, sharey=False)
        fig.tight_layout()
        # increase width and height space between subplots
        fig.subplots_adjust(wspace=0.3, hspace=0.5)
    else:
        # Create a 3x3 grid of axes within the subfigure
        axs = subfig.subplots(3, 3, sharex=False, sharey=False)
        subfig.subplots_adjust(wspace=0.3, hspace=1)

    plot_distribution(dyna_df, "temperature", axs[0, 0], color=kinetic_color)

    plot_distribution(dyna_df, "total_energy", axs[1, 1], color=spatial_color)

    plot_distribution(dyna_df, "total_kinetic_energy", axs[0, 1], color=potential_color)

    plot_distribution(dyna_df, "energy", axs[1, 0], color=state_color)

    plot_distribution(press_df, "vire", axs[2, 0], color=spatial_color)
    plot_distribution(press_df, "viri", axs[2, 0], color=state_color, shift=-0.2)
    plot_distribution(press_df["vire"] + press_df["viri"], "viri", axs[2, 0], color=state_color, shift=-0.4)

    plot_distribution(press_df, "press_e", axs[2, 1], color=spatial_color)
    plot_distribution(press_df, "press_i", axs[2, 1], color=observable_color, shift=0.2)

    plot_distribution(press_df, "volume", axs[2, 2], color=spatial_color)

    # subsample the data into blocks of 25% and plot the distribution of the mean of each block
    for i in range(0, 4):
        block = dyna_df.slice(i*len(dyna_df)//4, (i+1)*len(dyna_df)//4)
        # plot_distribution(block, "temperature", axs[0, 2], color=kinetic_color)
        # plot_distribution(block, "total_energy", axs[1, 2], color=spatial_color)

    N = 4 
    for i in range(0, N):
        block = press_df.slice(i*len(press_df)//N, (i+1)*len(press_df)//N)
        print(block.describe())
        plot_distribution(block/block["volume"].mean(), "volume", axs[1, 2], color=spatial_color, shift=-(1/N)*i)

    axs[0, 2].set_axis_off()

    #if subfig is None:
    #    plt.savefig("simulation_distribution_overview.png", bbox_inches="tight")

    return subfig


if __name__ == "__main__":
    test_dyna_line = "DYNA PRESS>       418.64436   -413.41713  -3099.63460  -1660.34713   9261.00000"
    print(read_press_line(test_dyna_line))
    test_dyna_line = "DYNA>        0      0.00000 -61671.83101    283.74974 -61955.58076    124.11071" 
    print(read_dyna_line(test_dyna_line))

    from pathlib import Path
    current_dir = Path(__file__).parent
    dyna_file = current_dir / ".." / ".." / "testdata" / "DYNA1"
    press_file = current_dir / ".." / ".." / "testdata" / "PRESS1"

    dyna_data = []
    press_data = []

    for line in open(dyna_file).readlines():
        if line.startswith("DYNA"):
            dyna_data.append(read_dyna_line(line))
    
    for line in open(press_file).readlines():   
        if line.startswith("DYNA PRESS"):
            press_data.append(read_press_line(line))

    dyna_df = pl.DataFrame(dyna_data)
    press_df = pl.DataFrame(press_data)

    n_steps = len(dyna_df)
    n_press = len(press_df)

    #dyna_df = dyna_df.unique(subset=["step", "time"])
    #press_df = press_df.unique()

    print(dyna_df)
    print(press_df)

    # describe the data
    print(dyna_df.describe())
    print(press_df.describe())

    # exclude the first 1000 frames
    dyna_df = dyna_df.slice(1000, None)
    press_df = press_df.slice(1000, None)

    fig = plt.figure(figsize=(20, 10))
    subfigs = fig.subfigures(2, 1, wspace=0.0, hspace=0.0)
    fig = plot_simulation_distribution_overview(dyna_df, press_df, subfigs[0])
    fig = plot_simulation_overview(dyna_df, press_df, subfigs[1])
    plt.savefig("simulation_overview.png", bbox_inches="tight")