import sys

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from sklearn.linear_model import LinearRegression





def plot_timeseries(data, columns, xcol):
    fig, axes = plt.subplots(len(columns) - 1, 1, sharex=True, figsize=(8, 12))
    fig.suptitle("Simulation Data Over Time")

    # columns = columns.to_numpy()

    for ax, col in zip(axes, columns[1:]):
        ax.plot(data[xcol].values, data[col].values, label=col)
        colname = "\n".join(col.split(" "))
        ax.set_ylabel(colname)
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel(xcol)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig("output.png")
    plt.show()



def plot_temperature_density(data, fitdf=None):
    data = data[data["Temperature (K)"] > 175.5]

    X = data["Temperature (K)"].values.reshape(-1, 1)
    y = data["Density (g/mL)"].values

    # fit polynomial degree 2
    fit = np.polyfit(X.ravel(), y, 2)
    print(fit)
    # scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, label="Data", alpha=0.01, color="gray")
    plt.plot(X, fit[0] * X**2 + fit[1] * X + fit[2], "--", label="Quadratic Fit", color="r")
    # plt.plot(X, fit.predict(X), label="Linear Fit")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Density (g/mL)")
    plt.legend()
    plt.grid(True)
    # diagonal line
    # plt.plot([0, 1], [0, 1], "--", color="k", transform=plt.gca().transAxes)
    plt.title("Temperature vs Density")
    plt.text(
        0.1,
        0.1,
        f"y = {fit[0]:.2e}x^2 + {fit[1]:.2e}x + {fit[2]:.2e}",
        transform=plt.gca().transAxes,
    )

    exp_df = None
    if fitdf is not None:
        exp_df = pd.read_json(fitdf) #"../../fitdata_meoh.json"
        dens_key = "Mass density, kg/m3"
        temp_key = "Temperature, K"
        pressure_key = "Pressure, kPa"
        exp_df = exp_df[exp_df[pressure_key] < 102]
        exp_df = exp_df[exp_df[pressure_key] > 101]
        # exp_df = exp_df[[dens_key, temp_key]]
        exp_df = exp_df.dropna()
        exp_df[dens_key] = exp_df[dens_key] / 1000
        # regress temperature vs density
        # linear fit to the data
        expX = exp_df[temp_key].values.reshape(-1, 1)
        y = exp_df[dens_key].values
        fit = LinearRegression().fit(expX, y)
        print(fit.coef_, fit.intercept_)
        plt.plot(X, fit.coef_ * X + fit.intercept_, "--", label="Linear Fit", color="g")

        plt.scatter(
            exp_df[temp_key],
            exp_df[dens_key],
            label="Experimental Data",
            color="g",
            zorder=10,
        )
    plt.show()
    return data, exp_df


def read_data(file):
    data = pd.read_csv(file)
    columns = list(data.columns)
    print(columns)
    xcol = columns[0]
    data = data.iloc[1000:]  # remove the first row
    return data, columns, xcol

def main(file, timestep=0.5, fitdf=None):
    data, columns, xcol = read_data(file)

    data["Time (ps)"] = data[xcol] * timestep * 1e-3  # convert to picoseconds
    xcol = "Time (ps)"
    plot_timeseries(data, columns, xcol)
    return plot_temperature_density(data, fitdf)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python plot.py <filename>")
        sys.exit(1)

    main(sys.argv[1])
