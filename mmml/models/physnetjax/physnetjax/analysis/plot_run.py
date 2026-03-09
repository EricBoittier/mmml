import matplotlib.pyplot as plt
import seaborn as sns

from physnetjax.utils.pretty_printer import get_panel


def get_acp_plot(base_df, title="plot", log=False):
    if log:
        base_df = np.log(base_df)
    return get_panel(base_df, title)


def plot_run(base_df, ax, hue, label, log=False):
    if len(base_df) == 0:
        raise ValueError("Empty DataFrame provided")

    from physnetjax.utils.pretty_printer import get_acp_plot

    get_acp_plot(
        base_df,
        ["train_loss", "valid_loss"],
        title="Log10 Loss",
        log=True,
        color="blue",
    )
    get_acp_plot(
        base_df,
        ["train_energy_mae", "valid_energy_mae"],
        title="Log10 Energy MAE",
        log=True,
        color="red",
    )
    get_acp_plot(
        base_df,
        ["train_forces_mae", "valid_forces_mae"],
        title="Log10 Forces MAE",
        log=True,
        color="green",
    )
    get_acp_plot(base_df, ["lr"], title="Learning Rate", log=False, color="orange")

    if len(base_df) > 1000:
        # plot only the relevant data
        base_df = base_df[:-10:100] + base_df[-10:]
    elif len(base_df) > 10_000:
        base_df = base_df[:-10:1000] + base_df[-10:]
    else:
        pass
    # base_df = base_df.to_pandas()
    # Define all the metrics to plot
    metrics = [
        "train_loss",
        "valid_loss",
        "train_energy_mae",
        "valid_energy_mae",
        "train_forces_mae",
        "valid_forces_mae",
        # "train_dipole_mae", "valid_dipole_mae",
        "lr",
    ]

    # Plot each metric
    for i, ycol in enumerate(metrics):
        col = i % 2
        row = i // 2
        line = sns.lineplot(
            data=base_df,
            x="epoch",
            y=ycol,
            color=sns.color_palette("Set2", 34)[hue],
            ax=ax[row][col],
            label=label,
        )
        ax[row][col].legend()
        lines, labels = [], []
        # Capture lines and labels for the shared legend
        for line_obj in line.get_lines():
            lines.append(line_obj)
        labels.append(i)

        # Apply shared settings
        #
        # ax[row][col].set_xlim(1000)
        if ycol != "lr":
            ymin = base_df[ycol].min() * 0.5
            ymax = base_df[ycol].median()
            std = base_df[ycol].std()
            if (
                isinstance(ymin, float)
                and isinstance(ymax, float)
                and isinstance(std, float)
            ):
                ax[row][col].set_ylim(ymin, ymax + std)
            elif isinstance(ymin, float) and isinstance(ymax, float):
                ax[row][col].set_ylim(ymin, ymax)
            elif isinstance(ymax, float) and isinstance(std, float):
                ax[row][col].set_ylim(0.0, ymax + std)
            else:
                # do nothing
                pass

        if log:
            ax[row][col].set_yscale("log")
        ax[row][col].set_xlabel("Epoch")
        ax[row][col].set_ylabel(ycol)
        ax[row][col].get_legend().remove()  # Remove legend from the main plot

    # Adjust the legend on the separate axis
    handles, labels = ax[row][col].get_legend_handles_labels()
    ax[-1][-1].legend(handles=handles, labels=labels, loc="center", title="Metrics")
    ax[-1][-1].axis("off")  # Turn off axis for the legend space

    # plt.tight_layout()
    # plt.show()
    return ax


#
# import altair as alt
#
# columns = ['valid_energy_mae', 'valid_forces_mae', 'train_energy_mae',
#            'train_forces_mae', 'train_loss', 'valid_loss', 'lr', 'batch_size',
#            'energy_w', 'charges_w', 'dipole_w', 'forces_w', 'epoch',]
#
#
# (
#     alt.Chart(base_df).mark_point(tooltip=True).encode(
#         x="epoch",
#         y="valid_forces_mae",
#         # color="species",
#     )
#     .properties(width=500)
#     .configure_scale(zero=False)
# )
#
# import polars as pl
# import polars.selectors as cs
#
# # Perform unpivoting
# df = base_df.melt(
#     id_vars=["epoch"],  # Columns to keep as identifiers
#     value_vars=columns,  # Columns to unpivot
#     variable_name="category",  # Optional renaming
#     value_name="y",  # Optional renaming
# )
#
# print(df)
#
# scaled_df = df.with_columns(
#     [
#         ((pl.col(col) - pl.col(col).min()) / (pl.col(col).max() - pl.col(col).min()))
#         .clip(0, 0.10)  # Apply clipping to the scaled values
#         .alias(col)        for col in df.select(cs.numeric()).columns if col != "epoch"
#     ]
# )
#
# print(scaled_df)
# cs.numeric()
#
# scaled_df = df


if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path

    import polars as pl

    from physnetjax.directories import LOGS_PATH
    from physnetjax.logger.tensorboard_interface import process_tensorboard_logs

    parser = ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    args = parser.parse_args()
    logs_path = args.logs
    key = logs_path.parent.name
    df = process_tensorboard_logs(logs_path)

    # pretty print polars dataframe with rich
    from rich.console import Console

    console = Console()
    console.print(df)

    fig, ax = plt.subplots(5, 2, figsize=(12, 12))
    plot_run(df, ax, 1, key, log=True)
    # save the plot
    save_path = LOGS_PATH / key / "tf_logs.png"
    # make the directory if it does not exist
    save_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path, bbox_inches="tight")
    # save the dataframe as a csv
    df.write_csv(LOGS_PATH / key / "tf_logs.csv")
