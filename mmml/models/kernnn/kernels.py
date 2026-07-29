"""JAX 1D kernel functions for KerNN descriptors.

Port of ``scripts/kernn/utils/kernels.py``. Elementwise in the last dimension.
``xi`` is the reference (usually min-energy) distance vector.
"""

from __future__ import annotations

import jax.numpy as jnp


def get_1d_kernels_k20(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (2.0 / xl - 2.0 / 3.0 * xs / xl**2)


def get_1d_kernels_k21(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (2.0 / (3.0 * xl**2) - 1.0 / 3.0 * xs / xl**3)


def get_1d_kernels_k22(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (1.0 / (3.0 * xl**3) - 1.0 / 5.0 * xs / xl**4)


def get_1d_kernels_k23(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (1.0 / (5.0 * xl**4) - 2.0 / 15.0 * xs / xl**5)


def get_1d_kernels_k24(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (2.0 / (15.0 * xl**5) - 2.0 / 21.0 * xs / xl**6)


def get_1d_kernels_k25(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (2.0 / (21.0 * xl**6) - 1.0 / 14.0 * xs / xl**7)


def get_1d_kernels_k26(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (1.0 / (14.0 * xl**7) - 1.0 / 18.0 * xs / xl**8)


def get_1d_kernels_k30(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (
        3.0 / xl - 3.0 / 2.0 * xs / xl**2 + 3.0 / 10.0 * xs**2 / xl**3
    )


def get_1d_kernels_k31(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (
        3.0 / (4.0 * xl**2)
        - 3.0 / 5.0 * xs / xl**3
        + 3.0 / 20.0 * xs**2 / xl**4
    )


def get_1d_kernels_k32(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (
        3.0 / (10.0 * xl**3)
        - 3.0 / 10.0 * xs / xl**4
        + 3.0 / 35.0 * xs**2 / xl**5
    )


def get_1d_kernels_k33(x, xi, scale=1.0):
    """1D k33 kernel used by the H2CO KerNN models."""
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (
        3.0 / (20.0 * xl**4)
        - 6.0 / 35.0 * xs / xl**5
        + 3.0 / 56.0 * xs**2 / xl**6
    )


def get_1d_kernels_k34(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (
        3.0 / (35.0 * xl**5)
        - 3.0 / 28.0 * xs / xl**6
        + 1.0 / 28.0 * xs**2 / xl**7
    )


def get_1d_kernels_k35(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (
        3.0 / (56.0 * xl**6)
        - 1.0 / 14.0 * xs / xl**7
        + 1.0 / 40.0 * xs**2 / xl**8
    )


def get_1d_kernels_k36(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (
        1.0 / (28.0 * xl**7)
        - 1.0 / 20.0 * xs / xl**8
        + 1.0 / 55.0 * xs**2 / xl**9
    )


KERNEL_FNS = {
    "k20": get_1d_kernels_k20,
    "k21": get_1d_kernels_k21,
    "k22": get_1d_kernels_k22,
    "k23": get_1d_kernels_k23,
    "k24": get_1d_kernels_k24,
    "k25": get_1d_kernels_k25,
    "k26": get_1d_kernels_k26,
    "k30": get_1d_kernels_k30,
    "k31": get_1d_kernels_k31,
    "k32": get_1d_kernels_k32,
    "k33": get_1d_kernels_k33,
    "k34": get_1d_kernels_k34,
    "k35": get_1d_kernels_k35,
    "k36": get_1d_kernels_k36,
}

# Catalog for CLI tables: xl=max(x,xi), xs=min(x,xi); optional scale multiplier.
KERNEL_INFO: dict[str, dict[str, str]] = {
    "k20": {
        "family": "k2*",
        "formula": "scale*(2/xl − 2/3·xs/xl²)",
    },
    "k21": {
        "family": "k2*",
        "formula": "scale*(2/(3 xl²) − 1/3·xs/xl³)",
    },
    "k22": {
        "family": "k2*",
        "formula": "scale*(1/(3 xl³) − 1/5·xs/xl⁴)",
    },
    "k23": {
        "family": "k2*",
        "formula": "scale*(1/(5 xl⁴) − 2/15·xs/xl⁵)",
    },
    "k24": {
        "family": "k2*",
        "formula": "scale*(2/(15 xl⁵) − 2/21·xs/xl⁶)",
    },
    "k25": {
        "family": "k2*",
        "formula": "scale*(2/(21 xl⁶) − 1/14·xs/xl⁷)",
    },
    "k26": {
        "family": "k2*",
        "formula": "scale*(1/(14 xl⁷) − 1/18·xs/xl⁸)",
    },
    "k30": {
        "family": "k3*",
        "formula": "scale*(3/xl − 3/2·xs/xl² + 3/10·xs²/xl³)",
    },
    "k31": {
        "family": "k3*",
        "formula": "scale*(3/(4 xl²) − 3/5·xs/xl³ + 3/20·xs²/xl⁴)",
    },
    "k32": {
        "family": "k3*",
        "formula": "scale*(3/(10 xl³) − 3/10·xs/xl⁴ + 3/35·xs²/xl⁵)",
    },
    "k33": {
        "family": "k3*",
        "formula": "scale*(3/(20 xl⁴) − 6/35·xs/xl⁵ + 3/56·xs²/xl⁶)",
    },
    "k34": {
        "family": "k3*",
        "formula": "scale*(3/(35 xl⁵) − 3/28·xs/xl⁶ + 1/28·xs²/xl⁷)",
    },
    "k35": {
        "family": "k3*",
        "formula": "scale*(3/(56 xl⁶) − 1/14·xs/xl⁷ + 1/40·xs²/xl⁸)",
    },
    "k36": {
        "family": "k3*",
        "formula": "scale*(1/(28 xl⁷) − 1/20·xs/xl⁸ + 1/55·xs²/xl⁹)",
    },
}

DEFAULT_KERNEL = "k33"


def list_kernel_rows(*, selected: str | None = None) -> list[dict[str, str]]:
    """Return rows for a kernel catalog table."""
    rows = []
    for name in sorted(KERNEL_FNS):
        info = KERNEL_INFO.get(name, {"family": "?", "formula": "(see source)"})
        mark = "← default" if name == DEFAULT_KERNEL else ""
        if selected and name == selected:
            mark = (mark + " · selected").strip(" ·") if mark else "← selected"
        rows.append(
            {
                "name": name,
                "family": info["family"],
                "formula": info["formula"],
                "note": mark,
            }
        )
    return rows


def print_kernel_table(*, selected: str | None = None, title: str | None = None) -> None:
    """Print all registered 1D KerNN kernels (Rich table when available)."""
    rows = list_kernel_rows(selected=selected)
    hdr = title or "KerNN 1D kernel functions"
    note = "xl = max(x, xi),  xs = min(x, xi),  xi = reference (min-energy) distances"
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title=hdr, show_header=True, header_style="bold")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Family", no_wrap=True)
        table.add_column("Formula")
        table.add_column("Note", style="green")
        for row in rows:
            table.add_row(row["name"], row["family"], row["formula"], row["note"])
        console = Console()
        console.print(table)
        console.print(f"[dim]{note}[/dim]")
    except Exception:
        print(hdr)
        print("-" * len(hdr))
        print(f"{'name':<6} {'family':<6} {'note':<14} formula")
        for row in rows:
            print(
                f"{row['name']:<6} {row['family']:<6} {row['note']:<14} {row['formula']}"
            )
        print(note)
