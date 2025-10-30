import ase
import matplotlib.pyplot as plt
import numpy as np
import optax
from ase.visualize.plot import plot_atoms
from jax import numpy as jnp
from rdkit.Chem import Draw
from scipy.spatial.distance import cdist

from dcmnet.loss import (
    esp_loss_eval,
    esp_loss_pots,
    esp_mono_loss_pots,
    get_predictions,
)
# NATOMS removed - now using dynamic shape inference
from dcmnet.multimodel import get_atoms_dcmol
from dcmnet.multipoles import plot_3d
from dcmnet.utils import apply_model, clip_colors, reshape_dipole


def infer_num_atoms(batch, batch_size):
    """Infer number of atoms from batch shape."""
    if "Z" in batch:
        return len(batch["Z"]) // batch_size
    elif "R" in batch:
        return len(batch["R"]) // batch_size // 3
    else:
        raise ValueError("Cannot infer num_atoms from batch")


# set the default color map to RWB
plt.set_cmap("bwr")


def evaluate_dc(
    batch, dipo, mono, batch_size, nDCM, plot=False, id=False
):

    esp_dc_pred = esp_mono_loss_pots(
        dipo, mono, batch["vdw_surface"][0], batch["mono"], batch_size, nDCM
    )

    mono_pred = esp_loss_pots(
        batch["R"],
        batch["mono"],
        batch["vdw_surface"],
        batch["mono"],
        batch_size,
    )

    non_zero = np.nonzero(batch["Z"])

    # Infer number of atoms from batch
    num_atoms = infer_num_atoms(batch, batch_size)
    
    esp_errors = []
    mono_errors = []
    xyzs = batch["R"].reshape(batch_size, num_atoms, 3)
    elems = batch["Z"].reshape(batch_size, num_atoms)
    monos_gt = batch["mono"].reshape(batch_size, num_atoms)
    monos_pred = mono.reshape(batch_size, num_atoms, nDCM)

    if id:
        from dcmnet.rdkit_utils import get_mol_from_id

        mols = get_mol_from_id(batch)
        images = [Draw.MolToImage(_) for _ in mols]

    for mbID in range(batch_size):

        xyz = xyzs[mbID]
        elem = elems[mbID]
        mono_gt = monos_gt[mbID]
        mono_pred_ = monos_pred[mbID]
        non_zero = np.nonzero(mono_gt)

        vdws = batch["vdw_surface"][mbID]

        idx_cut = batch["espMask"][0]
        loss1 = (
            esp_loss_eval(
                esp_dc_pred[mbID][idx_cut],
                batch["esp"][mbID][idx_cut],
                batch["n_grid"][mbID],
            )
            * 627.509
        )
        loss2 = (
            esp_loss_eval(
                mono_pred[mbID][idx_cut],
                batch["esp"][mbID][idx_cut],
                batch["n_grid"][mbID],
            )
            * 627.509
        )
        # print(mono.sum(axis=-1))
        esp_errors.append([loss1, loss2])
        loss = jnp.mean(
            abs(batch["mono"][non_zero] - mono.sum(axis=-1).flatten()[non_zero])
        )
        mono_errors.append(loss)

        if plot:

            fig = plt.figure(figsize=(12, 12))

            ax_scatter = fig.add_subplot(4, 4, 1)
            ax_scatter2 = fig.add_subplot(4, 4, 5)
            ax_scatter3 = fig.add_subplot(4, 4, 9)
            ax_rdkit = fig.add_subplot(4, 4, 4, frameon=False)
            ax1 = fig.add_subplot(4, 4, 3, projection="3d")
            ax2 = fig.add_subplot(4, 4, 7, projection="3d")
            ax4 = fig.add_subplot(4, 4, 8, projection="3d")
            axmol = fig.add_subplot(4, 4, 10, frameon=False)
            axmol3 = fig.add_subplot(4, 4, 2, frameon=False)
            axmol2 = fig.add_subplot(4, 4, 6, frameon=False)
            ax3 = fig.add_subplot(4, 4, 11, projection="3d")
            ax5 = fig.add_subplot(4, 4, 12, projection="3d")

            ax_scatter.scatter(
                mono_gt[non_zero],
                mono_pred_.sum(axis=-1).squeeze()[non_zero],
                c=mono_pred_.sum(axis=-1).squeeze()[non_zero],
                vmin=-1,
                vmax=1,
            )
            loss = jnp.mean(
                abs(batch["mono"][non_zero] - mono.sum(axis=-1).squeeze()[non_zero])
            )
            ax_scatter.set_title(f"MAE: {loss:.3f}")

            ax_scatter.plot([-1, 1], [-1, 1], c="k", alpha=0.5)
            ax_scatter.set_xlim(-1, 1)
            ax_scatter.set_ylim(-1, 1)
            ax_scatter.set_aspect("equal")
            ax_scatter.set_xlabel("$q_\mathrm{mono.}$ [$e$]")
            ax_scatter.set_ylabel("$q_\mathrm{dcmnet}$ [$e$]")

            ax_scatter2.scatter(
                batch["esp"][mbID][idx_cut],
                esp_dc_pred[mbID][idx_cut],
                alpha=0.9,
                s=0.1,
                color="k",
            )
            ax_scatter3.scatter(
                batch["esp"][mbID][idx_cut],
                mono_pred[mbID][idx_cut],
                alpha=0.9,
                s=0.1,
                color="k",
            )
            for ax in [ax_scatter2, ax_scatter3]:
                ax.set_xlabel("ESP$_\mathrm{DFT}$ [(kcal/mol)/$e$]")
                ax.set_xlim(-0.1, 0.1)
                ax.set_ylim(-0.1, 0.1)
                ax.plot([-1, 1], [-1, 1], c="k", alpha=0.5)
                ax.set_aspect("equal")
            ax_scatter2.set_ylabel("ESP$_\mathrm{dcmnet}$ [(kcal/mol)/$e$]")
            ax_scatter3.set_ylabel("ESP$_\mathrm{mono.}$ [(kcal/mol)/$e$]")

            if id:
                ax_rdkit.imshow(images[mbID])
            ax_rdkit.axis("off")

            s = ax1.scatter(
                *batch["vdw_surface"][mbID][idx_cut].T,
                c=clip_colors(batch["esp"][mbID][idx_cut]),
                vmin=-0.015,
                vmax=0.015,
            )
            if id:
                ax1.set_title(f"GT ({batch['id'][mbID]})")

            s = ax2.scatter(
                *batch["vdw_surface"][mbID][idx_cut].T,
                c=clip_colors(esp_dc_pred[mbID][idx_cut]),
                vmin=-0.015,
                vmax=0.015,
            )
            ax2.set_title(f"dcmnet: {loss1:.1f} (kcal/mol)/$e$")

            s = ax4.scatter(
                *batch["vdw_surface"][mbID][idx_cut].T,
                c=clip_colors(
                    esp_dc_pred[mbID][idx_cut]
                    - batch["esp"][mbID][idx_cut]
                ),
                vmin=-0.015,
                vmax=0.015,
            )

            elem = elem[non_zero]
            mono_gt = mono_gt[non_zero]
            xyz = xyz[non_zero]
            ROT = "45x,45y,-45z"
            vdw = batch["vdw_surface"] 
            no_dummy = batch["vdw_surface"] < 10000
            max_vals = np.min(vdw * no_dummy, axis=1)
            min_vals = np.max(vdw * no_dummy, axis=1)
            Rmax_vals = np.min(xyz, axis=0)
            Rmin_vals = np.max(xyz, axis=0)
            translate = - xyz.mean(axis=0) + (max_vals - min_vals)/2
            xyz = xyz + translate
            CELL = (max_vals - min_vals) * np.eye(3)
            
            atoms = ase.Atoms(
                numbers=elem,
                positions= xyz,
                cell = CELL
            )

            import dcmnet.utils
            d = dipo
            from matplotlib import cm
            from matplotlib.colors import Normalize

            norm = Normalize(vmin=-1, vmax=1)
            cmap = cm.get_cmap("bwr")
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

            pccolors = mappable.to_rgba(mono_gt.flatten()[: len(elem)])
            from ase.data.colors import jmol_colors

            atomcolors = [jmol_colors[_] for _ in elem]
            atomcolors_ = []
            for _ in atomcolors:
                atomcolors_.append(np.append(_, 0.015))
            dcmcolors = mappable.to_rgba(mono.flatten()[: len(elem) * nDCM])

            dcmol = ase.Atoms(
                ["X" if not _ else "He" for _ in mono.flatten()[: len(elem) * nDCM]],
                d[: len(elem) * nDCM] + translate,
                # cell = CELL,
            )

            plot_atoms(
                atoms, axmol, rotation=(ROT), colors=pccolors, scale=1
            )

            axmol.axis("off")
            plot_atoms(
                dcmol + atoms,
                axmol2,
                colors=list(dcmcolors) + list(atomcolors_),
                radii=[
                    (
                        0.1
                        if i < len(dcmcolors)
                        else ase.data.vdw_radii[elem[i - len(dcmcolors)]] / 2.5
                    )
                    for i, _ in enumerate(list(dcmcolors) + list(atomcolors_))
                ],
                rotation=(ROT),
                scale=5,
            )
            axmol2.axis("off")

            # combined atoms and dcm
            plot_atoms(
                atoms,
                axmol3,
                rotation=(ROT),
                colors=list(atomcolors),
                scale=1,
            )
            axmol3.axis("off")

            s = ax3.scatter(
                *batch["vdw_surface"][mbID][idx_cut].T,
                c=clip_colors(mono_pred[mbID][idx_cut]),
                vmin=-0.015,
                vmax=0.015,
            )
            ax3.set_title(f"mono.: {loss2:.1f} (kcal/mol)/$e$")

            s = ax5.scatter(
                *batch["vdw_surface"][mbID][idx_cut].T,
                c=clip_colors(
                    mono_pred[mbID][idx_cut]
                    - batch["esp"][mbID][idx_cut]
                ),
                vmin=-0.015,
                vmax=0.015,
            )

            for _ in [ax1, ax2, ax3, ax4, ax5]:
                # _.set_proj_type('ortho')
                _.set_xlim(-10, 10)
                _.set_xlabel("$x~[\mathrm{\AA}]$")
                _.set_ylim(-10, 10)
                _.set_ylabel("$y~[\mathrm{\AA}]$")
                _.set_zlim(-10, 10)
                _.set_zlabel("$z~[\mathrm{\AA}]$")

            # adjust white space
            plt.subplots_adjust(wspace=0.5, hspace=0.75)
            plt.tight_layout()
            if id:
                key = batch["id"][mbID]
            else:
                key = ""
            plt.savefig(
                f"/pchem-data/meuwly/boittier/home/jaxeq/figures/summary-{plot}-{key}.pdf",
                bbox_inches="tight",
            )
            plt.show()
            # plt.clf()
    if id:
        return esp_errors, mono_pred, mono_errors, batch["id"]
    else:
        return esp_errors, mono_pred, mono_errors, None


def plot_3d_combined(combined, batch, batch_size=1):
    xyz2 = combined[:, :3]
    q2 = combined[:, 3]
    i = 0
    nonzero = np.nonzero(batch["Z"].reshape(batch_size, NATOMS)[i])
    xyz = batch["R"].reshape(batch_size, NATOMS, 3)[i][nonzero]
    elem = batch["Z"].reshape(batch_size, NATOMS)[i][nonzero]

    from ase import Atoms
    from ase.visualize import view

    mol = Atoms(elem, xyz)
    V1 = view(mol, viewer="x3d")
    dcmol = Atoms(["X" if _ > 0 else "He" for _ in q2], xyz2)
    V2 = view(dcmol, viewer="x3d")
    combined = dcmol + mol
    V3 = view(combined, viewer="x3d")
    return V1, V2, V3


def plot_model(DCM2, params, batch, batch_size, nDCM, plot=True):
    mono_dc2, dipo_dc2 = apply_model(DCM2, params, batch, batch_size)

    esp_errors, mono_pred, _, _ = evaluate_dc(
        batch,
        dipo_dc2,
        mono_dc2,
        batch_size,
        nDCM,
        plot=plot,
        rcut0=3,
        rcut=4,
    )

    atoms, dcmol, grid, esp, esp_dc_pred, idx_cut = create_plots2(
        mono_dc2, dipo_dc2, batch, batch_size, nDCM
    )
    outDict = {
        "mono": mono_dc2,
        "dipo": dipo_dc2,
        "esp_errors": esp_errors,
        "atoms": atoms,
        "dcmol": dcmol,
        "grid": grid,
        "esp": esp,
        "esp_dc_pred": esp_dc_pred,
        "esp_mono_pred": mono_pred,
        "idx_cut": idx_cut,
    }
    return outDict


def plot_esp(esp, batch, batch_size, rcut=4.0, charges=None):
    """
    Plot ESP predictions and molecular structure with charges.
    
    Args:
        esp: Predicted ESP values
        batch: Batch data dictionary
        batch_size: Number of molecules in batch
        rcut: Cutoff distance for ESP visualization
        charges: Optional predicted charges array (batch_size, NATOMS) or (batch_size, NATOMS, nDCM)
    """
    # Infer number of atoms from batch
    num_atoms = infer_num_atoms(batch, batch_size)
    
    mbID = 0
    xyzs = batch["R"].reshape(batch_size, num_atoms, 3)
    elems = batch["Z"].reshape(batch_size, num_atoms)
    vdws = batch["vdw_surface"][mbID][: batch["n_grid"][mbID]]
    diff = xyzs[mbID][:, None, :] - vdws[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    min_d = np.min(r, axis=-2)
    wheremind = np.where(min_d < rcut, min_d, 0)
    idx_cut = np.nonzero(wheremind)[0]

    mono_pred = esp_loss_pots(
        batch["R"],
        batch["mono"],
        batch["vdw_surface"],
        batch["mono"],
        batch_size,
    )

    print(len(mono_pred[0][idx_cut]))

    loss_mono = optax.l2_loss(
        mono_pred[0][idx_cut] * 627.509, batch["esp"][0][idx_cut] * 627.509
    )
    loss_mono = np.mean(loss_mono * 2) ** 0.5
    loss_dc = optax.l2_loss(esp[idx_cut] * 627.509, batch["esp"][0][idx_cut] * 627.509)
    loss_dc = np.mean(loss_dc * 2) ** 0.5

    # Create figure with 6 subplots (added one for molecular structure)
    fig = plt.figure(figsize=(15, 6))

    # set white background
    fig.patch.set_facecolor("white")
    # whitebackground in 3d
    fig.patch.set_alpha(0.0)

    # Subplot 1: Molecular structure with charges
    ax0 = fig.add_subplot(161)
    
    # Create ASE Atoms object for the molecule
    coords = xyzs[mbID]
    atomic_numbers = elems[mbID]
    non_zero_mask = atomic_numbers > 0
    
    if np.any(non_zero_mask):
        atoms = ase.Atoms(
            numbers=atomic_numbers[non_zero_mask].astype(int),
            positions=coords[non_zero_mask]
        )
        
        # Plot atoms
        plot_atoms(atoms, ax0, radii=0.3, rotation=('0x,0y,0z'))
        
        # Get charges for visualization
        if charges is not None:
            charge_array = charges.reshape(batch_size, num_atoms, -1)[mbID]
            # Sum over DCM sites if multiple
            if charge_array.ndim > 1:
                atom_charges = np.sum(charge_array, axis=-1)
            else:
                atom_charges = charge_array
        else:
            # Use monopoles from batch as fallback
            atom_charges = batch["mono"].reshape(batch_size, num_atoms)[mbID]
        
        atom_charges = atom_charges[non_zero_mask]
        
        # Normalize charges for color mapping (same scale as ESP)
        charge_norm = plt.Normalize(vmin=-0.5, vmax=0.5)
        cmap = plt.cm.bwr
        
        # Add charge visualization as colored circles
        for i, (pos, charge) in enumerate(zip(atoms.positions[:, :2], atom_charges)):
            color = cmap(charge_norm(charge))
            circle = plt.Circle(pos, 0.4, color=color, alpha=0.6, zorder=10)
            ax0.add_patch(circle)
            # Add charge value as text
            ax0.text(pos[0], pos[1], f'{charge:.2f}', 
                    ha='center', va='center', fontsize=8, 
                    color='black', weight='bold', zorder=11)
        
        ax0.set_title('Molecule + Charges')
        ax0.set_aspect('equal')
        
        # Add colorbar for charges
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=charge_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax0, fraction=0.046, pad=0.04)
        cbar.set_label('Charge (e)', rotation=270, labelpad=15)
    
    # Subplots 2-6: ESP visualizations
    ax1 = fig.add_subplot(162, projection="3d")
    s = ax1.scatter(
        *batch["vdw_surface"][mbID][: batch["n_grid"][mbID]][idx_cut].T,
        c=clip_colors(batch["esp"][mbID][: batch["n_grid"][mbID]][idx_cut]),
        vmin=-0.015,
        vmax=0.015,
        cmap='bwr'
    )
    ax1.set_title(f"GT ESP {mbID}")

    ax2 = fig.add_subplot(163, projection="3d")
    s = ax2.scatter(
        *batch["vdw_surface"][mbID][: batch["n_grid"][mbID]][idx_cut].T,
        c=clip_colors(esp[idx_cut]),
        vmin=-0.015,
        vmax=0.015,
        cmap='bwr'
    )
    ax2.set_title(f'DC Pred\nRMSE: {loss_dc:.3f}')

    ax4 = fig.add_subplot(164, projection="3d")
    s = ax4.scatter(
        *batch["vdw_surface"][mbID][: batch["n_grid"][mbID]][idx_cut].T,
        c=clip_colors(
            esp[idx_cut] - batch["esp"][mbID][: batch["n_grid"][mbID]][idx_cut]
        ),
        vmin=-0.015,
        vmax=0.015,
        cmap='bwr'
    )
    ax4.set_title('DC Error')

    ax3 = fig.add_subplot(165, projection="3d")
    s = ax3.scatter(
        *batch["vdw_surface"][mbID][: batch["n_grid"][mbID]][idx_cut].T,
        c=clip_colors(mono_pred[mbID][: batch["n_grid"][mbID]][idx_cut]),
        vmin=-0.015,
        vmax=0.015,
        cmap='bwr'
    )
    ax3.set_title(f'Mono Pred\nRMSE: {loss_mono:.3f}')

    ax5 = fig.add_subplot(166, projection="3d")
    s = ax5.scatter(
        *batch["vdw_surface"][mbID][: batch["n_grid"][mbID]][idx_cut].T,
        c=clip_colors(
            mono_pred[mbID][: batch["n_grid"][mbID]][idx_cut]
            - batch["esp"][mbID][: batch["n_grid"][mbID]][idx_cut]
        ),
        vmin=-0.015,
        vmax=0.015,
        cmap='bwr'
    )
    ax5.set_title('Mono Error')

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-10, 10)
        # Remove axis labels for cleaner look
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
    
    plt.tight_layout()
    plt.show()
    return loss_dc, loss_mono


def plot_3d_combined(combined, batch, batch_size):
    xyz2 = combined[:, :3]
    q2 = combined[:, 3]
    i = 0
    nonzero = np.nonzero(batch["Z"].reshape(batch_size, NATOMS)[i])
    xyz = batch["R"].reshape(batch_size, NATOMS, 3)[i][nonzero]
    elem = batch["Z"].reshape(batch_size, NATOMS)[i][nonzero]

    from ase import Atoms
    from ase.visualize import view

    mol = Atoms(elem, xyz)
    V1 = view(mol, viewer="x3d")
    dcmol = Atoms(["X" if _ > 0 else "He" for _ in q2], xyz2)
    V2 = view(dcmol, viewer="x3d")
    combined = dcmol + mol
    V3 = view(combined, viewer="x3d")
    return V1, V2, V3


def create_plots2(mono_dc2, dipo_dc2, batch, batch_size, nDCM):
    esp_dc_pred, mono_pred = get_predictions(
        mono_dc2, dipo_dc2, batch, batch_size, nDCM
    )
    dipo_dc2 = reshape_dipole(dipo_dc2, nDCM)
    atoms, dcmol, end = get_atoms_dcmol(batch, mono_dc2, dipo_dc2, nDCM)

    grid = batch["vdw_surface"][0]
    # esp = esp_dc_pred[0]
    # esp = batch["esp"][0]
    esp = esp_dc_pred[0] - batch["esp"][0]

    print(
        "rmse:",
        jnp.mean(2 * optax.l2_loss(esp_dc_pred[0] * 627.503, batch["esp"][0] * 627.503))
        ** 0.5,
    )

    xyz = batch["R"][:end]

    cull_min = 2.5
    cull_max = 4.0
    grid_idx = np.where(np.all(cdist(grid, xyz) >= (cull_min - 1e-10), axis=-1))[0]
    print(grid_idx)
    grid_idx = jnp.asarray(grid_idx)
    grid = grid[grid_idx]
    esp = esp[grid_idx]
    grid_idx = np.where(np.all(cdist(grid, xyz) >= (cull_max - 1e-10), axis=-1))[0]
    grid_idx = [_ for _ in range(grid.shape[0]) if _ not in grid_idx]
    grid_idx = jnp.asarray(grid_idx)
    grid = grid[grid_idx]
    esp = esp[grid_idx]
    # try:
    #     display(get_rdkit(batch))
    # except:
    #     pass
    plot_3d(grid, esp, atoms=atoms + dcmol)
    return atoms, dcmol, grid, esp, esp_dc_pred[0], grid_idx
