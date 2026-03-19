""""""
import jax

from mmml.models.physnetjax.physnetjax.data.read_h5 import prepare_h5_datasets
from mmml.models.physnetjax.physnetjax.models.model import EF
from mmml.models.physnetjax.physnetjax.training.training import train_model

key = jax.random.PRNGKey(40)

DATA = "/run/media/ericb/5416-320D/qcell_dimers.h5"

train_data, valid_data, natoms = prepare_h5_datasets(
    key,
    filepath=DATA,
    train_size=100000,
    valid_size=300,
    charge_filter=0.0,   # only neutral molecules
    natoms=34,
    verbose=True,
)

model = EF(features=32, num_iterations=2, max_degree=0, natoms=natoms, cutoff=6.0)

# from mmml.models.physnetjax.physnetjax.training.progressive import train_model_progressive
#ema_params, best_loss = train_model_progressive(
#    key, train_data, valid_data,
#    growth_stages=[
#        {"features": 16, "max_degree": 0},
#        {"features": 32, "max_degree": 0},
#        {"features": 32, "max_degree": 1},
#        {"features": 64, "max_degree": 1},
#    ],
#    base_model_kwargs={"natoms": natoms, "cutoff": 6.0, "zbl": True},
#    growth_patience=10,
#    num_epochs=500,
#    batch_size=300,
#    log_tb=False,
#    batch_method="default",
#    resume_stage=1,
#    resume_checkpoint="/home/ericb/mmml/mmml/physnetjax/ckpts/progressive-stage0-5bf8dd87-42a3-4232-8e79-330cd313bdef/",
#    data_keys=("R", "Z", "F", "E", "N", "D", "dst_idx", "src_idx", "batch_segments"),
#)


train_model(
    key, model, train_data, valid_data,
    num_epochs=1000,
    restart=None, #nb_dir / "ACO-b4f39bb9-8ca7-485e-bf51-2e5236e51b56",
    batch_size=248,
    num_atoms=natoms,
    learning_rate=0.0001,
    batch_method="default",
    data_keys=("R", "Z", "F", "E", "N", "D", "dst_idx", "src_idx", "batch_segments"),
    log_tb=False,
)
