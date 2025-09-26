DCMNet API
==========

Core DCMNet modules for training distributed charge multipole networks.

Data Preparation
----------------

.. function:: mmml.dcmnet.dcmnet.data.prepare_datasets(key, num_train, num_valid, filename, clean=False, esp_mask=False, clip_esp=False, natoms=60)

   Prepare datasets for training and validation.

   :param key: Random key for dataset shuffling
   :param num_train: Number of training samples
   :param num_valid: Number of validation samples  
   :param filename: Filename(s) to load datasets from
   :param clean: Whether to filter failed calculations
   :param esp_mask: Whether to create ESP masks
   :param clip_esp: Whether to clip ESP to first 1000 points
   :param natoms: Maximum number of atoms per system
   :returns: Tuple of (train_data, valid_data) dictionaries

.. function:: mmml.dcmnet.dcmnet.data.prepare_batches(key, data, batch_size, include_id=False, data_keys=None, num_atoms=60, dst_idx=None, src_idx=None)

   Prepare batches for training.

   :param key: Random key for shuffling
   :param data: Dictionary containing the dataset
   :param batch_size: Size of each batch
   :param include_id: Whether to include ID in output
   :param data_keys: List of keys to include
   :param num_atoms: Number of atoms per system
   :param dst_idx: Destination indices for message passing
   :param src_idx: Source indices for message passing
   :returns: List of batch dictionaries

Training Functions
------------------

.. function:: mmml.dcmnet.dcmnet.training.train_model(key, model, train_data, valid_data, num_epochs, learning_rate, batch_size, writer, ndcm, esp_w=1.0, restart_params=None, ema_decay=0.999)

   Train DCMNet with ESP and monopole losses.

   :param key: Random key for training
   :param model: MessagePassingModel instance
   :param train_data: Training dataset
   :param valid_data: Validation dataset
   :param num_epochs: Number of training epochs
   :param learning_rate: Learning rate
   :param batch_size: Batch size
   :param writer: TensorBoard writer
   :param ndcm: Number of distributed multipoles
   :param esp_w: ESP loss weight
   :param restart_params: Parameters to restart from
   :param ema_decay: Exponential moving average decay
   :returns: Tuple of (final_params, final_valid_loss)

.. function:: mmml.dcmnet.dcmnet.training.train_model_dipo(key, model, train_data, valid_data, num_epochs, learning_rate, batch_size, writer, ndcm, esp_w=1.0, restart_params=None)

   Train DCMNet with dipole-augmented losses.

   :param key: Random key for training
   :param model: MessagePassingModel instance
   :param train_data: Training dataset (must include Dxyz, com, espMask)
   :param valid_data: Validation dataset
   :param num_epochs: Number of training epochs
   :param learning_rate: Learning rate
   :param batch_size: Batch size
   :param writer: TensorBoard writer
   :param ndcm: Number of distributed multipoles
   :param esp_w: ESP loss weight
   :param restart_params: Parameters to restart from
   :returns: Tuple of (final_params, final_valid_loss)

Model Architecture
-------------------

.. class:: mmml.dcmnet.dcmnet.modules.MessagePassingModel(features, max_degree, num_iterations, num_basis_functions, cutoff, n_dcm, include_pseudotensors=False)

   E(3)-equivariant message passing model for distributed multipoles.

   :param features: Number of features per atom
   :param max_degree: Maximum spherical harmonic degree
   :param num_iterations: Number of message passing iterations
   :param num_basis_functions: Number of radial basis functions
   :param cutoff: Distance cutoff for interactions
   :param n_dcm: Number of distributed multipoles per atom
   :param include_pseudotensors: Whether to include pseudotensors

Loss Functions
--------------

.. function:: mmml.dcmnet.dcmnet.loss.esp_mono_loss(dipo_prediction, mono_prediction, vdw_surface, esp_target, mono, ngrid, n_atoms, batch_size, esp_w, n_dcm)

   Combined ESP and monopole loss function.

   :param dipo_prediction: Predicted distributed dipoles
   :param mono_prediction: Predicted monopoles
   :param vdw_surface: Surface grid points
   :param esp_target: Target ESP values
   :param mono: Reference monopoles
   :param ngrid: Number of grid points per system
   :param n_atoms: Number of atoms per system
   :param batch_size: Batch size
   :param esp_w: ESP loss weight
   :param n_dcm: Number of distributed multipoles
   :returns: Total loss value

.. function:: mmml.dcmnet.dcmnet.loss.dipo_esp_mono_loss(dipo_prediction, mono_prediction, vdw_surface, esp_target, mono, Dxyz, com, espMask, n_atoms, batch_size, esp_w, n_dcm)

   Dipole-augmented ESP and monopole loss function.

   :param dipo_prediction: Predicted distributed dipoles
   :param mono_prediction: Predicted monopoles
   :param vdw_surface: Surface grid points
   :param esp_target: Target ESP values
   :param mono: Reference monopoles
   :param Dxyz: Reference dipole positions
   :param com: Center of mass coordinates
   :param espMask: ESP evaluation masks
   :param n_atoms: Number of atoms per system
   :param batch_size: Batch size
   :param esp_w: ESP loss weight
   :param n_dcm: Number of distributed multipoles
   :returns: Tuple of (esp_loss, mono_loss, dipole_loss)
