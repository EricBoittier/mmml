# Distributed Charge Models predicted using Equivarient Neural Networks

## Introduction

## Requirements

e3x - Equivariant Neural Networks in Jax

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Unified Training API (dcmnet/training.py)

This project provides a **unified, flexible training API** for both default (ESP/Mono) and dipole training modes, as well as custom training configurations. The API is implemented in `dcmnet/training.py` and is designed for extensibility and ease of use.

### 1. Default Training (ESP/Mono)

```python
from dcmnet.training import train_model

params, val_loss = train_model(
    key=key,
    model=model,
    train_data=train_data,
    valid_data=valid_data,
    num_epochs=100,
    learning_rate=1e-3,
    batch_size=32,
    writer=writer,
    ndcm=2,
    esp_w=1.0,
    restart_params=None,
)
```
- **Logs:** Only total loss.
- **Optimizer:** Adam.
- **EMA:** Used for validation and checkpointing.

---

### 2. Dipole Training

```python
from dcmnet.training import train_model_dipo

params, val_loss = train_model_dipo(
    key=key,
    model=model,
    train_data=train_data,
    valid_data=valid_data,
    num_epochs=100,
    learning_rate=1e-3,
    batch_size=32,
    writer=writer,
    ndcm=2,
    esp_w=1.0,
    restart_params=None,
)
```
- **Logs:** Total loss, esp_l, mono_l, dipo_l (for both train and valid).
- **Optimizer:** AdamW with cosine schedule.
- **EMA:** Not used.
- **Gradient clipping:** Enabled.

---

### 3. Custom Training Mode

You can define your own loss, eval, optimizer, and logging functions, then use:

```python
from dcmnet.training import train_model_general

params, val_loss = train_model_general(
    key=key,
    model=model,
    train_data=train_data,
    valid_data=valid_data,
    num_epochs=100,
    learning_rate=1e-3,
    batch_size=32,
    writer=writer,
    ndcm=2,
    esp_w=1.0,
    loss_step_fn=my_train_step,
    eval_step_fn=my_eval_step,
    optimizer_fn=my_optimizer_fn,
    use_ema=True,
    ema_decay=0.99,
    use_grad_clip=True,
    grad_clip_norm=1.0,
    log_extra_metrics=my_logging_fn,
    save_best_params_with_ema=True,
)
```
- **You can mix and match features** (e.g., use EMA with dipole, or custom logging).

---

### How to Add a New Training Mode
1. **Write your loss and eval step functions** (see `train_step` and `train_step_dipo` for templates).
2. **Write an optimizer function** (see `create_adam_optimizer_with_exponential_decay`).
3. **Write a logging function** if you want to log extra metrics.
4. **Call `train_model_general`** with your custom functions.

---

### API Documentation

#### `train_model_general` Arguments
- **key, model, train_data, valid_data, num_epochs, learning_rate, batch_size, writer, ndcm, esp_w, restart_params**: as before.
- **loss_step_fn**: Function for a single training step. Should return `(params, opt_state, loss, *extras)`.
- **eval_step_fn**: Function for a single validation step. Should return `loss` or `(loss, *extras)`.
- **optimizer_fn**: Function that returns an Optax optimizer.
- **use_ema**: Whether to use exponential moving average for validation/checkpointing.
- **ema_decay**: EMA decay rate.
- **use_grad_clip**: Whether to use gradient clipping.
- **grad_clip_norm**: Norm for gradient clipping.
- **log_extra_metrics**: Function to log extra metrics (see `_log_extra_metrics_dipo` for an example).
- **save_best_params_with_ema**: If True, checkpoints EMA params; else, raw params.
- **extra_valid_args, extra_train_args**: Dicts for passing extra arguments to step functions.

---

### Extending Further
- Add new metrics to the logging function.
- Add new optimizer schedules.
- Add new loss functions or batch handling logic.

If you want a template for a new training mode, or want to see a diagram of the new structure, or have any other requests, see the code in `dcmnet/training.py` or ask the maintainers!
