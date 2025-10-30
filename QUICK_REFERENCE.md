# DCMNet Multi-Batch Training - Quick Reference

## 🚀 One-Liners

### Train a Model
```bash
# Command line
python -m mmml.dcmnet.dcmnet.train_runner --name my_model --num-epochs 100

# Python
from mmml.dcmnet.dcmnet.train_runner import run_training
run_training(name="my_model", num_epochs=100)
```

### Analyze Results
```python
from mmml.dcmnet.dcmnet.analysis_multibatch import batch_analysis_summary
from mmml.dcmnet.dcmnet.analysis import create_model

model = create_model(n_dcm=2)
results = batch_analysis_summary("experiments/my_model", model, test_data)
```

### Resume Training
```bash
python -m mmml.dcmnet.dcmnet.train_runner \
    --name my_model \
    --restart-checkpoint experiments/my_model/checkpoints/checkpoint_latest.pkl
```

---

## 📋 Common Commands

### Basic Training
```bash
python -m mmml.dcmnet.dcmnet.train_runner \
    --name basic \
    --num-epochs 100 \
    --batch-size 1 \
    --learning-rate 1e-4
```

### With Gradient Accumulation
```bash
python -m mmml.dcmnet.dcmnet.train_runner \
    --name accumulated \
    --batch-size 1 \
    --gradient-accumulation-steps 8 \
    --num-epochs 100
```

### With LR Schedule
```bash
python -m mmml.dcmnet.dcmnet.train_runner \
    --name scheduled \
    --use-lr-schedule \
    --lr-schedule-type cosine \
    --warmup-epochs 10 \
    --num-epochs 200
```

### Large Model
```bash
python -m mmml.dcmnet.dcmnet.train_runner \
    --name large \
    --n-dcm 4 \
    --features 32 \
    --num-iterations 3 \
    --num-epochs 100
```

---

## 🐍 Python API Snippets

### Custom Configuration
```python
from mmml.dcmnet.dcmnet.training_config import ExperimentConfig, TrainingConfig, ModelConfig

config = ExperimentConfig(name="custom")
config.model = ModelConfig(n_dcm=3, features=32)
config.training = TrainingConfig(
    batch_size=2,
    gradient_accumulation_steps=4,
    use_lr_schedule=True
)
```

### Full Control Training
```python
from mmml.dcmnet.dcmnet.training_multibatch import train_model_multibatch

params, loss = train_model_multibatch(
    key=jax.random.PRNGKey(42),
    model=model,
    train_data=train_data,
    valid_data=valid_data,
    config=config
)
```

### Checkpoint Analysis
```python
from mmml.dcmnet.dcmnet.analysis_multibatch import analyze_checkpoint

analysis = analyze_checkpoint(
    checkpoint_path="experiments/my_model/checkpoints/checkpoint_best.pkl",
    model=model,
    test_data=test_data
)
print(f"MAE: {analysis['summary']['mae']:.6e}")
print(f"RMSE: {analysis['summary']['rmse']:.6e}")
```

### Compare Models
```python
from mmml.dcmnet.dcmnet.analysis_multibatch import compare_checkpoints

df = compare_checkpoints(
    checkpoint_paths=[
        "experiments/model1/checkpoints/checkpoint_best.pkl",
        "experiments/model2/checkpoints/checkpoint_best.pkl",
    ],
    model=model,
    test_data=test_data,
    labels=["Model 1", "Model 2"]
)
print(df)
```

---

## ⚙️ Configuration Templates

### Small Memory (<16GB)
```python
config.training = TrainingConfig(
    batch_size=1,
    gradient_accumulation_steps=8,
    use_grad_clip=True,
    grad_clip_norm=1.0
)
```

### Medium Memory (16-32GB)
```python
config.training = TrainingConfig(
    batch_size=4,
    gradient_accumulation_steps=4,
    use_grad_clip=True,
    grad_clip_norm=2.0
)
```

### Production Training
```python
config.training = TrainingConfig(
    num_epochs=500,
    use_lr_schedule=True,
    lr_schedule_type="cosine",
    warmup_epochs=20,
    save_every_n_epochs=10,
    use_ema=True
)
```

---

## 📂 File Locations

### Experiment Outputs
```
experiments/
└── my_model/
    ├── config.json              # Configuration
    ├── checkpoints/
    │   ├── checkpoint_best.pkl  # Best model
    │   ├── checkpoint_latest.pkl # For resuming
    │   └── checkpoint_epoch_*.pkl # Periodic
    └── analysis/               # Analysis outputs
        ├── best_analysis.json
        ├── predictions.csv
        └── training_history.csv
```

### Source Files
```
mmml/dcmnet/dcmnet/
├── training_config.py         # Configuration management
├── training_multibatch.py     # Multi-batch training
├── analysis_multibatch.py     # Analysis tools
└── train_runner.py            # Convenient interface
```

---

## 🔍 Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `name` | required | Experiment name |
| `num_epochs` | 100 | Number of epochs |
| `batch_size` | 1 | Physical batch size |
| `gradient_accumulation_steps` | 1 | Gradient accumulation |
| `learning_rate` | 1e-4 | Learning rate |
| `n_dcm` | 2 | Distributed multipoles |
| `features` | 16 | Model features |
| `esp_w` | 1.0 | ESP loss weight |
| `chg_w` | 0.01 | Charge loss weight |
| `use_lr_schedule` | False | Use LR schedule |
| `use_grad_clip` | True | Gradient clipping |
| `use_ema` | True | Exponential MA |

---

## 💡 Tips & Tricks

### Effective Batch Size
```
Effective Batch Size = batch_size × gradient_accumulation_steps
```

### Memory vs Speed Trade-off
- ↑ `batch_size` → More memory, faster
- ↑ `gradient_accumulation_steps` → Same memory, slower, better gradients

### Best Practices
1. Start with small model, scale up
2. Use LR schedule for long training
3. Enable gradient clipping for stability
4. Use EMA for better final performance
5. Save checkpoints frequently

### Debugging
```python
# Check configuration
config = ExperimentConfig.load("experiments/my_model/config.json")
print(config.to_dict())

# Load checkpoint
from mmml.dcmnet.dcmnet.analysis_multibatch import load_checkpoint
ckpt = load_checkpoint("experiments/my_model/checkpoints/checkpoint_latest.pkl")
print(f"Epoch: {ckpt['epoch']}")
print(f"Loss: {ckpt['metrics']['valid']['loss']}")
```

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `MULTIBATCH_TRAINING_GUIDE.md` | Complete user guide |
| `MULTI_BATCH_SUMMARY.md` | Implementation details |
| `SESSION_SUMMARY.md` | What was done |
| `QUICK_REFERENCE.md` | This file |
| `TRAINING_UPDATES.md` | Training enhancements |
| `RESHAPE_BUG_FIXES.md` | Bug fix details |
| `CLEANUP_SUMMARY.md` | Cleanup report |

---

## 🆘 Troubleshooting

### Out of Memory
```python
# Reduce batch size
--batch-size 1
```

### Unstable Training
```python
# Increase gradient clipping
--grad-clip-norm 0.5
```

### Slow Convergence
```python
# Use LR schedule
--use-lr-schedule --lr-schedule-type cosine
```

### Overfitting
```python
# Use gradient accumulation
--gradient-accumulation-steps 8
--use-ema
```

---

## 🎯 Examples

Full examples in:
```
examples/dcm/train_multibatch_example.py
```

Run with:
```bash
cd examples/dcm
python train_multibatch_example.py
```

---

## 📞 Getting Help

1. Check documentation: `MULTIBATCH_TRAINING_GUIDE.md`
2. Look at examples: `examples/dcm/train_multibatch_example.py`
3. Read docstrings: All functions are documented
4. Check error messages: Often include helpful hints

---

**Quick Reference v1.0** | *For detailed information, see MULTIBATCH_TRAINING_GUIDE.md*

