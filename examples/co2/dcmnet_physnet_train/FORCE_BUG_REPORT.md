# Critical Bug Report: PhysNet Force Calculation

**Status**: 🚨 **CRITICAL BUG FOUND**

**Impact**: MD simulations explode due to incorrect forces

---

## 🐛 **The Bugs**

### **Bug #1: Missing Negative Sign** ✅ FIXED

**Location**: `mmml/physnetjax/physnetjax/models/model.py:959`

**Issue**: Forces weren't negated after autodiff  
```python
# Before (WRONG):
forces = gradient

# After (CORRECT):
forces = -gradient  # F = -dE/dr
```

**Status**: ✅ **FIXED** 

---

### **Bug #2: Forces 2× Too Large** ❌ **UNRESOLVED**

**Test Results**:
- Numerical forces (correct): ±0.074 eV/Å
- Analytical forces (from model): ±0.141 eV/Å  
- **Error: ~90%** (factor of 1.9)

**Possible causes**:
1. **Double counting** of pairwise interactions
2. Missing factor of 1/2 in energy formula
3. Edge list counting both i→j and j→i
4. Incorrect force aggregation in message passing

**Evidence**:
```bash
$ python test_physnet_forces.py --checkpoint ... --geometry ...

❌ PhysNet analytical forces don't match numerical!
Max difference: 0.066738 eV/Å
```

---

### **Bug #3: MD Explosions** ❌ **UNRESOLVED**

**Symptom**: Forces jump from 0.14 → 18.6 eV/Å for 0.05 Å displacement

**Test Results**:
```
Step 0: F_max = 0.14 eV/Å
Step 1: F_max = 18.6 eV/Å (after 0.05 Å move)
Temperature: 21.5 K → 318,536 K in 1 step!
```

**This is NOT physically reasonable!**

---

## 🧪 **How to Reproduce**

### Test 1: Force Validation
```bash
python test_physnet_forces.py \
  --checkpoint /path/to/checkpoint \
  --geometry optimized.xyz
```

**Expected**: Analytical = Numerical  
**Actual**: Analytical ≈ 2 × Numerical

### Test 2: NVE Stability
```bash
python test_nve.py \
  --checkpoint /path/to/checkpoint \
  --geometry optimized.xyz \
  --nsteps 10 \
  --timestep 0.05
```

**Expected**: Stable dynamics, E conserved  
**Actual**: Explosion at step 1-2

---

## 💡 **Root Cause Analysis**

### Hypothesis 1: Training Data Issue

**If training used the same buggy force calculation**:
- Model learned with incorrect force supervision
- Energy might be okay (loss converged)
- But force field is fundamentally wrong

**Check**: Were forces in training data computed with same PhysNet code?

### Hypothesis 2: Edge List Double Counting

**Observation**: Edge list includes both i→j AND j→i  
**Impact**: Each pairwise interaction counted twice  
**Expected**: Energy should have factor of 1/2 to compensate  
**Actual**: Might be missing, causing 2× force error

### Hypothesis 3: Autodiff Configuration

**Check**: Is `argnums=1` correct in `jax.value_and_grad`?
```python
energy_and_forces = jax.value_and_grad(self.energy, argnums=1, has_aux=True)
```

`argnums=1` means gradient w.r.t. **positions** (2nd argument after self).
This should be correct.

---

## 🔧 **Workarounds**

### **Option 1: Use ASE MD** (Slow but works)

ASE uses numerical forces internally:
```bash
python spectroscopy_suite.py \
  --use-ase-md \
  --molecule CO2 \
  --quick-analysis
```

**Pros**: Stable, correct  
**Cons**: 10-100× slower

### **Option 2: Retrain Model**

After fixing the bugs, retrain from scratch:
```bash
python trainer.py \
  --train-efd ... \
  --epochs 500 \
  --batch-size 100
```

**Pros**: Will have correct forces  
**Cons**: Takes time to retrain

### **Option 3: Use for Energy Only**

Use model for:
- ✅ Energy predictions
- ✅ ESP calculations  
- ✅ Charge predictions
- ❌ MD simulations (broken)

---

## 📊 **Test Scripts Created**

1. `test_physnet_forces.py` - Compare analytical vs numerical forces
2. `test_forces.py` - Test joint model forces
3. `test_nve.py` - Test NVE stability with diagnostics

---

## ✅ **Action Items**

### **Immediate**:
1. ✅ Fixed sign bug (forces = -gradient)
2. ❌ **Still need to fix 2× magnitude error**
3. ❌ **Still need to understand force explosion**

### **Medium term**:
1. Investigate PhysNet energy formula for missing 1/2 factor
2. Check edge list handling in message passing
3. Verify training data had correct forces

### **Long term**:
1. Retrain model with fixed force calculation
2. Add force validation to training pipeline
3. Add MD stability tests to CI/CD

---

##📝 **Notes for Debugging**

The force explosion (0.14 → 18.6 eV/Å) suggests the potential energy surface is **extremely steep** even near the optimized geometry. This could mean:

1. Model extrapolates poorly (trained on different geometries)
2. Numerical instabilities in basis functions
3. Charge predictions become unstable off-equilibrium
4. DCMNet contribution causes discontinuities

**Next steps**: Check if pure PhysNet (without DCMNet) is more stable.

---

**Created**: 2025-11-02  
**Status**: 🔴 **BLOCKING MD SIMULATIONS**

