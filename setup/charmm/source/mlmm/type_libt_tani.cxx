#if KEY_MLMM==1 && KEY_MLPTORCH==1
#ifdef WITH_TORCH
// ============================================================================
// type_libt_tani.cxx
// ============================================================================
//
// MODEL-SPECIFIC TorchScript bridge for TorchANI/TANI.
//
// ----------------------------------------------------------------------------
// CURRENT FORTRAN ABI MATCHED HERE
// ----------------------------------------------------------------------------
//
// Setup:
//
//   charmm_tani_internal_setup(
//       int tani_in_use,
//       int tani_use_gpu,
//       const char* tani_pt_name,
//       int tani_pt_nml,
//       const int* tani_in_mlidx,
//       const int* tani_in_mlZid,
//       const int* tani_in_mlmaskid,
//       int tani_in_natoms,
//       int* tani_out_setup_err)
//
// Force:
//
//   charmm_tani_internal_force(
//       double* E_tani_c,
//       const double* ml_x_c,
//       const double* ml_y_c,
//       const double* ml_z_c,
//       double* ml_dx_c,
//       double* ml_dy_c,
//       double* ml_dz_c)
//
// ----------------------------------------------------------------------------
//
//   1) build compact ML coordinates
//   2) create atom type tensor from mlZidx
//   3) create tuple input: (atom_types, ml_coords)
//   4) model.forward(inputs)
//   5) output must be tuple-like, energy is tuple element 1
//   6) use torch::autograd::grad on ml_coords
//   7) gradients and energy are in Hartree-style units
//   8) convert to kcal/mol-style engine units on return
//
//  This file preserves that LOGIC FLOW and DATATYPE EXPECTATIONS,
//
// - tuple input {atom_types, ml_coords} 
// - energy taken from tuple element 1 and grad from autograd 
// - Hartree -> kcal/mol conversion in old scatter kernel 
//
// ----------------------------------------------------------------------------
// UNIT CONVENTION
// ----------------------------------------------------------------------------
//
//
//   energy   : Hartree -> kcal/mol
//   gradient : Hartree/Angstrom -> kcal/mol/Angstrom
//
// Numerically, that is the same Hartree->kcal/mol factor, since Angstrom
// remains Angstrom.
//
//
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/autograd.h>

#if KEY_CUDA==1
#include <torch/cuda.h>
#endif

namespace {

// Same conversion intent as old tani.cu:
//   TORCH_E_HARTREE_TO_KCALMOL
//   TORCH_F_HARTREE_TO_KCALMOL
//
// If your project uses a slightly different exact constant elsewhere,
// replace this with that project-wide constant.
constexpr double ENERGY_SCALE = 627.5094740631; // Hartree -> kcal/mol

// ============================================================================
// TANI CACHED STATE
// ============================================================================
//
// Everything here is fixed after setup and reused in force calls.
//
struct TaniState {
    bool initialized = false;

    // Model path for debugging
    std::string model_path;

    // Device configuration
    bool use_gpu = false;
    int  gpu_id  = -1;
    torch::Device device = torch::kCPU;

    // Cached sizes
    int n_ml   = 0;   // compact ML atom count
    int natoms = 0;   // total atom count in full system

    // Cached setup arrays
    //
    // ml_idx  : compact ML index -> global atom index
    // ml_zid  : atomic number for each compact ML atom
    // ml_mask : natoms-sized mask of ML region
    //
    // NOTE:
    // For TANI we cache atomic numbers (not compact species ids).
    // This matches the old tani.cu behavior, which used mlZidx directly
    // and then converted int32 -> int64 for TorchANI input.
    std::vector<int> ml_idx;
    std::vector<int> ml_zid;
    std::vector<int> ml_mask;

    // Loaded TorchScript model
    torch::jit::script::Module model;

    void reset() { *this = TaniState{}; }
};

TaniState& cache() {
    static TaniState S;
    return S;
}

// ============================================================================
// SMALL HELPERS
// ============================================================================

static void print_err(const char* where, const char* msg) {
    std::fprintf(stderr, "[%s] %s\n", where, msg);
}

static void zero_array(double* p, int n) {
    if (!p || n <= 0) return;
    for (int i = 0; i < n; ++i) p[i] = 0.0;
}

#if defined(_WIN32)
static bool set_env_kv(const char* key, const std::string& val) {
    std::string kv = std::string(key) + "=" + val;
    return (_putenv(kv.c_str()) == 0);
}
#else
static bool set_env_kv(const char* key, const std::string& val) {
    return ::setenv(key, val.c_str(), 1) == 0;
}
#endif

// ============================================================================
// DEVICE CONFIGURATION
// ============================================================================
//
// This is intentionally similar to type_dpmm.cxx so future model files can
// reuse the same pattern.
//
static void configure_device(TaniState& S, int gpu_id) {
    S.gpu_id  = gpu_id;
    S.use_gpu = (gpu_id >= 0);

#if KEY_CUDA==1

    if (S.use_gpu) {
        if (!set_env_kv("CUDA_VISIBLE_DEVICES", std::to_string(S.gpu_id))) {
            print_err("setup", "warning: failed to set CUDA_VISIBLE_DEVICES");
        }
    }

    if (S.use_gpu && torch::cuda::is_available()) {
        // With CUDA_VISIBLE_DEVICES restricted, chosen device is seen as cuda:0.
        S.device = torch::Device(torch::kCUDA, 0);
    } else {
        if (S.use_gpu && !torch::cuda::is_available()) {
            print_err("setup", "CUDA requested but unavailable; falling back to CPU");
        }
        S.use_gpu = false;
        S.device = torch::kCPU;
    }

#else

    if (S.use_gpu) {
        print_err(
            "setup",
            "CUDA requested, but this build was compiled without KEY_CUDA=1; falling back to CPU"
        );
    }

    S.use_gpu = false;
    S.device = torch::kCPU;

#endif
}

// ============================================================================
// TANI SETUP
// ============================================================================
//
// What this does:
//   1) validate setup ABI inputs
//   2) choose CPU/GPU
//   3) load TorchScript model
//   4) cache ML index / atomic number / mask arrays
//
// Future model authors should adapt this block if their setup needs different
// cached state.
//
static void tani_setup_impl(
    int tani_in_use,
    int tani_use_gpu,
    const char* tani_pt_name,
    int tani_pt_nml,
    const int* tani_in_mlidx,
    const int* tani_in_mlZid,
    const int* tani_in_mlmaskid,
    int tani_in_natoms,
    int* tani_out_setup_err)
{
    auto& S = cache();
    S.reset();

    // Assume failure until the entire setup completes successfully.
    if (tani_out_setup_err) *tani_out_setup_err = 1;

    if (tani_in_use != 1) {
        return;
    }

    if (!tani_pt_name || !tani_pt_name[0]) {
        print_err("setup", "empty model path");
        return;
    }
    if (tani_pt_nml <= 0) {
        print_err("setup", "invalid ML atom count");
        return;
    }
    if (tani_in_natoms <= 0) {
        print_err("setup", "invalid natoms");
        return;
    }
    if (!tani_in_mlidx || !tani_in_mlZid || !tani_in_mlmaskid) {
        print_err("setup", "null setup pointer");
        return;
    }

    try {
        configure_device(S, tani_use_gpu);

        S.model = torch::jit::load(std::string(tani_pt_name), S.device);
        S.model.eval();

        S.model_path = tani_pt_name;
        S.n_ml       = tani_pt_nml;
        S.natoms     = tani_in_natoms;

        S.ml_idx.assign(tani_in_mlidx, tani_in_mlidx + S.n_ml);
        S.ml_zid.assign(tani_in_mlZid, tani_in_mlZid + S.n_ml);
        S.ml_mask.assign(tani_in_mlmaskid, tani_in_mlmaskid + S.natoms);

        S.initialized = true;

        // Setup completed successfully
        if (tani_out_setup_err) *tani_out_setup_err = 0;

    } catch (const c10::Error& e) {
        std::fprintf(stderr, "[setup] Torch load failed: %s\n", e.what());
        S.reset();
        return;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[setup] setup failed: %s\n", e.what());
        S.reset();
        return;
    }
}

// ============================================================================
// TANI FORCE
// ============================================================================
//
// This is the model-specific runtime path.
//
// The old tani.cu logic is:
//
//   atom_types_i32  -> to(int64) -> batched
//   ml_coords       -> float32 -> batched -> requires_grad(true)
//   species_coords  = tuple(atom_types, ml_coords)
//   inputs          = [species_coords]
//   out_iv          = model.forward(inputs)
//   out_iv must be tuple-like
//   energy tensor   = tuple element 1
//   grad            = autograd::grad(sum(energy), ml_coords)
//
// We preserve that logic here. 
//
// IMPORTANT DIFFERENCE FROM DPMM:
// - DO NOT use torch::NoGradGuard here
// - autograd requires grad tracking on ml_coords
//
static void tani_force_impl(
    double* E_tani_c,
    const double* ml_x_c,
    const double* ml_y_c,
    const double* ml_z_c,
    double* ml_dx_c,
    double* ml_dy_c,
    double* ml_dz_c)
{
    auto& S = cache();

    if (!S.initialized) {
        print_err("force", "called before setup");
        return;
    }

    if (!E_tani_c || !ml_x_c || !ml_y_c || !ml_z_c ||
        !ml_dx_c || !ml_dy_c || !ml_dz_c) {
        print_err("force", "null pointer in force call");
        return;
    }

    const int64_t Nq = static_cast<int64_t>(S.n_ml);

    // Always zero outputs first
    *E_tani_c = 0.0;
    zero_array(ml_dx_c, S.n_ml);
    zero_array(ml_dy_c, S.n_ml);
    zero_array(ml_dz_c, S.n_ml);

    try {
        auto f32cpu = at::TensorOptions().dtype(at::kFloat).device(torch::kCPU);
        auto i32cpu = at::TensorOptions().dtype(at::kInt).device(torch::kCPU);
        auto i64cpu = at::TensorOptions().dtype(at::kLong).device(torch::kCPU);

        // --------------------------------------------------------------------
        // 1) Build ML coordinate tensor on CPU
        // Shape before batching: (Nq, 3)
        //
        // We then move to target device and call requires_grad_(true),
        // matching the old tani.cu autograd requirement. 
        // --------------------------------------------------------------------
        at::Tensor ml_coords_cpu = torch::empty({Nq, 3}, f32cpu);
        {
            auto qc = ml_coords_cpu.accessor<float, 2>();
            for (int64_t i = 0; i < Nq; ++i) {
                qc[i][0] = static_cast<float>(ml_x_c[i]);
                qc[i][1] = static_cast<float>(ml_y_c[i]);
                qc[i][2] = static_cast<float>(ml_z_c[i]);
            }
        }

        // --------------------------------------------------------------------
        // 2) Build atomic-number tensor as int32 first, then convert to int64
        //
        // This intentionally follows the old tani.cu flow:
        //   mlZidx -> Int tensor -> Long tensor -> shape (1, Nq)
        // rather than jumping directly to int64. 
        // --------------------------------------------------------------------
        at::Tensor atom_types_i32_cpu =
            torch::from_blob(S.ml_zid.data(), {Nq}, i32cpu).clone();

        // --------------------------------------------------------------------
        // 3) Move to device and batch
        // --------------------------------------------------------------------
        at::Tensor ml_coords = ml_coords_cpu.to(S.device).unsqueeze(0).contiguous();
        ml_coords.set_requires_grad(true);

        at::Tensor atom_types = atom_types_i32_cpu.to(S.device)
                                                 .to(at::kLong)
                                                 .view({1, Nq})
                                                 .contiguous();

        // --------------------------------------------------------------------
        // 4) Build TorchANI-style tuple input
        //
        // Old tani.cu:
        //   species_coords = Tuple(atom_types, ml_coords)
        //   inputs = [species_coords]
        // --------------------------------------------------------------------
        auto species_coords = c10::ivalue::Tuple::create(
            std::vector<c10::IValue>{atom_types, ml_coords}
        );

        std::vector<c10::IValue> inputs;
        inputs.reserve(1);
        inputs.emplace_back(species_coords);

        // --------------------------------------------------------------------
        // 5) Forward
        // --------------------------------------------------------------------
        c10::IValue out_iv;
        try {
            out_iv = S.model.forward(inputs);
        } catch (const c10::Error& e) {
            std::fprintf(stderr, "[force] model.forward FAILED: %s\n", e.what());
            return;
        } catch (...) {
            std::fprintf(stderr, "[force] model.forward FAILED: unknown exception\n");
            return;
        }

        // --------------------------------------------------------------------
        // 6) Output must be tuple-like
        //
        // Old tani.cu expects tuple output and uses element 1 as energy. 
        // --------------------------------------------------------------------
        if (!out_iv.isTuple()) {
            print_err("force", "model output is not tuple-like");
            return;
        }

        auto out_tuple = out_iv.toTuple();
        const auto& elems = out_tuple->elements();

        if (elems.size() < 2) {
            print_err("force", "model output tuple has size < 2");
            return;
        }

        at::Tensor eT = elems[1].toTensor().contiguous();
        if (!eT.defined()) {
            print_err("force", "energy tensor is undefined");
            return;
        }

        // --------------------------------------------------------------------
        // 7) Autograd
        // --------------------------------------------------------------------
        at::Tensor e_scalar = eT.sum();

        std::vector<at::Tensor> grads;
        try {
            grads = torch::autograd::grad(
                {e_scalar},
                {ml_coords},
                {},
                false,  // retain_graph
                false,  // create_graph
                false   // allow_unused
            );
        } catch (const c10::Error& e) {
            std::fprintf(stderr, "[force] autograd FAILED: %s\n", e.what());
            return;
        } catch (...) {
            std::fprintf(stderr, "[force] autograd FAILED: unknown exception\n");
            return;
        }

        if (grads.empty() || !grads[0].defined()) {
            print_err("force", "autograd returned empty/undefined gradient");
            return;
        }

        // --------------------------------------------------------------------
        // 8) Move energy and gradient to CPU before host access
        // --------------------------------------------------------------------
        at::Tensor grad_flat = grads[0]
            .detach()
            .to(torch::kCPU)
            .contiguous()
            .view({-1});

        at::Tensor e1 = eT
            .detach()
            .to(torch::kCPU)
            .contiguous()
            .view({-1});

        if (grad_flat.scalar_type() != at::kFloat) {
            grad_flat = grad_flat.to(at::kFloat).contiguous();
        }

        if (e1.scalar_type() != at::kFloat) {
            e1 = e1.to(at::kFloat).contiguous();
        }


        // --------------------------------------------------------------------
        // 9) Copy energy back
        // --------------------------------------------------------------------
        {
            const float* ep = e1.data_ptr<float>();
            *E_tani_c = static_cast<double>(ep[0]) * ENERGY_SCALE;
        }

        // --------------------------------------------------------------------
        // 10) Copy gradients back
        // --------------------------------------------------------------------
        {
            const float* gp = grad_flat.data_ptr<float>();
        
            for (int64_t i = 0; i < Nq; ++i) {
                ml_dx_c[i] = static_cast<double>(gp[3*i + 0]) * ENERGY_SCALE;
                ml_dy_c[i] = static_cast<double>(gp[3*i + 1]) * ENERGY_SCALE;
                ml_dz_c[i] = static_cast<double>(gp[3*i + 2]) * ENERGY_SCALE;
            }
        }

    } catch (const c10::Error& e) {
        std::fprintf(stderr, "[force] Torch execution failed: %s\n", e.what());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[force] force evaluation failed: %s\n", e.what());
    }
}

} // anonymous namespace

// ============================================================================
// EXPORTED C ABI WRAPPERS
// ============================================================================
//
// Keep these wrappers thin. Future model files should follow the same pattern.
//

extern "C" {

// ---------------------------------------------------------------------------
// Fortran bind(C, name="charmm_tani_internal_setup")
// ---------------------------------------------------------------------------
void charmm_tani_internal_setup(
    int tani_in_use,
    int tani_use_gpu,
    const char* tani_pt_name,
    int tani_pt_nml,
    const int* tani_in_mlidx,
    const int* tani_in_mlZid,
    const int* tani_in_mlmaskid,
    int tani_in_natoms,
    int* tani_out_setup_err)
{
    tani_setup_impl(
        tani_in_use,
        tani_use_gpu,
        tani_pt_name,
        tani_pt_nml,
        tani_in_mlidx,
        tani_in_mlZid,
        tani_in_mlmaskid,
        tani_in_natoms,
        tani_out_setup_err);
}

// ---------------------------------------------------------------------------
// Fortran bind(C, name="charmm_tani_internal_force")
// ---------------------------------------------------------------------------
void charmm_tani_internal_force(
    double* E_tani_c,
    const double* ml_x_c,
    const double* ml_y_c,
    const double* ml_z_c,
    double* ml_dx_c,
    double* ml_dy_c,
    double* ml_dz_c)
{
    tani_force_impl(
        E_tani_c,
        ml_x_c,
        ml_y_c,
        ml_z_c,
        ml_dx_c,
        ml_dy_c,
        ml_dz_c);
}

} // extern "C"

#endif
#endif