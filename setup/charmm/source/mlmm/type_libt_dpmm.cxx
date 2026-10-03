#if KEY_MLMM==1 && KEY_MLPTORCH==1
#ifdef WITH_TORCH
// ============================================================================
// type_libt_dpmm.cxx
// ============================================================================
//
// MODEL-SPECIFIC TorchScript bridge for DPMM.
//
// DPMM MODEL-SPECIFIC FILE
//
// The intended workflow for future models is:
//   1) copy this file
//   2) rename it to type_newmodel.cpp
//   3) change the cached state, input packing, and output unpacking
//
// This keeps each model isolated and easy to debug.
//
// ----------------------------------------------------------------------------
// CURRENT FORTRAN ABI MATCHED HERE (mlps_abi.F90):
// ----------------------------------------------------------------------------
//
// Setup:
//
//   charmm_dpmm_internal_setup(
//       int dpmm_in_use,
//       int dpmm_use_gpu,
//       const char* dpmm_pt_name,
//       int dpmm_pt_nml,
//       int dpmm_pt_nmm_max,
//       const int* dpmm_in_mlidx,
//       const int* dpmm_in_mlSid,
//       const int* dpmm_in_mlmaskid,
//       int dpmm_in_natoms,
//       int* dpmm_out_setup_err)
//
// Force:
//
//   charmm_dpmm_internal_force(
//       double* E_dpmm_c,
//       int mm_count_c,
//       const double* ml_x_c,
//       const double* ml_y_c,
//       const double* ml_z_c,
//       const double* mm_cg_c,
//       const double* mm_x_c,
//       const double* mm_y_c,
//       const double* mm_z_c,
//       double* ml_dx_c,
//       double* ml_dy_c,
//       double* ml_dz_c,
//       double* mm_dx_c,
//       double* mm_dy_c,
//       double* mm_dz_c)
//
// ----------------------------------------------------------------------------
// DPMM! IMPORTANT DESIGN CHOICE
// ----------------------------------------------------------------------------
//
// Fortran now sends a COMPACT MM list and the ACTUAL mm_count.
// Padding is done HERE in C++, not in Fortran.
//
// This is a better template design because future ML/MM models may not want
// the same padding behavior.
//
// ----------------------------------------------------------------------------
// DPMM!MODEL OUTPUT ASSUMPTIONS
// ----------------------------------------------------------------------------
//
// This DPMM bridge assumes the TorchScript model returns a dict containing:
//
//   energy or dE
//   qm_grad / qm_dgrad / qm_grad_high / qm_grad_low
//   mm_espgrad_d
//
// If a future model uses different names, change only the output unpacking.
//
// ----------------------------------------------------------------------------
// DPMM! UNIT CONVENTION
// ----------------------------------------------------------------------------
//
// The old working bridge assumed eV outputs and converted to kcal/mol.
// If your model already outputs kcal/mol, set ENERGY_SCALE = 1.0.
//
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>

#include <torch/script.h>
#if KEY_CUDA==1
#include <torch/cuda.h>
#endif

namespace {

// ============================================================================
// ENERGY UNIT SCALE
// ============================================================================
constexpr double ENERGY_SCALE = 23.060547830619; // eV -> kcal/mol

// ============================================================================
// DPMM CACHED STATE
// ============================================================================
//
// Everything stored here is fixed after setup and reused in every force call.
//
struct DpmmState {
    bool initialized = false;

    // Model path for logging/debug
    std::string model_path;

    // Device info
    bool use_gpu = false;
    int  gpu_id  = -1;
    torch::Device device = torch::kCPU;

    // Sizes
    int n_ml   = 0;   // number of compact ML atoms
    int mm_max = 0;   // padded MM capacity (MXMM)
    int natoms = 0;   // total atom count in the system

    // Cached setup arrays
    std::vector<int> ml_idx;   // compact ML index -> global atom index
    std::vector<int> ml_sid;   // compact ML species ids
    std::vector<int> ml_mask;  // natoms-sized ML mask

    // TorchScript model
    torch::jit::script::Module model;

    void reset() { *this = DpmmState{}; }
};

DpmmState& cache() {
    static DpmmState S;
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
// TORCH OUTPUT HELPERS
// ============================================================================

static at::Tensor get_energy_cpu(const c10::impl::GenericDict& D) {
    for (auto key : {"energy", "dE"}) {
        auto it = D.find(c10::IValue(std::string(key)));
        if (it != D.end()) {
            return it->value().toTensor().to(torch::kCPU).squeeze();
        }
    }
    throw std::runtime_error("missing energy/dE");
}

static at::Tensor get_f32_cpu_3d(const c10::impl::GenericDict& D,
                                 std::initializer_list<const char*> keys) {
    for (auto key : keys) {
        auto it = D.find(c10::IValue(std::string(key)));
        if (it != D.end()) {
            at::Tensor t = it->value().toTensor().to(torch::kCPU);
            if (t.scalar_type() != at::kFloat) t = t.to(at::kFloat);
            if (!t.is_contiguous()) t = t.contiguous();

            if (t.dim() != 3) {
                std::fprintf(stderr, "[bridge] key '%s' has dim=%d (want 3)\n",
                             key, static_cast<int>(t.dim()));
                throw std::runtime_error("bad tensor rank");
            }
            return t;
        }
    }
    throw std::runtime_error("missing expected 3D tensor");
}

// ============================================================================
// DEVICE CONFIGURATION
// ============================================================================
//
// Generic enough that future model files can reuse this almost unchanged.
//
static void configure_device(DpmmState& S, int gpu_id) {
    S.gpu_id  = gpu_id;
    S.use_gpu = (gpu_id >= 0);

#if KEY_CUDA==1

    if (S.use_gpu) {
        if (!set_env_kv("CUDA_VISIBLE_DEVICES", std::to_string(S.gpu_id))) {
            print_err("setup", "warning: failed to set CUDA_VISIBLE_DEVICES");
        }
    }

    if (S.use_gpu && torch::cuda::is_available()) {
        // Because CUDA_VISIBLE_DEVICES is restricted, requested device is seen as cuda:0 here.
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
        print_err("setup", "CUDA requested, but this build was compiled without KEY_CUDA=1; falling back to CPU");
    }

    S.use_gpu = false;
    S.device = torch::kCPU;

#endif
}

// ============================================================================
// DPMM SETUP
// ============================================================================
//
// Future model require modifications in this section if their setup needs more
// or less cached information.
//
static void dpmm_setup_impl(
    int dpmm_in_use,
    int dpmm_use_gpu,
    const char* dpmm_pt_name,
    int dpmm_pt_nml,
    int dpmm_pt_nmm_max,
    const int* dpmm_in_mlidx,
    const int* dpmm_in_mlSid,
    const int* dpmm_in_mlmaskid,
    int dpmm_in_natoms,
    int* dpmm_out_setup_err)
{
    auto& S = cache();
    S.reset();

    // Assume failure until the entire setup completes successfully.
    if (dpmm_out_setup_err) *dpmm_out_setup_err = 1;

    if (dpmm_in_use != 1) {
        return;
    }

    // -------------------------
    // Validate setup inputs
    // -------------------------
    if (!dpmm_pt_name || !dpmm_pt_name[0]) {
        print_err("setup", "empty model path");
        return;
    }
    if (dpmm_pt_nml <= 0) {
        print_err("setup", "invalid ML atom count");
        return;
    }
    if (dpmm_pt_nmm_max <= 0) {
        print_err("setup", "invalid MM max count");
        return;
    }
    if (dpmm_in_natoms <= 0) {
        print_err("setup", "invalid natoms");
        return;
    }
    if (!dpmm_in_mlidx || !dpmm_in_mlSid || !dpmm_in_mlmaskid) {
        print_err("setup", "null setup pointer");
        return;
    }

    try {
        // -------------------------
        // Select device and load model
        // -------------------------
        configure_device(S, dpmm_use_gpu);

        S.model = torch::jit::load(std::string(dpmm_pt_name), S.device);
        S.model.eval();

        // -------------------------
        // Cache setup-time metadata
        // -------------------------
        S.model_path = dpmm_pt_name;
        S.n_ml       = dpmm_pt_nml;
        S.mm_max     = dpmm_pt_nmm_max;
        S.natoms     = dpmm_in_natoms;

        S.ml_idx.assign(dpmm_in_mlidx, dpmm_in_mlidx + S.n_ml);
        S.ml_sid.assign(dpmm_in_mlSid, dpmm_in_mlSid + S.n_ml);
        S.ml_mask.assign(dpmm_in_mlmaskid, dpmm_in_mlmaskid + S.natoms);

        S.initialized = true;

        // Everything succeeded.
        if (dpmm_out_setup_err) *dpmm_out_setup_err = 0;
        
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
// DPMM FORCE
// ============================================================================
//
// This is where DPMM-specific input packing happens.
//
// IMPORTANT:
// - Fortran passes COMPACT MM data with length mm_count
// - This function pads internally up to S.mm_max
// - This is the key change from your old design
//
static void dpmm_force_impl(
    double* E_dpmm_c,
    int mm_count_c,
    const double* ml_x_c,
    const double* ml_y_c,
    const double* ml_z_c,
    const double* mm_cg_c,
    const double* mm_x_c,
    const double* mm_y_c,
    const double* mm_z_c,
    double* ml_dx_c,
    double* ml_dy_c,
    double* ml_dz_c,
    double* mm_dx_c,
    double* mm_dy_c,
    double* mm_dz_c)
{
    auto& S = cache();

    if (!S.initialized) {
        print_err("force", "called before setup");
        return;
    }

    if (!E_dpmm_c || !ml_x_c || !ml_y_c || !ml_z_c ||
        !mm_cg_c || !mm_x_c || !mm_y_c || !mm_z_c ||
        !ml_dx_c || !ml_dy_c || !ml_dz_c ||
        !mm_dx_c || !mm_dy_c || !mm_dz_c) {
        print_err("force", "null pointer in force call");
        return;
    }

    if (mm_count_c < 0 || mm_count_c > S.mm_max) {
        print_err("force", "invalid mm_count");
        return;
    }

    const int64_t Nq       = static_cast<int64_t>(S.n_ml);
    const int64_t mm_count = static_cast<int64_t>(mm_count_c);
    const int64_t Mmax     = static_cast<int64_t>(S.mm_max);

    // -------------------------
    // Clear outputs
    // -------------------------
    *E_dpmm_c = 0.0;
    zero_array(ml_dx_c, S.n_ml);
    zero_array(ml_dy_c, S.n_ml);
    zero_array(ml_dz_c, S.n_ml);
    zero_array(mm_dx_c, S.mm_max);
    zero_array(mm_dy_c, S.mm_max);
    zero_array(mm_dz_c, S.mm_max);

    try {
        torch::NoGradGuard no_grad;

        auto f32cpu = at::TensorOptions().dtype(at::kFloat).device(torch::kCPU);
        auto i64cpu = at::TensorOptions().dtype(at::kLong).device(torch::kCPU);

        // --------------------------------------------------------------------
        // 1) Build QM coordinate tensor
        // Shape: (Nq, 3)
        // --------------------------------------------------------------------
        at::Tensor qm_coords_cpu = torch::empty({Nq, 3}, f32cpu);
        {
            auto qc = qm_coords_cpu.accessor<float, 2>();
            for (int64_t i = 0; i < Nq; ++i) {
                qc[i][0] = static_cast<float>(ml_x_c[i]);
                qc[i][1] = static_cast<float>(ml_y_c[i]);
                qc[i][2] = static_cast<float>(ml_z_c[i]);
            }
        }

        // --------------------------------------------------------------------
        // 2) Build ML atom type tensor from cached species ids
        // Shape: (Nq)
        // --------------------------------------------------------------------
        std::vector<int64_t> atom_types_vec(Nq);
        for (int64_t i = 0; i < Nq; ++i) {
            atom_types_vec[i] = static_cast<int64_t>(S.ml_sid[static_cast<size_t>(i)]);
        }
        at::Tensor atom_types_cpu =
            torch::from_blob(atom_types_vec.data(), {Nq}, i64cpu).clone();

        // --------------------------------------------------------------------
        // 3) Build MM tensors with INTERNAL padding
        //
        // Real MM rows occupy [0 .. mm_count-1]
        // Padded rows occupy [mm_count .. Mmax-1]
        //
        // Shapes:
        //   mm_coords_cpu : (Mmax, 3)
        //   mm_Q_cpu      : (Mmax)
        //   mm_type_cpu   : (Mmax)
        // --------------------------------------------------------------------
        at::Tensor mm_coords_cpu = torch::zeros({Mmax, 3}, f32cpu);
        at::Tensor mm_Q_cpu      = torch::zeros({Mmax}, f32cpu);
        at::Tensor mm_type_cpu   = torch::zeros({Mmax}, f32cpu);

        {
            auto mc = mm_coords_cpu.accessor<float, 2>();
            auto mq = mm_Q_cpu.accessor<float, 1>();
            auto mt = mm_type_cpu.accessor<float, 1>();

            // Copy real MM rows
            for (int64_t i = 0; i < mm_count; ++i) {
                mc[i][0] = static_cast<float>(mm_x_c[i]);
                mc[i][1] = static_cast<float>(mm_y_c[i]);
                mc[i][2] = static_cast<float>(mm_z_c[i]);
                mq[i]    = static_cast<float>(mm_cg_c[i]);
                mt[i]    = 1.0f;
            }

            // Remaining rows stay zero-padded with mt=0
        }

        // --------------------------------------------------------------------
        // 4) Move to target device and add batch dimension
        // --------------------------------------------------------------------
        at::Tensor qm_coords  = qm_coords_cpu.to(S.device).unsqueeze(0);
        at::Tensor atom_types = atom_types_cpu.to(S.device).unsqueeze(0);
        at::Tensor mm_coords  = mm_coords_cpu.to(S.device).unsqueeze(0);
        at::Tensor mm_Q       = mm_Q_cpu.to(S.device).unsqueeze(0);
        at::Tensor mm_type    = mm_type_cpu.to(S.device).unsqueeze(0);

        // --------------------------------------------------------------------
        // 5) Forward pass
        //
        // If future model needs different inputs, change THIS block.
        // --------------------------------------------------------------------
        std::vector<c10::IValue> inputs;
        inputs.reserve(5);
        inputs.emplace_back(qm_coords);
        inputs.emplace_back(atom_types);
        inputs.emplace_back(mm_coords);
        inputs.emplace_back(mm_Q);
        inputs.emplace_back(mm_type);

        c10::IValue out_iv = S.model.forward(inputs);
        if (!out_iv.isGenericDict()) {
            throw std::runtime_error("model output is not a dict");
        }

        auto out_dict = out_iv.toGenericDict();

        // --------------------------------------------------------------------
        // 6) Energy
        // --------------------------------------------------------------------
        {
            at::Tensor eT = get_energy_cpu(out_dict);
            const double e_model = eT.item<double>();
            *E_dpmm_c = e_model * ENERGY_SCALE;
        }

        // --------------------------------------------------------------------
        // 7) QM gradients
        //
        // Accepted keys:
        //   qm_grad
        //   qm_dgrad
        //   qm_grad_high
        //   qm_grad_low
        // --------------------------------------------------------------------
        {
            at::Tensor qgT = get_f32_cpu_3d(
                out_dict, {"qm_grad", "qm_dgrad", "qm_grad_high", "qm_grad_low"});

            if (qgT.size(1) != Nq || qgT.size(2) != 3) {
                std::fprintf(stderr,
                             "[force] qm_grad shape mismatch: got (%ld,%ld,%ld), want (*,%ld,3)\n",
                             static_cast<long>(qgT.size(0)),
                             static_cast<long>(qgT.size(1)),
                             static_cast<long>(qgT.size(2)),
                             static_cast<long>(Nq));
                return;
            }

            auto qg = qgT.accessor<float, 3>();
            for (int64_t i = 0; i < Nq; ++i) {
                ml_dx_c[i] = static_cast<double>(qg[0][i][0]) * ENERGY_SCALE;
                ml_dy_c[i] = static_cast<double>(qg[0][i][1]) * ENERGY_SCALE;
                ml_dz_c[i] = static_cast<double>(qg[0][i][2]) * ENERGY_SCALE;
            }
        }

        // --------------------------------------------------------------------
        // 8) MM forces from electrostatic-gradient output
        //
        // Expected key:
        //   mm_espgrad_d
        //
        // NOTE:
        // Only the real MM rows [0 .. mm_count-1] are copied back.
        // Padded rows are ignored.
        // --------------------------------------------------------------------
        {
            // Prefer mm_espgrad_high, but fall back to mm_espgrad_d for older models.
            // Assumes this tensor is still per-unit-charge ESP-gradient-like output.
            at::Tensor egT = get_f32_cpu_3d(out_dict, {"mm_espgrad_high", "mm_espgrad_d", "mm_espgrad", "mm_espgrad_low"});
        
            if (egT.size(1) < Mmax || egT.size(2) != 3) {
                std::fprintf(stderr,
                             "[force] MM espgrad shape mismatch: got (%ld,%ld,%ld), want (*,%ld,3)\n",
                             static_cast<long>(egT.size(0)),
                             static_cast<long>(egT.size(1)),
                             static_cast<long>(egT.size(2)),
                             static_cast<long>(Mmax));
                return;
            }
        
            auto eg = egT.accessor<float, 3>();
            for (int64_t i = 0; i < mm_count; ++i) {
                const double q = mm_cg_c[i];
                mm_dx_c[i] = static_cast<double>(eg[0][i][0]) * q * ENERGY_SCALE;
                mm_dy_c[i] = static_cast<double>(eg[0][i][1]) * q * ENERGY_SCALE;
                mm_dz_c[i] = static_cast<double>(eg[0][i][2]) * q * ENERGY_SCALE;
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
extern "C" {

// ---------------------------------------------------------------------------
// Fortran bind(C, name="charmm_dpmm_internal_setup")
// ---------------------------------------------------------------------------
void charmm_dpmm_internal_setup(
    int dpmm_in_use,
    int dpmm_use_gpu,
    const char* dpmm_pt_name,
    int dpmm_pt_nml,
    int dpmm_pt_nmm_max,
    const int* dpmm_in_mlidx,
    const int* dpmm_in_mlSid,
    const int* dpmm_in_mlmaskid,
    int dpmm_in_natoms,
    int* dpmm_out_setup_err)
{
    dpmm_setup_impl(
        dpmm_in_use,
        dpmm_use_gpu,
        dpmm_pt_name,
        dpmm_pt_nml,
        dpmm_pt_nmm_max,
        dpmm_in_mlidx,
        dpmm_in_mlSid,
        dpmm_in_mlmaskid,
        dpmm_in_natoms,
        dpmm_out_setup_err);
}

// ---------------------------------------------------------------------------
// Fortran bind(C, name="charmm_dpmm_internal_force")
// ---------------------------------------------------------------------------
void charmm_dpmm_internal_force(
    double* E_dpmm_c,
    int mm_count_c,
    const double* ml_x_c,
    const double* ml_y_c,
    const double* ml_z_c,
    const double* mm_cg_c,
    const double* mm_x_c,
    const double* mm_y_c,
    const double* mm_z_c,
    double* ml_dx_c,
    double* ml_dy_c,
    double* ml_dz_c,
    double* mm_dx_c,
    double* mm_dy_c,
    double* mm_dz_c)
{
    dpmm_force_impl(
        E_dpmm_c,
        mm_count_c,
        ml_x_c,
        ml_y_c,
        ml_z_c,
        mm_cg_c,
        mm_x_c,
        mm_y_c,
        mm_z_c,
        ml_dx_c,
        ml_dy_c,
        ml_dz_c,
        mm_dx_c,
        mm_dy_c,
        mm_dz_c);
}

} // extern "C"

#endif
#endif