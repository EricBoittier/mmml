#if KEY_MLMM==1 && KEY_MLPTORCH==1
#ifdef WITH_TORCH
// ============================================================================
// type_libt_dummy.cxx
// ============================================================================
// Native LibTorch harmonic dummy predictor for MLMM interface testing.
//
// Potential (coordinates in Angstrom):
//
//   E = 1/2 k sum_i |r_i - r_i^(0)|^2
//   dE/dr_i = k (r_i - r_i^(0))
//
// The first force call stores r^(0), so the first returned energy and gradient
// are exactly zero.  The spring constant is read from MLPS_DUMMY_K and defaults
// to 1.0 kcal mol^-1 Angstrom^-2.
//
// this implementation returns CHARMM units directly:
//
//   energy   : kcal/mol
//   gradient : kcal/mol/Angstrom
//
// ============================================================================

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>
#include <torch/autograd.h>

#if KEY_CUDA==1
#include <torch/cuda.h>
#endif

namespace {

static void print_err(const char* where, const char* msg) {
    std::fprintf(stderr, "[LIBT-DUMMY:%s] %s\n", where, msg);
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

static double read_dummy_k() {
    constexpr double DEFAULT_K = 1.0;
    const char* raw = std::getenv("MLPS_DUMMY_K");
    if (!raw || !raw[0]) return DEFAULT_K;

    errno = 0;
    char* end = nullptr;
    const double value = std::strtod(raw, &end);

    if (errno != 0 || end == raw || (end && *end != '\0') || !std::isfinite(value)) {
        throw std::runtime_error(
            std::string("invalid MLPS_DUMMY_K value: '") + raw + "'");
    }
    return value;
}

// ============================================================================
// TORCH MODULE
// ============================================================================

class HarmonicDummyImpl : public torch::nn::Module {
public:
    explicit HarmonicDummyImpl(double spring_constant)
        : k_(spring_constant) {}

    torch::Tensor forward(const torch::Tensor& coords) {
        if (!coords.defined()) {
            throw std::runtime_error("undefined coordinate tensor");
        }
        if (coords.dim() != 2 || coords.size(1) != 3) {
            throw std::runtime_error("coordinates must have shape (N,3)");
        }

        // Match the Python dummy: establish the reference geometry on the
        // first force call. detach() prevents r0 from entering the graph.
        if (!has_reference_) {
            r0_ = coords.detach().clone();
            has_reference_ = true;
        }

        const torch::Tensor dr = coords - r0_;
        return 0.5 * k_ * dr.pow(2).sum();  
    }

    bool has_reference() const noexcept { return has_reference_; }

private:
    double k_ = 1.0;
    bool has_reference_ = false;
    torch::Tensor r0_;
};

// ============================================================================
// CACHED STATE
// ============================================================================

struct DummyState {
    bool initialized = false;

    bool use_gpu = false;
    int gpu_id = -1;
    torch::Device device = torch::kCPU;

    int n_ml = 0;
    int natoms = 0;
    double k = 1.0;

    // Cached only for ABI consistency/debugging.  The harmonic potential does
    // not depend on atom identity.
    std::vector<int> ml_idx;
    std::vector<int> ml_zid;
    std::vector<int> ml_mask;

    std::shared_ptr<HarmonicDummyImpl> model;

    void reset() { *this = DummyState{}; }
};

DummyState& cache() {
    static DummyState S;
    return S;
}

static void configure_device(DummyState& S, int gpu_id) {
    S.gpu_id = gpu_id;
    S.use_gpu = (gpu_id >= 0);

#if KEY_CUDA==1
    if (S.use_gpu) {
        if (!set_env_kv("CUDA_VISIBLE_DEVICES", std::to_string(S.gpu_id))) {
            print_err("setup", "warning: failed to set CUDA_VISIBLE_DEVICES");
        }
    }

    if (S.use_gpu && torch::cuda::is_available()) {
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
            "CUDA requested, but this build was compiled without KEY_CUDA=1; falling back to CPU");
    }
    S.use_gpu = false;
    S.device = torch::kCPU;
#endif
}

// ============================================================================
// SETUP
// ============================================================================

static void dummy_setup_impl(
    int dummy_in_use,
    int dummy_use_gpu,
    int dummy_in_nml,
    const int* dummy_in_mlidx,
    const int* dummy_in_mlZid,
    const int* dummy_in_mlmaskid,
    int dummy_in_natoms,
    int* dummy_out_setup_err)
{
    auto& S = cache();
    S.reset();

    if (dummy_out_setup_err) *dummy_out_setup_err = 1;

    if (dummy_in_use != 1) return;

    if (dummy_in_nml <= 0) {
        print_err("setup", "invalid ML atom count");
        return;
    }
    if (dummy_in_natoms <= 0) {
        print_err("setup", "invalid natoms");
        return;
    }
    if (!dummy_in_mlidx || !dummy_in_mlZid || !dummy_in_mlmaskid) {
        print_err("setup", "null setup pointer");
        return;
    }

    try {
        configure_device(S, dummy_use_gpu);

        S.n_ml = dummy_in_nml;
        S.natoms = dummy_in_natoms;
        S.k = read_dummy_k();

        S.ml_idx.assign(dummy_in_mlidx, dummy_in_mlidx + S.n_ml);
        S.ml_zid.assign(dummy_in_mlZid, dummy_in_mlZid + S.n_ml);
        S.ml_mask.assign(dummy_in_mlmaskid, dummy_in_mlmaskid + S.natoms);

        S.model = std::make_shared<HarmonicDummyImpl>(S.k);
        S.model->eval();

        S.initialized = true;
        if (dummy_out_setup_err) *dummy_out_setup_err = 0;

        std::fprintf(
            stderr,
            "[LIBT-DUMMY] initialized: nml=%d k=%.12g device=%s\n",
            S.n_ml,
            S.k,
            S.device.str().c_str());
        std::fflush(stderr);

    } catch (const c10::Error& e) {
        std::fprintf(stderr, "[LIBT-DUMMY:setup] Torch setup failed: %s\n", e.what());
        S.reset();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[LIBT-DUMMY:setup] setup failed: %s\n", e.what());
        S.reset();
    }
}

// ============================================================================
// FORCE
// ============================================================================

static void dummy_force_impl(
    double* E_dummy_c,
    const double* ml_x_c,
    const double* ml_y_c,
    const double* ml_z_c,
    double* ml_dx_c,
    double* ml_dy_c,
    double* ml_dz_c)
{
    auto& S = cache();

    if (E_dummy_c) *E_dummy_c = 0.0;
    zero_array(ml_dx_c, S.n_ml);
    zero_array(ml_dy_c, S.n_ml);
    zero_array(ml_dz_c, S.n_ml);

    if (!S.initialized || !S.model) {
        print_err("force", "called before setup");
        return;
    }

    if (!E_dummy_c || !ml_x_c || !ml_y_c || !ml_z_c ||
        !ml_dx_c || !ml_dy_c || !ml_dz_c) {
        print_err("force", "null pointer in force call");
        return;
    }

    const int64_t Nq = static_cast<int64_t>(S.n_ml);

    try {
        // Use float64 throughout so this test path does not introduce an
        // unnecessary CHARMM double -> float -> double conversion.
        std::vector<double> xyz(static_cast<std::size_t>(3 * Nq));
        for (int64_t i = 0; i < Nq; ++i) {
            xyz[static_cast<std::size_t>(3*i + 0)] = ml_x_c[i];
            xyz[static_cast<std::size_t>(3*i + 1)] = ml_y_c[i];
            xyz[static_cast<std::size_t>(3*i + 2)] = ml_z_c[i];
        }

        auto cpu_opts = torch::TensorOptions()
            .dtype(torch::kFloat64)
            .device(torch::kCPU);

        torch::Tensor coords = torch::from_blob(
            xyz.data(), {Nq, 3}, cpu_opts).clone().to(S.device);

        coords.set_requires_grad(true);

        torch::Tensor energy = S.model->forward(coords);
        if (!energy.defined() || energy.numel() != 1) {
            print_err("force", "dummy model returned invalid energy tensor");
            return;
        }

        std::vector<torch::Tensor> grads = torch::autograd::grad(
            {energy},
            {coords},
            {},
            false,  // retain_graph
            false,  // create_graph
            false   // allow_unused
        );

        if (grads.empty() || !grads[0].defined()) {
            print_err("force", "autograd returned empty/undefined gradient");
            return;
        }

        torch::Tensor energy_cpu = energy.detach()
            .to(torch::kCPU, torch::kFloat64)
            .contiguous()
            .view({-1});

        torch::Tensor grad_cpu = grads[0].detach()
            .to(torch::kCPU, torch::kFloat64)
            .contiguous()
            .view({Nq, 3});

        *E_dummy_c = energy_cpu.data_ptr<double>()[0];

        const double* gp = grad_cpu.data_ptr<double>();
        for (int64_t i = 0; i < Nq; ++i) {
            ml_dx_c[i] = gp[3*i + 0];
            ml_dy_c[i] = gp[3*i + 1];
            ml_dz_c[i] = gp[3*i + 2];
        }

    } catch (const c10::Error& e) {
        std::fprintf(stderr, "[LIBT-DUMMY:force] Torch execution failed: %s\n", e.what());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[LIBT-DUMMY:force] force evaluation failed: %s\n", e.what());
    }
}

} // anonymous namespace

extern "C" {

void charmm_dummy_internal_setup(
    int dummy_in_use,
    int dummy_use_gpu,
    int dummy_in_nml,
    const int* dummy_in_mlidx,
    const int* dummy_in_mlZid,
    const int* dummy_in_mlmaskid,
    int dummy_in_natoms,
    int* dummy_out_setup_err)
{
    dummy_setup_impl(
        dummy_in_use,
        dummy_use_gpu,
        dummy_in_nml,
        dummy_in_mlidx,
        dummy_in_mlZid,
        dummy_in_mlmaskid,
        dummy_in_natoms,
        dummy_out_setup_err);
}

void charmm_dummy_internal_force(
    double* E_dummy_c,
    const double* ml_x_c,
    const double* ml_y_c,
    const double* ml_z_c,
    double* ml_dx_c,
    double* ml_dy_c,
    double* ml_dz_c)
{
    dummy_force_impl(
        E_dummy_c,
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