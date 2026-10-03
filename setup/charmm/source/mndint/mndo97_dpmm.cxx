#if KEY_MNDO97==1 /*mndo97*/
#if KEY_MLPTORCH==1
#ifdef WITH_TORCH
// mndo97_dpmm.cxx bridge — OMP-aware (MPI arg repurposed as OMP thread count)
//
// Fortran bind(C) symbols (ABI unchanged):
//   bind(C,name="mndo97_dpmm_internal_setup")
//   bind(C,name="mndo97_dpmm_internal")
//
// Arg semantics:
// - setup(..., dpmm_use_gpu, dpmm_use_omp, ...):
//     dpmm_use_gpu: -1 => CPU (default), >=0 => expose that GPU via CUDA_VISIBLE_DEVICES
//     dpmm_use_omp: <=0 => library defaults; >0 => set OMP/MKL/BLAS threads and torch thread pools
// - internal(..., /*dpmm_use_gpu*/, /*dpmm_use_omp*/, ...): ignored (device/threads fixed in setup)
//
// Model outputs used: "energy", "qm_grad", "mm_espgrad_d" (ESP-grad per unit charge).
// We multiply mm_espgrad_d by q on the host to get MM Cartesian forces.
//
// Build: link against LibTorch (C++ API).
// LibTorch headers will require C++17 or later

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cctype>
#include <stdexcept>

#include <torch/script.h>
#include <ATen/Parallel.h>  // at::set_num_threads, at::set_num_interop_threads

#if KEY_CUDA==1
#include <torch/cuda.h>
#endif

namespace {

struct CachedState {
    bool initialized = false;
    std::string model_path, ctrl_path;

    bool use_gpu = false;
    int  gpu_id  = -1;
    torch::Device device = torch::kCPU;

    int n_qm_max = 0;
    int mm_max   = 0;
    std::vector<int> species_z;

    torch::jit::script::Module model;

    void reset() { *this = CachedState{}; }
};

CachedState& cache() { static CachedState S; return S; }

// ---------------- helpers ----------------
static inline std::string ltrim(const std::string& s) {
    size_t i = 0; while (i < s.size() && std::isspace((unsigned char)s[i])) ++i;
    return s.substr(i);
}

static bool parse_meta(const char* path,
                       int& out_n_qm, int& out_max_mm,
                       std::vector<int>& out_species, int& out_n_types_opt)
{
    out_n_qm = 0; out_max_mm = 0; out_species.clear(); out_n_types_opt = -1;
    std::ifstream fin(path);
    if (!fin) { std::fprintf(stderr,"[setup] open meta failed: %s\n", path?path:"(null)"); return false; }
    std::string line;
    while (std::getline(fin, line)) {
        line = ltrim(line);
        if (line.empty() || line[0]=='#') continue;
        std::istringstream iss(line);
        std::string key; iss >> key;
        if (key=="n_qm") iss>>out_n_qm;
        else if (key=="max_mm") iss>>out_max_mm;
        else if (key=="n_types") iss>>out_n_types_opt;
        else if (key=="species_z_ordered"){ int z; while(iss>>z) out_species.push_back(z); }
    }
    if (out_species.empty()) { std::fprintf(stderr,"[setup] meta missing species_z_ordered\n"); return false; }
    if (out_max_mm<=0) { std::fprintf(stderr,"[setup] meta missing/invalid max_mm\n"); return false; }
    if (out_n_types_opt>=0 && out_n_types_opt!=(int)out_species.size())
        std::fprintf(stderr,"[setup] n_types(%d)!=len(species)(%zu) [ok]\n",out_n_types_opt,out_species.size());
    if (out_n_qm<=0) std::fprintf(stderr,"[setup] warn: meta missing/invalid n_qm [ok]\n");
    return true;
}

static inline int64_t z_to_type(const std::vector<int>& species, int z) {
    for (size_t i=0;i<species.size();++i) if (species[i]==z) return (int64_t)i;
    return -1;
}

#if defined(_WIN32)
static bool set_env_kv(const char* key, const std::string& val) {
    std::string kv = std::string(key)+"="+val; return (_putenv(kv.c_str())==0);
}
#else
static bool set_env_kv(const char* key, const std::string& val) {
    return ::setenv(key, val.c_str(), 1)==0;
}
#endif

// (best-effort) set OMP/BLAS thread envs before heavy libs spawn threads
static void set_omp_env(int threads) {
    if (threads <= 0) return;
    const std::string th = std::to_string(threads);
    set_env_kv("OMP_NUM_THREADS", th);
    set_env_kv("MKL_NUM_THREADS", th);
    set_env_kv("OPENBLAS_NUM_THREADS", th);
    set_env_kv("BLIS_NUM_THREADS", th);
    set_env_kv("NUMEXPR_NUM_THREADS", th);
    // reduce oversubscription
    set_env_kv("OMP_PROC_BIND", "true");
    set_env_kv("OMP_PLACES", "cores");
}

// Fetch tensor by any of the keys; ensure contiguous float32 on CPU with rank-3.
static at::Tensor get_f32_cpu_3d(const c10::impl::GenericDict& D,
                                 std::initializer_list<const char*> keys) {
    for (auto k : keys) {
        auto it = D.find(c10::IValue(std::string(k)));
        if (it != D.end()) {
            at::Tensor t = it->value().toTensor().to(torch::kCPU);
            if (t.scalar_type() != at::kFloat) t = t.to(at::kFloat);
            if (!t.is_contiguous()) t = t.contiguous();
            if (t.dim()!=3) {
                std::fprintf(stderr,"[bridge] key '%s' has dim=%d (want 3)\n", k, (int)t.dim());
                throw std::runtime_error("bad rank");
            }
            return t;
        }
    }
    throw std::runtime_error("missing expected 3D tensor");
}

static at::Tensor get_energy_cpu(const c10::impl::GenericDict& D) {
    for (auto k : {"energy","dE"}) {
        auto it = D.find(c10::IValue(std::string(k)));
        if (it != D.end()) {
            return it->value().toTensor().to(torch::kCPU).squeeze();
        }
    }
    throw std::runtime_error("missing energy/dE");
}

} // namespace

extern "C" {

// -------------------------------
// setup (C binding)
// -------------------------------
void mndo97_dpmm_internal_setup(
    const char* dpmm_ptname, const char* dpmm_ctrlname,
    int dpmm_use_gpu, int dpmm_use_omp,  // NOTE: 4th arg repurposed: OMP threads (was MPI)
    int* dpmm_types, int* dpmm_ntypes, int* dpmm_qmmax, int* dpmm_mmmax, int* dpmm_setup_err)
{
    auto& S = cache(); S.reset();

    // Assume failure until the entire setup completes successfully.
    if (dpmm_setup_err) *dpmm_setup_err = 1;

    if (!dpmm_ptname||!dpmm_ptname[0]) { std::fprintf(stderr,"[setup] empty model path\n"); return; }
    if (!dpmm_ctrlname||!dpmm_ctrlname[0]) { std::fprintf(stderr,"[setup] empty meta path\n"); return; }

    // OMP/threads policy first (before Torch spawns pools)
    if (dpmm_use_omp > 0) {
        set_omp_env(dpmm_use_omp);
        try {
            at::set_num_threads(dpmm_use_omp);                  // intra-op
            at::set_num_interop_threads(std::max(1, std::min(2, dpmm_use_omp))); // conservative
        } catch (...) {
            // non-fatal
        }
        std::fprintf(stderr, "[setup] OMP threads = %d\n", dpmm_use_omp);
    }


    // Device policy: CPU default; GPU only if KEY_CUDA=1 and CUDA is available.
    S.gpu_id  = dpmm_use_gpu;
    S.use_gpu = (dpmm_use_gpu >= 0);

#if KEY_CUDA==1

    if (S.use_gpu) {
        if (!set_env_kv("CUDA_VISIBLE_DEVICES", std::to_string(S.gpu_id))) {
            std::fprintf(stderr,"[setup] warn: set CUDA_VISIBLE_DEVICES failed\n");
        }
    } else {
        // Keep CUDA invisible by default.
        (void)set_env_kv("CUDA_VISIBLE_DEVICES", "");
    }

    if (S.use_gpu && torch::cuda::is_available()) {
        // With CUDA_VISIBLE_DEVICES restricted, requested GPU appears as cuda:0.
        S.device = torch::Device(torch::kCUDA, 0);
    } else {
        if (S.use_gpu && !torch::cuda::is_available()) {
            std::fprintf(stderr,"[setup] CUDA requested but unavailable -> CPU\n");
        }
        S.use_gpu = false;
        S.device = torch::kCPU;
    }

#else

    if (S.use_gpu) {
        std::fprintf(stderr,
            "[setup] CUDA requested, but this build was compiled without KEY_CUDA=1 -> CPU\n");
    }

    // In a non-CUDA build, always force CPU.
    S.use_gpu = false;
    S.device = torch::kCPU;

#endif

    // Meta first
    int n_qm_cfg=0, max_mm_cfg=0, n_types_opt=-1;
    std::vector<int> species;
    if (!parse_meta(dpmm_ctrlname, n_qm_cfg, max_mm_cfg, species, n_types_opt)) return;

    try {
        S.model = torch::jit::load(std::string(dpmm_ptname), S.device);
        S.model.eval();
    
        // Cache / return meta
        S.model_path = dpmm_ptname;
        S.ctrl_path  = dpmm_ctrlname;
        S.n_qm_max   = n_qm_cfg;
        S.mm_max     = max_mm_cfg;
        S.species_z  = species;
    
        if (dpmm_types) {
            for (size_t i = 0; i < species.size(); ++i) {
                dpmm_types[i] = species[i];
            }
        }
    
        if (dpmm_ntypes)
            *dpmm_ntypes = static_cast<int>(species.size());
    
        if (dpmm_qmmax)
            *dpmm_qmmax = (n_qm_cfg > 0 ? n_qm_cfg : 0);
    
        if (dpmm_mmmax)
            *dpmm_mmmax = max_mm_cfg;
    
        S.initialized = true;

        // Everything succeeded.
        if (dpmm_setup_err) *dpmm_setup_err = 0;

    }
    catch (const c10::Error& e) {
        std::fprintf(stderr, "[setup] DPMM Torch setup failed: %s\n", e.what());
        S.initialized = false;
        return;
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "[setup] DPMM setup failed: %s\n", e.what());
        S.initialized = false;
        return;
    }


}
// -------------------------------
// internal (C binding)
// -------------------------------
void mndo97_dpmm_internal(
    int dpmm_nqm, int dpmm_nmm, int dpmm_mmax,
    const int* /*dpmm_types_in*/, int /*dpmm_use_gpu*/, int /*dpmm_use_omp*/,
    const double* dpmm_qmx, const double* dpmm_qmy, const double* dpmm_qmz,
    const int*    dpmm_qmatomz,
    const double* dpmm_mmx, const double* dpmm_mmy, const double* dpmm_mmz,
    const double* dpmm_mmcg,
    double* dpmm_e,
    double* dpmm_qmdx, double* dpmm_qmdy, double* dpmm_qmdz,
    double* dpmm_mmdx, double* dpmm_mmdy, double* dpmm_mmdz)
{
    auto& S = cache();
    if (!S.initialized) { std::fprintf(stderr,"[internal] called before setup\n"); return; }
    if (dpmm_nqm<=0) { std::fprintf(stderr,"[internal] nqm<=0\n"); return; }
    if (dpmm_nmm<0 || dpmm_mmax<=0 || dpmm_nmm>dpmm_mmax) {
        std::fprintf(stderr,"[internal] bad nmm/mmax (nmm=%d, mmax=%d)\n", dpmm_nmm, dpmm_mmax); return;
    }

    // Zero all outputs up front, so any early return below (Z-not-found, shape
    // mismatch, or a caught Torch error) leaves defined zero values instead of
    // the previous step's data / uninitialized memory.
    if (dpmm_e) *dpmm_e = 0.0;
    for (int i=0;i<dpmm_nqm;++i){ dpmm_qmdx[i]=0.0; dpmm_qmdy[i]=0.0; dpmm_qmdz[i]=0.0; }
    for (int i=0;i<dpmm_mmax;++i){ dpmm_mmdx[i]=0.0; dpmm_mmdy[i]=0.0; dpmm_mmdz[i]=0.0; }

    const int64_t Nq=dpmm_nqm, Nm=dpmm_nmm, Mmax=dpmm_mmax;

    // Map Z → type ids
    std::vector<int64_t> atom_types_vec(Nq);
    for (int64_t i=0;i<Nq;++i) {
        int64_t t = z_to_type(S.species_z, dpmm_qmatomz[i]);
        if (t<0) { std::fprintf(stderr,"[internal] Z=%d not in species_z_ordered\n", dpmm_qmatomz[i]); return; }
        atom_types_vec[i]=t;
    }

    // ---- Build inputs on CPU, fill, then move to S.device ----
    // Outer guard: no Torch (c10::Error) or std exception from tensor
    // construction, the .to(device) transfers, model.forward, or the output
    // fetches below may escape across the bind(C) boundary into Fortran.
    try {
    auto f32cpu = at::TensorOptions().dtype(at::kFloat).device(torch::kCPU);
    auto i64cpu = at::TensorOptions().dtype(at::kLong).device(torch::kCPU);

    at::Tensor qm_coords_cpu = torch::empty({Nq,3}, f32cpu);
    {
        auto qc = qm_coords_cpu.accessor<float,2>();
        for (int64_t i=0;i<Nq;++i){ qc[i][0]=float(dpmm_qmx[i]); qc[i][1]=float(dpmm_qmy[i]); qc[i][2]=float(dpmm_qmz[i]); }
    }
    at::Tensor atom_types_cpu = torch::from_blob(atom_types_vec.data(), {Nq}, i64cpu).clone();

    at::Tensor mm_coords_cpu = torch::zeros({Mmax,3}, f32cpu);
    at::Tensor mm_Q_cpu      = torch::zeros({Mmax},   f32cpu);
    at::Tensor mm_type_cpu   = torch::zeros({Mmax},   f32cpu);
    {
        auto mc = mm_coords_cpu.accessor<float,2>();
        for (int64_t i=0;i<Nm;++i){ mc[i][0]=float(dpmm_mmx[i]); mc[i][1]=float(dpmm_mmy[i]); mc[i][2]=float(dpmm_mmz[i]); }
        auto mq = mm_Q_cpu.accessor<float,1>();
        for (int64_t i=0;i<Nm;++i) mq[i]=float(dpmm_mmcg[i]);
        if (Nm>0) mm_type_cpu.narrow(0,0,Nm).fill_(1.0f); // pads stay 0
    }

    // Move to device & add batch dim
    at::Tensor qm_coords  = qm_coords_cpu.to(S.device).unsqueeze(0);
    at::Tensor atom_types = atom_types_cpu.to(S.device).unsqueeze(0);
    at::Tensor mm_coords  = mm_coords_cpu.to(S.device).unsqueeze(0);
    at::Tensor mm_Q       = mm_Q_cpu.to(S.device).unsqueeze(0);
    at::Tensor mm_type    = mm_type_cpu.to(S.device).unsqueeze(0);

    std::vector<c10::IValue> inputs; inputs.reserve(5);
    inputs.emplace_back(qm_coords);
    inputs.emplace_back(atom_types);
    inputs.emplace_back(mm_coords);
    inputs.emplace_back(mm_Q);
    inputs.emplace_back(mm_type);

    c10::IValue out_iv;
    try { out_iv = S.model.forward(inputs); }
    catch (const c10::Error& e) { std::fprintf(stderr,"[internal] model.forward failed: %s\n", e.what()); return; }

    if (!out_iv.isGenericDict()) { std::fprintf(stderr,"[internal] model output is not a dict\n"); return; }
    auto out_dict = out_iv.toGenericDict();

    constexpr double EV_TO_KCALMOL = 23.060547830619;

    // Energy
    try {
        at::Tensor eT = get_energy_cpu(out_dict); // scalar on CPU
        const double e_eV = eT.item<double>();
        if (dpmm_e) *dpmm_e = e_eV * EV_TO_KCALMOL;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"[internal] energy fetch error: %s\n", e.what()); return;
    }

    // QM grads (B,Nq,3)
    try {
        at::Tensor qgT = get_f32_cpu_3d(out_dict, {"qm_grad","qm_dgrad","qm_grad_high","qm_grad_low"});
        if (qgT.size(1) != Nq || qgT.size(2) != 3) {
            std::fprintf(stderr,"[internal] qm_grad shape mismatch: got (%ld,%ld,%ld), want (*,%ld,3)\n",
                         (long)qgT.size(0),(long)qgT.size(1),(long)qgT.size(2),(long)Nq);
            return;
        }
        auto qg = qgT.accessor<float,3>();
        for (int64_t i=0;i<Nq;++i) {
            dpmm_qmdx[i] = double(qg[0][i][0]) * EV_TO_KCALMOL;
            dpmm_qmdy[i] = double(qg[0][i][1]) * EV_TO_KCALMOL;
            dpmm_qmdz[i] = double(qg[0][i][2]) * EV_TO_KCALMOL;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr,"[internal] qm_grad fetch error: %s\n", e.what()); return;
    }

    // MM forces via mm_espgrad_d (B,Mmax,3) * q   (pads untouched)
    try {
        at::Tensor egT = get_f32_cpu_3d(out_dict, {"mm_espgrad_d"});
        if (egT.size(1) < Mmax || egT.size(2) != 3) {
            std::fprintf(stderr,"[internal] mm_espgrad_d shape mismatch: got (%ld,%ld,%ld), expect (*,%ld,3)\n",
                         (long)egT.size(0),(long)egT.size(1),(long)egT.size(2),(long)Mmax);
            return;
        }
        auto eg = egT.accessor<float,3>();
        for (int64_t i=0;i<Nm;++i) {
            const double q = dpmm_mmcg[i]; // zero q → zero force
            dpmm_mmdx[i] = double(eg[0][i][0]) * q * EV_TO_KCALMOL;
            dpmm_mmdy[i] = double(eg[0][i][1]) * q * EV_TO_KCALMOL;
            dpmm_mmdz[i] = double(eg[0][i][2]) * q * EV_TO_KCALMOL;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr,"[internal] mm_espgrad_d fetch error: %s\n", e.what()); return;
    }
    } // end outer guard
    catch (const c10::Error& e) { std::fprintf(stderr,"[internal] torch error: %s\n", e.what()); return; }
    catch (const std::exception& e) { std::fprintf(stderr,"[internal] error: %s\n", e.what()); return; }
}

} // extern "C"

#endif
#endif
#endif