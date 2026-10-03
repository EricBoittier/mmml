#if KEY_MLMM==1 

// type_pyth_custom.cxx
// Persistent Python-socket ML-only backend for CHARMM MLMM.
//
// Runtime model:
//   setup():
//     - cache ML atom metadata from Fortran
//     - create a unique Unix-domain socket path
//     - launch RUNF Python script once
//     - connect to the Python server
//     - send SETUP metadata once
//   force():
//     - send compact ML coordinates every energy/gradient call
//     - receive energy + compact ML gradients
//
// Units expected by this backend:
//   coordinates sent to Python : Angstrom
//   energy returned by Python  : kcal/mol
//   gradients returned         : kcal/mol/Angstrom, dE/dR, not force
//
// IMPORTANT:
//   - The startup sleep/retry loop happens only during setup while waiting
//     for Python to bind the socket. It does not change MD physics.
//   - The force call is synchronous/blocking. A slow Python model only slows
//     wall-clock speed; it does not change the MD time step or trajectory,
//     unless you intentionally return bad/zero gradients on failure.
//   - Therefore force communication failures are fatal here, not silent zeros.

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>

#include <string>
#include <vector>

#include <poll.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace {

constexpr int32_t MLPS_MAGIC   = 0x53504C4D;  // bytes "MLPS" on little endian
constexpr int32_t MLPS_VERSION = 1;
constexpr int32_t CMD_SETUP    = 1;
constexpr int32_t CMD_FORCE    = 2;
constexpr int32_t CMD_STOP     = 3;
constexpr int32_t STATUS_OK    = 0;

struct PySocketState {
    bool active = false;

    int use_gpu = -1;
    int nml = 0;
    int natoms = 0;

    pid_t child_pid = -1;
    int sock_fd = -1;

    std::string spec_file;
    std::string runf_file;
    std::string socket_path;

    std::vector<int32_t> ml_idx;     // CHARMM 1-based atom indices, metadata only
    std::vector<int32_t> ml_zid;     // atomic numbers
    std::vector<int32_t> ml_maskid;  // natom-sized ML mask, 1/0
};

PySocketState S;

int getenv_int(const char* key, int fallback) {
    const char* v = std::getenv(key);
    if (!v || !*v) return fallback;
    char* end = nullptr;
    long x = std::strtol(v, &end, 10);
    if (end == v || x <= 0 || x > INT_MAX) return fallback;
    return static_cast<int>(x);
}

[[noreturn]] void die(const char* msg) {
    std::fprintf(stderr, "[PYTH] FATAL: %s\n", msg);
    std::fflush(stderr);
    std::abort();
}

[[noreturn]] void die_errno(const char* msg) {
    std::fprintf(stderr, "[PYTH] FATAL: %s: %s\n", msg, std::strerror(errno));
    std::fflush(stderr);
    std::abort();
}

bool poll_fd(int fd, short events, int timeout_ms, const char* what) {
    pollfd pfd{};
    pfd.fd = fd;
    pfd.events = events;

    while (true) {
        int rc = ::poll(&pfd, 1, timeout_ms);
        if (rc > 0) {
            if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
                std::fprintf(stderr, "[PYTH] socket error while waiting for %s; revents=%d\n",
                             what, static_cast<int>(pfd.revents));
                return false;
            }
            return (pfd.revents & events) != 0;
        }
        if (rc == 0) {
            std::fprintf(stderr, "[PYTH] timeout while waiting for %s\n", what);
            return false;
        }
        if (errno == EINTR) continue;
        std::fprintf(stderr, "[PYTH] poll failed while waiting for %s: %s\n",
                     what, std::strerror(errno));
        return false;
    }
}

bool write_all(int fd, const void* buf, size_t nbytes, const char* what) {
    const char* p = static_cast<const char*>(buf);
    size_t done = 0;

    const int timeout_ms = getenv_int("MLPS_PY_IO_TIMEOUT_MS", 600000); // default 10 min

    while (done < nbytes) {
        if (!poll_fd(fd, POLLOUT, timeout_ms, what)) return false;

        ssize_t rc = ::write(fd, p + done, nbytes - done);
        if (rc < 0) {
            if (errno == EINTR) continue;
            std::fprintf(stderr, "[PYTH] write failed for %s: %s\n", what, std::strerror(errno));
            return false;
        }
        if (rc == 0) return false;
        done += static_cast<size_t>(rc);
    }
    return true;
}

bool read_all(int fd, void* buf, size_t nbytes, const char* what) {
    char* p = static_cast<char*>(buf);
    size_t done = 0;

    const int timeout_ms = getenv_int("MLPS_PY_IO_TIMEOUT_MS", 600000); // default 10 min

    while (done < nbytes) {
        if (!poll_fd(fd, POLLIN, timeout_ms, what)) return false;

        ssize_t rc = ::read(fd, p + done, nbytes - done);
        if (rc < 0) {
            if (errno == EINTR) continue;
            std::fprintf(stderr, "[PYTH] read failed for %s: %s\n", what, std::strerror(errno));
            return false;
        }
        if (rc == 0) {
            std::fprintf(stderr, "[PYTH] EOF while reading %s\n", what);
            return false;
        }
        done += static_cast<size_t>(rc);
    }
    return true;
}

template <typename T>
void send_scalar(int fd, T v, const char* what) {
    if (!write_all(fd, &v, sizeof(T), what)) die("socket write failed");
}

template <typename T>
T recv_scalar(int fd, const char* what) {
    T v{};
    if (!read_all(fd, &v, sizeof(T), what)) die("socket read failed");
    return v;
}

void send_bytes(int fd, const void* ptr, size_t nbytes, const char* what) {
    if (nbytes == 0) return;
    if (!write_all(fd, ptr, nbytes, what)) die("socket write failed");
}

void recv_bytes(int fd, void* ptr, size_t nbytes, const char* what) {
    if (nbytes == 0) return;
    if (!read_all(fd, ptr, nbytes, what)) die("socket read failed");
}

void send_string(int fd, const std::string& s, const char* what) {
    if (s.size() > static_cast<size_t>(INT32_MAX)) die("string too long");
    int32_t n = static_cast<int32_t>(s.size());
    send_scalar<int32_t>(fd, n, what);
    if (n > 0) send_bytes(fd, s.data(), static_cast<size_t>(n), what);
}

void close_socket() {
    if (S.sock_fd >= 0) {
        ::close(S.sock_fd);
        S.sock_fd = -1;
    }
}

void cleanup_child(bool send_stop) {
    S.active = false;

    if (S.sock_fd >= 0 && send_stop) {
        const int32_t magic = MLPS_MAGIC;
        const int32_t version = MLPS_VERSION;
        const int32_t cmd = CMD_STOP;
        (void)write_all(S.sock_fd, &magic, sizeof(magic), "STOP magic");
        (void)write_all(S.sock_fd, &version, sizeof(version), "STOP version");
        (void)write_all(S.sock_fd, &cmd, sizeof(cmd), "STOP cmd");
    }

    close_socket();

    if (!S.socket_path.empty()) {
        ::unlink(S.socket_path.c_str());
        S.socket_path.clear();
    }

    if (S.child_pid > 0) {
        int status = 0;
        pid_t rc = ::waitpid(S.child_pid, &status, WNOHANG);
        if (rc == 0) {
            ::kill(S.child_pid, SIGTERM);

            // Shutdown grace period only affects cleanup/reinitialization,
            // never normal MD integration.
            for (int i = 0; i < 20; ++i) {
                rc = ::waitpid(S.child_pid, &status, WNOHANG);
                if (rc == S.child_pid) break;
                ::usleep(50000); // 50 ms
            }

            if (rc == 0) {
                ::kill(S.child_pid, SIGKILL);
                (void)::waitpid(S.child_pid, &status, 0);
            }
        }
        S.child_pid = -1;
    }
}

std::string make_socket_path() {
    // getpid() is rank-local process id. If only rank 0 launches PYTH, this is unique enough.
    // If you later allow multiple workers, include rank/backend in the path too.
    return "/tmp/charmm_mlps_pyth_" + std::to_string(static_cast<long>(::getpid())) + ".sock";
}

void launch_python_server() {
    if (S.runf_file.empty()) die("RUNF script is empty for PYTH backend");

    S.socket_path = make_socket_path();
    ::unlink(S.socket_path.c_str());

    std::string gpu_arg = std::to_string(S.use_gpu);

    // Important: argv points to strings that remain alive until posix_spawnp returns.
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>("python3"));
    argv.push_back(const_cast<char*>(S.runf_file.c_str()));
    argv.push_back(const_cast<char*>("--socket"));
    argv.push_back(const_cast<char*>(S.socket_path.c_str()));
    argv.push_back(const_cast<char*>("--gpu"));
    argv.push_back(const_cast<char*>(gpu_arg.c_str()));
    if (!S.spec_file.empty()) {
        argv.push_back(const_cast<char*>("--spec"));
        argv.push_back(const_cast<char*>(S.spec_file.c_str()));
    }
    argv.push_back(nullptr);

    pid_t pid = -1;
    int rc = ::posix_spawnp(&pid, "python3", nullptr, nullptr, argv.data(), environ);
    if (rc != 0) {
        errno = rc;
        die_errno("posix_spawnp python3 failed");
    }

    S.child_pid = pid;
}

void connect_with_retry() {
    // This sleep/retry loop only waits for Python to finish importing libraries
    // and bind the Unix socket during setup. It does NOT run during force calls.
    //
    // Heavy models such as UMA may take longer than 30 s to import/load, so the
    // total startup timeout is configurable:
    //   export MLPS_PY_STARTUP_TIMEOUT_SEC=300
    const int startup_timeout_sec = getenv_int("MLPS_PY_STARTUP_TIMEOUT_SEC", 300);
    const int sleep_us = getenv_int("MLPS_PY_STARTUP_SLEEP_US", 100000); // 0.1 s
    const int max_tries = std::max(1, (startup_timeout_sec * 1000000) / sleep_us);

    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) die_errno("socket(AF_UNIX) failed");

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (S.socket_path.size() >= sizeof(addr.sun_path)) {
        ::close(fd);
        die("socket path is too long");
    }
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", S.socket_path.c_str());

    for (int t = 0; t < max_tries; ++t) {
        int rc = ::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        if (rc == 0) {
            S.sock_fd = fd;
            return;
        }

        // Expected early errors are ENOENT/ECONNREFUSED while Python starts.
        // Other errors are still retried briefly, but child death is checked.
        if (S.child_pid > 0) {
            int status = 0;
            pid_t w = ::waitpid(S.child_pid, &status, WNOHANG);
            if (w == S.child_pid) {
                std::fprintf(stderr,
                             "[PYTH] Python server exited before socket connection. status=%d\n",
                             status);
                ::close(fd);
                S.child_pid = -1;
                die("Python server failed during startup");
            }
        }

        ::usleep(static_cast<useconds_t>(sleep_us));
    }

    ::close(fd);
    die("timed out connecting to Python socket server");
}

void send_setup_message() {
    const int32_t magic = MLPS_MAGIC;
    const int32_t version = MLPS_VERSION;
    const int32_t cmd = CMD_SETUP;
    const int32_t natoms = static_cast<int32_t>(S.natoms);
    const int32_t nml = static_cast<int32_t>(S.nml);
    const int32_t gpu = static_cast<int32_t>(S.use_gpu);

    send_scalar<int32_t>(S.sock_fd, magic, "SETUP magic");
    send_scalar<int32_t>(S.sock_fd, version, "SETUP version");
    send_scalar<int32_t>(S.sock_fd, cmd, "SETUP cmd");
    send_scalar<int32_t>(S.sock_fd, natoms, "SETUP natoms");
    send_scalar<int32_t>(S.sock_fd, nml, "SETUP nml");
    send_scalar<int32_t>(S.sock_fd, gpu, "SETUP gpu");
    send_string(S.sock_fd, S.spec_file, "SETUP spec");

    send_bytes(S.sock_fd, S.ml_idx.data(),    static_cast<size_t>(S.nml)    * sizeof(int32_t), "SETUP ml_idx");
    send_bytes(S.sock_fd, S.ml_zid.data(),    static_cast<size_t>(S.nml)    * sizeof(int32_t), "SETUP ml_zid");
    send_bytes(S.sock_fd, S.ml_maskid.data(), static_cast<size_t>(S.natoms) * sizeof(int32_t), "SETUP ml_maskid");

    int32_t status = recv_scalar<int32_t>(S.sock_fd, "SETUP status");
    if (status != STATUS_OK) {
        std::fprintf(stderr, "[PYTH] Python SETUP failed with status=%d\n",
                     static_cast<int>(status));
        die("Python setup failed");
    }
}

} // namespace

extern "C" void charmm_pyth_internal_setup_custom(
    int pyth_in_use,
    int pyth_use_gpu,
    const char* spec_name,
    const char* runf_name,
    int pyth_pt_nml,
    const int* pyth_in_mlidx,
    const int* pyth_in_mlZid,
    const int* pyth_in_mlmaskid,
    int pyth_in_natoms,
    int* pythcstm_out_setup_err)
{
    // Assume failure until setup completes successfully.
    if (pythcstm_out_setup_err) *pythcstm_out_setup_err = 1;

    if (pyth_in_use != 1) {
        std::fprintf(stderr, "[PYTH] setup called with pyth_in_use=%d; disabling backend.\n", pyth_in_use);
        cleanup_child(false);
        return;
    }
    if (pyth_pt_nml <= 0) {
        std::fprintf(stderr,"[PYTH] setup failed: requires nml > 0\n");
        cleanup_child(false);
        return;
    }
    if (pyth_in_natoms <= 0) {
        std::fprintf(stderr,"[PYTH] setup failed: requires natoms > 0\n");
        cleanup_child(false);
        return;
    }
    if (!pyth_in_mlidx || !pyth_in_mlZid || !pyth_in_mlmaskid) {
        std::fprintf(stderr,"[PYTH] setup failed: received null arrays\n");
        cleanup_child(false);
        return;
    }
    if (!runf_name || std::strlen(runf_name) == 0) {
        std::fprintf(stderr,"[PYTH] setup failed: requires RUNF python script file\n");
        cleanup_child(false);
        return;
    }

    try {
        cleanup_child(true);

        S.active = false;
        S.use_gpu = pyth_use_gpu;
        S.nml = pyth_pt_nml;
        S.natoms = pyth_in_natoms;
        S.spec_file = (spec_name && std::strlen(spec_name) > 0) ? std::string(spec_name) : std::string();
        S.runf_file = std::string(runf_name);

        S.ml_idx.assign(pyth_in_mlidx, pyth_in_mlidx + S.nml);
        S.ml_zid.assign(pyth_in_mlZid, pyth_in_mlZid + S.nml);
        S.ml_maskid.assign(pyth_in_mlmaskid, pyth_in_mlmaskid + S.natoms);

        std::fprintf(stderr, "[PYTH] launching Python server: RUNF=%s SPEC=%s NML=%d NATOM=%d GPU=%d\n",
                 S.runf_file.c_str(),
                 S.spec_file.empty() ? "<none>" : S.spec_file.c_str(),
                 S.nml, S.natoms, S.use_gpu);

        launch_python_server();
        connect_with_retry();
        send_setup_message();

        S.active = true;

        // Setup completed successfully.
        if (pythcstm_out_setup_err) *pythcstm_out_setup_err = 0;

        std::fprintf(stderr, "[PYTH] Python socket backend ready: %s\n", S.socket_path.c_str());
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "[setup] setup failed: %s\n", e.what());
        S.active = false;
        cleanup_child(true);
        return;
    }
    catch (...) {
        std::fprintf(stderr, "[setup] setup failed: unknown exception\n");
        S.active = false;
        cleanup_child(true);
        return;
    }
}

extern "C" void charmm_pyth_internal_force_custom(
    double* E_pyth_c,
    const double* ml_x_c,
    const double* ml_y_c,
    const double* ml_z_c,
    double* ml_dx_c,
    double* ml_dy_c,
    double* ml_dz_c)
{
    if (!E_pyth_c || !ml_x_c || !ml_y_c || !ml_z_c ||
        !ml_dx_c || !ml_dy_c || !ml_dz_c) {
        die("PYTH force received null pointer");
    }

    *E_pyth_c = 0.0;

    if (!S.active || S.sock_fd < 0) {
        // Fatal is safer than returning zero gradients, because zero gradients
        // would silently corrupt dynamics.
        die("PYTH force called before active setup");
    }

    const int32_t magic = MLPS_MAGIC;
    const int32_t version = MLPS_VERSION;
    const int32_t cmd = CMD_FORCE;
    const int32_t nml = static_cast<int32_t>(S.nml);

    send_scalar<int32_t>(S.sock_fd, magic, "FORCE magic");
    send_scalar<int32_t>(S.sock_fd, version, "FORCE version");
    send_scalar<int32_t>(S.sock_fd, cmd, "FORCE cmd");
    send_scalar<int32_t>(S.sock_fd, nml, "FORCE nml");

    // Compact ML coordinates as split x/y/z arrays to match Fortran storage.
    send_bytes(S.sock_fd, ml_x_c, static_cast<size_t>(S.nml) * sizeof(double), "FORCE ml_x");
    send_bytes(S.sock_fd, ml_y_c, static_cast<size_t>(S.nml) * sizeof(double), "FORCE ml_y");
    send_bytes(S.sock_fd, ml_z_c, static_cast<size_t>(S.nml) * sizeof(double), "FORCE ml_z");

    int32_t status = recv_scalar<int32_t>(S.sock_fd, "FORCE status");
    if (status != STATUS_OK) {
        std::fprintf(stderr, "[PYTH] Python FORCE failed with status=%d\n",
                     static_cast<int>(status));
        die("Python force failed");
    }
    recv_bytes(S.sock_fd, E_pyth_c, sizeof(double), "FORCE energy");
    recv_bytes(S.sock_fd, ml_dx_c, static_cast<size_t>(S.nml) * sizeof(double), "FORCE ml_dx");
    recv_bytes(S.sock_fd, ml_dy_c, static_cast<size_t>(S.nml) * sizeof(double), "FORCE ml_dy");
    recv_bytes(S.sock_fd, ml_dz_c, static_cast<size_t>(S.nml) * sizeof(double), "FORCE ml_dz");
}

#endif // KEY_MLMM==1