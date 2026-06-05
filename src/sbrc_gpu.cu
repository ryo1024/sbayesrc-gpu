// sbrc_gpu.cu — Phase 2 milestone-1 GPU implementation of SBayesRC inner Gibbs sweep.
//
// Reproduces ApproxBayesRC::SnpEffects::sampleFromFC_eigen (model.cpp:5949-6097)
// with these differences:
//   - SNP order randomized on device via cuRAND (different sequence than CPU's shuffle_index)
//   - mixture-component categorical sampling uses device-generated uniform draws (not host)
//   - reductions in float32 may differ in summation order vs CPU (within 1 ULP per op)
//
// Validated distributionally in Phase 1; β-sparsity and ‖β‖² agree with CPU within MC error.

#include "sbrc_gpu.hpp"
#include "data.hpp"     // for LDBlockInfo (only used opaquely)

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <omp.h>
#include <chrono>
#include <mutex>
#include <cublas_v2.h>

// ─── Per-stage cumulative dispatch timing (printed at process exit) ────────
struct DispatchTiming {
    double upload = 0;
    double kernel = 0;
    double download = 0;
    double agg = 0;
    double total = 0;
    int calls = 0;
    ~DispatchTiming() {
        if (calls == 0) return;
        std::fprintf(stderr, "\n[sbrc_gpu] dispatch totals: %d calls\n", calls);
        std::fprintf(stderr, "  upload (h2d + transpose):  %.2f s  (avg %.3f s/call)\n", upload, upload/calls);
        std::fprintf(stderr, "  kernel:                    %.2f s  (avg %.3f s/call)\n", kernel, kernel/calls);
        std::fprintf(stderr, "  download (d2h + transpose):%.2f s  (avg %.3f s/call)\n", download, download/calls);
        std::fprintf(stderr, "  host aggregation:          %.2f s  (avg %.3f s/call)\n", agg, agg/calls);
        std::fprintf(stderr, "  total in dispatch:         %.2f s  (avg %.3f s/call)\n", total, total/calls);
    }
};
static DispatchTiming g_dispatch_timing;
using clock_now = std::chrono::steady_clock;

bool sbrc_gpu_enabled = false;

#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    std::abort(); } } while (0)

struct BlockMeta {
    int N;
    int q;
    long long QOffset;     // offset into flat Q buffer
    int localSnpOffset;    // offset into packed per-block buffers (perm, urnd, nrnd)
    int globalSnpStart;    // global SNP index for this block's first SNP (β, pip, membership, snpPi)
    int qOffset;           // offset into wcorr/what
    float vareDn;          // updated each sweep
};

struct SbrcGpuImpl {
    int nBlocks       = 0;
    int totalSnps     = 0;       // GLOBAL SNP count (size of β / pip / membership / snpPi)
    int localSnpTotal = 0;       // sum of N across blocks (size of packed perm/urnd/nrnd)
    int totalQDim     = 0;       // sum of q across blocks (size of wcorr/what)
    long long totalQElems = 0;   // sum of N*q (size of Q_flat)
    int ndist         = 4;
    int maxQ          = 0;

    // Device buffers (persistent)
    float* d_Q = nullptr;
    float* d_beta = nullptr;
    float* d_pip = nullptr;
    float* d_wcorr = nullptr;
    float* d_what  = nullptr;
    int* d_membership = nullptr;
    int* d_perm = nullptr;
    float* d_urnd = nullptr;
    float* d_nrnd = nullptr;
    BlockMeta* d_meta = nullptr;
    float* d_snpPi = nullptr;       // m × ndist
    float* d_gamma = nullptr;       // ndist
    float* d_deltaPi = nullptr;     // m × ndist (output)
    int*   d_badSnps = nullptr;     // length totalSnps; non-zero = skip SNP (force β=0)
    curandState* d_rng = nullptr;   // one per CUDA thread for RNG inside kernel

    // Host-side mirror of meta
    std::vector<BlockMeta> h_meta;

    // Pinned host buffers — direct row-major mirror of device snpPi/deltaPi,
    // populated by OMP-parallel transposes (no per-iter alloc).
    float* h_snpPi_pinned = nullptr;     // m × ndist row-major
    float* h_deltaPi_pinned = nullptr;   // m × ndist row-major
    float* h_wcorr_pinned = nullptr;     // totalQDim
    float* h_what_pinned = nullptr;      // totalQDim
    int*   h_membership_pinned = nullptr; // totalSnps

    bool initialized = false;
    bool owns_Q = false;     // false → d_Q is shared from g_q_cache; do not free.
    bool gpu_failed = false; // true → cudaMalloc OOMed for this caller; fall back to CPU

    // Per-impl mutex serializing sweep() calls. Multiple ApproxBayesR instances run
    // concurrently via MultiModelSBayesR's OMP loop, but each instance's GPU state
    // must be sequentially accessed (host bookkeeping like d_meta isn't thread-safe).
    std::mutex sweep_mutex;

    // Per-impl CUDA stream. Multi-chain runs need this so different chains' kernels
    // can overlap on the GPU (default stream forces serialization). Created lazily.
    cudaStream_t stream = 0;
};

// Shared Q cache keyed on first-block data pointer + size, with reference counting.
// When an impl's caller_id is released (SnpEffects destructor), decrement refcount;
// free the device buffer when it hits zero. Avoids both the use-after-free race of
// proactive eviction and the OOM accumulation of pure leak-on-overwrite.
struct QCacheEntry {
    float* d_Q = nullptr;
    long long totalQElems = 0;
    int refcount = 0;
};
// Composite key: (data pointer, totalQElems). Two different Qblocks vectors can have
// identical pointers if Eigen reuses storage, so size disambiguates.
struct QKey {
    const float* ptr;
    long long size;
    bool operator==(const QKey& o) const { return ptr == o.ptr && size == o.size; }
};
struct QKeyHash {
    size_t operator()(const QKey& k) const {
        return std::hash<const float*>()(k.ptr) ^ std::hash<long long>()(k.size);
    }
};
static std::unordered_map<QKey, QCacheEntry, QKeyHash> g_q_cache;

// Mutex protecting g_q_cache + g_sbrc_states. MultiModelSBayesR's OMP-parallel model loop
// concurrently calls get_or_init_state and sbrc_gpu_release from multiple threads.
static std::mutex g_state_mutex;

// ─── RNG init kernel ───────────────────────────────────────────────────────
__global__ void init_rng_kernel(curandState* states, uint64_t seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) curand_init(seed, idx, 0, &states[idx]);
}

// ─── Pre-RNG kernel: fills urnd, nrnd, perm (Fisher-Yates) ─────────────────
__global__ void pregen_rng_kernel(
    curandState* states, int totalSnps,
    float* __restrict__ urnd,
    float* __restrict__ nrnd,
    int* __restrict__ perm,
    const BlockMeta* __restrict__ meta, int nBlocks)
{
    int blkIdx = blockIdx.x;
    if (blkIdx >= nBlocks) return;
    int snpOff = meta[blkIdx].localSnpOffset;
    int N = meta[blkIdx].N;
    int tid = threadIdx.x;

    // urnd, nrnd: data-parallel
    int rngIdx = blkIdx * blockDim.x + tid;
    curandState s = states[rngIdx];
    for (int i = tid; i < N; i += blockDim.x) {
        urnd[snpOff + i] = curand_uniform(&s);
        nrnd[snpOff + i] = curand_normal(&s);
    }

    // Permutation: init to identity, then Fisher-Yates on thread 0
    for (int i = tid; i < N; i += blockDim.x) perm[snpOff + i] = i;
    __syncthreads();
    if (tid == 0) {
        for (int i = N - 1; i > 0; --i) {
            float u = curand_uniform(&s);                  // (0, 1]
            int j = (int)(u * (i + 1));
            if (j > i) j = i;
            int tmp = perm[snpOff + i];
            perm[snpOff + i] = perm[snpOff + j];
            perm[snpOff + j] = tmp;
        }
    }
    states[rngIdx] = s;
}

// ─── Main inner-sweep kernel ───────────────────────────────────────────────
// One CUDA threadblock per LD block, processes the full block's Gauss-Seidel sweep.
__global__ void block_sweep_kernel(
    const BlockMeta* __restrict__ meta,
    const float* __restrict__ Q_flat,
    float* __restrict__ beta_flat,
    float* __restrict__ wcorr_flat,
    float* __restrict__ what_flat,
    float* __restrict__ pip_flat,
    int* __restrict__ membership_flat,
    float* __restrict__ deltaPi_flat,        // m × ndist row-major (out)
    const int* __restrict__ badSnps_flat,    // length totalSnps; non-zero = skip
    const int* __restrict__ perm_flat,
    const float* __restrict__ urnd_flat,
    const float* __restrict__ nrnd_flat,
    const float* __restrict__ snpPi_flat,    // m × ndist row-major (in)
    const float* __restrict__ gamma_arr,     // ndist
    int ndist,
    float sigmaSq,
    float varg,
    int hsqPercModel)
{
    const int blkIdx = blockIdx.x;
    const BlockMeta m = meta[blkIdx];
    const int N = m.N;
    const int q = m.q;
    const float vareDn = m.vareDn;

    const float* Q   = Q_flat + m.QOffset;
    float* beta      = beta_flat + m.globalSnpStart;
    float* pip       = pip_flat  + m.globalSnpStart;
    float* wcorr_g   = wcorr_flat + m.qOffset;
    float* what_g    = what_flat + m.qOffset;
    const int* perm  = perm_flat + m.localSnpOffset;
    const float* urnd = urnd_flat + m.localSnpOffset;
    const float* nrnd = nrnd_flat + m.localSnpOffset;
    int* membership  = membership_flat + m.globalSnpStart;
    const float* snpPi = snpPi_flat + (long long)m.globalSnpStart * ndist;
    float* deltaPi = deltaPi_flat + (long long)m.globalSnpStart * ndist;
    const int* badSnps = badSnps_flat + m.globalSnpStart;

    extern __shared__ float smem[];
    // K1a: stage Qi in shared mem to eliminate the duplicate HBM read between
    // the dot-product step and the axpy step within each SNP iter. Adds q floats
    // per block of shared mem; total shmem = 3q + nwarps.
    float* wcorr   = smem;
    float* what    = smem + q;
    float* Qi_smem = smem + 2*q;
    float* sshared = smem + 3*q;

    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const unsigned mask = 0xffffffff;

    // Per-mixture scratch (NDIST = ndist ≤ 8 in practice)
    constexpr int MAXDIST = 8;
    float invWtdSigmaSq[MAXDIST], logWtdSigmaSq[MAXDIST];
    float invLhs[MAXDIST], logInvLhsMsigma[MAXDIST], sqrtInvLhs[MAXDIST];
    for (int k = 0; k < ndist; ++k) {
        float ws = gamma_arr[k] * (hsqPercModel && varg > 0 ? 0.01f * varg : sigmaSq);
        invWtdSigmaSq[k] = (ws > 0.0f) ? 1.0f / ws : 0.0f;
        logWtdSigmaSq[k] = (ws > 0.0f) ? logf(ws) : 0.0f;
        invLhs[k] = 1.0f / (vareDn + invWtdSigmaSq[k]);
        logInvLhsMsigma[k] = logf(invLhs[k]) - logWtdSigmaSq[k];
        sqrtInvLhs[k] = sqrtf(invLhs[k]);
    }

    for (int i = tid; i < q; i += blockSize) {
        wcorr[i] = wcorr_g[i];
        what[i]  = 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < N; ++t) {
        const int snp = perm[t];
        // Skip bad SNPs: force β=0, no wcorr update, no mixture sampling.
        if (badSnps[snp] != 0) {
            if (tid == 0) {
                beta[snp] = 0.0f;
                membership[snp] = 0;
                pip[snp] = 0.0f;
                float* dpiRow = deltaPi + (long long)snp * ndist;
                dpiRow[0] = 1.0f;
                for (int kk = 1; kk < ndist; ++kk) dpiRow[kk] = 0.0f;
            }
            __syncthreads();
            continue;
        }
        const float oldSample = beta[snp];
        const float* Qi_glob = Q + (size_t)snp * q;

        // K1a: fold the Qi-stage into the dot-product loop. Each thread reads
        // Qi[r] from HBM once, writes to Qi_smem, AND multiplies into the partial
        // dot. The existing __syncthreads() at the end of the reduction guarantees
        // Qi_smem is fully populated before the axpy step reads it. Saves the
        // duplicate HBM Qi read in axpy without adding a sync.
        float partial = 0.0f;
        for (int r = tid; r < q; r += blockSize) {
            const float qir = Qi_glob[r];
            Qi_smem[r] = qir;
            partial += qir * wcorr[r];
        }

        partial += __shfl_xor_sync(mask, partial, 16);
        partial += __shfl_xor_sync(mask, partial, 8);
        partial += __shfl_xor_sync(mask, partial, 4);
        partial += __shfl_xor_sync(mask, partial, 2);
        partial += __shfl_xor_sync(mask, partial, 1);
        const int warpId = tid >> 5;
        const int lane   = tid & 31;
        if (lane == 0) sshared[warpId] = partial;
        __syncthreads();

        if (warpId == 0) {
            const int nwarps = blockSize >> 5;
            float v = (tid < nwarps) ? sshared[tid] : 0.0f;
            v += __shfl_xor_sync(mask, v, 16);
            v += __shfl_xor_sync(mask, v, 8);
            v += __shfl_xor_sync(mask, v, 4);
            v += __shfl_xor_sync(mask, v, 2);
            v += __shfl_xor_sync(mask, v, 1);
            if (tid == 0) sshared[0] = v;
        }
        __syncthreads();
        const float dot = sshared[0];
        const float rhs = (dot + oldSample) * vareDn;

        const float* myPi = snpPi + (long long)snp * ndist;

        float uhat[MAXDIST], logDelta[MAXDIST];
        for (int k = 0; k < ndist; ++k) {
            uhat[k] = invLhs[k] * rhs;
            const float lpi = logf(fmaxf(1e-30f, myPi[k]));
            logDelta[k] = 0.5f * (logInvLhsMsigma[k] + uhat[k] * rhs) + lpi;
        }
        // Force k=0 (zero-effect) to depend only on its prior weight (matches GCTB)
        logDelta[0] = logf(fmaxf(1e-30f, myPi[0]));

        // Stable softmax → categorical
        float maxL = logDelta[0];
        for (int k = 1; k < ndist; ++k) if (logDelta[k] > maxL) maxL = logDelta[k];
        float Z = 0.0f, probDelta[MAXDIST];
        for (int k = 0; k < ndist; ++k) { probDelta[k] = __expf(logDelta[k] - maxL); Z += probDelta[k]; }
        const float u = urnd[snp] * Z;
        int delta = ndist - 1;
        {
            float cum = 0.0f;
            for (int k = 0; k < ndist; ++k) {
                cum += probDelta[k];
                if (u <= cum) { delta = k; break; }
            }
        }

        float newSample = (delta != 0) ? (uhat[delta] + nrnd[snp] * sqrtInvLhs[delta]) : 0.0f;
        if (tid == 0) {
            beta[snp] = newSample;
            membership[snp] = delta;
            pip[snp] = 1.0f - probDelta[0] / Z;
            // Write normalized probDelta[k] to deltaPi for this snp
            float* deltaPiRow = deltaPi + (long long)snp * ndist;
            for (int kk = 0; kk < ndist; ++kk) deltaPiRow[kk] = probDelta[kk] / Z;
        }

        if (delta != 0) {
            const float dw = oldSample - newSample;
            for (int r = tid; r < q; r += blockSize) {
                const float qir = Qi_smem[r];
                wcorr[r] += qir * dw;
                what[r]  += qir * newSample;
            }
        } else if (oldSample != 0.0f) {
            for (int r = tid; r < q; r += blockSize) wcorr[r] += Qi_smem[r] * oldSample;
        }
        __syncthreads();
    }

    for (int i = tid; i < q; i += blockSize) {
        wcorr_g[i] = wcorr[i];
        what_g[i]  = what[i];
    }
}

// ─── State management (keyed on caller_identity) ───────────────────────────
static std::unordered_map<const void*, SbrcGpuImpl*> g_sbrc_states;

static SbrcGpuImpl& get_or_init_state(
    const void* caller_id, int nBlocks,
    const std::vector<MatrixXf>& Qblocks,
    const std::vector<VectorXf>& wcorrBlocks_init,
    const std::vector<LDBlockInfo*>& keptLdBlockInfoVec,
    int totalSnps, int ndist)
{
    std::lock_guard<std::mutex> lk(g_state_mutex);
    auto it = g_sbrc_states.find(caller_id);
    if (it != g_sbrc_states.end()) return *it->second;

    SbrcGpuImpl* impl = new SbrcGpuImpl();
    g_sbrc_states[caller_id] = impl;
    impl->nBlocks   = nBlocks;
    impl->totalSnps = totalSnps;
    impl->ndist     = ndist;

    impl->h_meta.resize(nBlocks);
    long long Qoff = 0;
    int localSnpOff = 0;
    int qOff = 0;
    int maxQ = 0;
    impl->localSnpTotal = 0;
    for (int i = 0; i < nBlocks; ++i) {
        int qi = (int)Qblocks[i].rows();
        int Ni = (int)Qblocks[i].cols();
        impl->h_meta[i].N = Ni;
        impl->h_meta[i].q = qi;
        impl->h_meta[i].QOffset = Qoff;
        impl->h_meta[i].localSnpOffset = localSnpOff;
        impl->h_meta[i].qOffset = qOff;
        impl->h_meta[i].vareDn = 0.0f;
        impl->h_meta[i].globalSnpStart = keptLdBlockInfoVec[i]->startSnpIdx;
        // Sanity: block must fit inside global SNP space.
        if (impl->h_meta[i].globalSnpStart + Ni > totalSnps) {
            std::fprintf(stderr, "[sbrc_gpu] FATAL: block %d globalSnpStart=%d + N=%d exceeds totalSnps=%d\n",
                         i, impl->h_meta[i].globalSnpStart, Ni, totalSnps);
            std::abort();
        }
        Qoff += (long long)Ni * qi;
        localSnpOff += Ni;
        qOff += qi;
        if (qi > maxQ) maxQ = qi;
    }
    impl->totalQElems   = Qoff;
    impl->localSnpTotal = localSnpOff;
    impl->totalQDim     = qOff;
    impl->maxQ          = maxQ;

    std::fprintf(stderr, "[sbrc_gpu] init: nBlocks=%d  totalSnps=%d  totalQElems=%lld (%.2f GB)  maxQ=%d  ndist=%d\n",
                 nBlocks, totalSnps, impl->totalQElems, impl->totalQElems * 4.0 / 1e9, maxQ, ndist);
    std::fprintf(stderr, "[sbrc_gpu] first 3 blocks: ");
    for (int i = 0; i < std::min(3, nBlocks); ++i) {
        std::fprintf(stderr, "blk%d{gStart=%d,N=%d,q=%d} ",
                     i, impl->h_meta[i].globalSnpStart, impl->h_meta[i].N, impl->h_meta[i].q);
    }
    std::fprintf(stderr, "\n");

    // Reference-counted Q cache. On cache hit (same pointer + size), increment refcount.
    // On miss, allocate new + insert with refcount=1. Release is deferred to
    // sbrc_gpu_release() which decrements refcount and frees when it hits zero. This
    // avoids the use-after-free race of proactive eviction and the OOM growth of
    // pure leaking.
    QKey qk{Qblocks[0].data(), impl->totalQElems};
    auto qit = g_q_cache.find(qk);
    if (qit != g_q_cache.end()) {
        impl->d_Q = qit->second.d_Q;
        qit->second.refcount += 1;
        impl->owns_Q = false;
        std::fprintf(stderr, "[sbrc_gpu] reusing cached d_Q (%.2f GB) for caller %p (refcount=%d)\n",
                     impl->totalQElems * 4.0 / 1e9, caller_id, qit->second.refcount);
    } else {
        cudaError_t err = cudaMalloc(&impl->d_Q, sizeof(float) * impl->totalQElems);
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                "[sbrc_gpu] cudaMalloc(%.2f GB) failed (OOM); this caller will fall back to CPU.\n"
                "  (Common during findBestFitModel's eigen-cutoff sweep when prior 70-GB Q is still resident.)\n",
                impl->totalQElems * 4.0 / 1e9);
            cudaGetLastError();  // clear sticky error
            impl->d_Q = nullptr;
            impl->gpu_failed = true;
            impl->initialized = true;   // initialized but failed — return without re-trying
            return *impl;
        }
        impl->owns_Q = true;
        // The g_q_cache insertion happens after Q is actually uploaded (below).
    }
    CUDA_CHECK(cudaMalloc(&impl->d_beta, sizeof(float) * totalSnps));               // global-sized
    CUDA_CHECK(cudaMalloc(&impl->d_pip,  sizeof(float) * totalSnps));
    CUDA_CHECK(cudaMalloc(&impl->d_wcorr, sizeof(float) * impl->totalQDim));
    CUDA_CHECK(cudaMalloc(&impl->d_what,  sizeof(float) * impl->totalQDim));
    CUDA_CHECK(cudaMalloc(&impl->d_membership, sizeof(int) * totalSnps));           // global-sized
    CUDA_CHECK(cudaMalloc(&impl->d_perm, sizeof(int) * impl->localSnpTotal));       // packed
    CUDA_CHECK(cudaMalloc(&impl->d_urnd, sizeof(float) * impl->localSnpTotal));     // packed
    CUDA_CHECK(cudaMalloc(&impl->d_nrnd, sizeof(float) * impl->localSnpTotal));     // packed
    CUDA_CHECK(cudaMalloc(&impl->d_meta, sizeof(BlockMeta) * nBlocks));
    CUDA_CHECK(cudaMalloc(&impl->d_snpPi, sizeof(float) * (size_t)totalSnps * ndist));    // global
    CUDA_CHECK(cudaMalloc(&impl->d_gamma, sizeof(float) * ndist));
    CUDA_CHECK(cudaMalloc(&impl->d_deltaPi, sizeof(float) * (size_t)totalSnps * ndist));  // global
    CUDA_CHECK(cudaMalloc(&impl->d_badSnps, sizeof(int) * totalSnps));
    CUDA_CHECK(cudaMemset(impl->d_badSnps, 0, sizeof(int) * totalSnps));   // start all-zero (no bad SNPs)

    // Pinned host buffers (one-time alloc, reused every sweep)
    CUDA_CHECK(cudaMallocHost(&impl->h_snpPi_pinned,   sizeof(float) * (size_t)totalSnps * ndist));
    CUDA_CHECK(cudaMallocHost(&impl->h_deltaPi_pinned, sizeof(float) * (size_t)totalSnps * ndist));
    CUDA_CHECK(cudaMallocHost(&impl->h_wcorr_pinned,   sizeof(float) * impl->totalQDim));
    CUDA_CHECK(cudaMallocHost(&impl->h_what_pinned,    sizeof(float) * impl->totalQDim));
    CUDA_CHECK(cudaMallocHost(&impl->h_membership_pinned, sizeof(int) * totalSnps));

    // Upload Q (one-time per unique Qblocks); skipped if reused from cache.
    if (impl->owns_Q) {
        Qoff = 0;
        for (int i = 0; i < nBlocks; ++i) {
            size_t n = (size_t)impl->h_meta[i].N * impl->h_meta[i].q;
            CUDA_CHECK(cudaMemcpy(impl->d_Q + Qoff, Qblocks[i].data(),
                                  sizeof(float) * n, cudaMemcpyHostToDevice));
            Qoff += n;
        }
        QCacheEntry e;
        e.d_Q = impl->d_Q;
        e.totalQElems = impl->totalQElems;
        e.refcount = 1;
        g_q_cache[qk] = e;
    }

    // Upload initial wcorr (per-block, packed)
    {
        std::vector<float> wbuf(impl->totalQDim);
        int qOff2 = 0;
        for (int i = 0; i < nBlocks; ++i) {
            int qi = impl->h_meta[i].q;
            std::memcpy(wbuf.data() + qOff2, wcorrBlocks_init[i].data(), sizeof(float) * qi);
            qOff2 += qi;
        }
        CUDA_CHECK(cudaMemcpy(impl->d_wcorr, wbuf.data(), sizeof(float) * impl->totalQDim,
                              cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMemset(impl->d_beta, 0, sizeof(float) * totalSnps));
    CUDA_CHECK(cudaMemset(impl->d_pip,  0, sizeof(float) * totalSnps));
    CUDA_CHECK(cudaMemset(impl->d_what, 0, sizeof(float) * impl->totalQDim));
    CUDA_CHECK(cudaMemset(impl->d_membership, 0, sizeof(int) * totalSnps));

    // Set up dynamic shared memory ceiling
    int nwarps = 16;   // for 512-thread default
    size_t shmemBytes = sizeof(float) * (2 * maxQ + nwarps);
    if (shmemBytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(block_sweep_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmemBytes));
    }

    // RNG: one curandState per (block, thread). We use 512 threads/block so per-block 512 states.
    const int rngPerBlock = 512;
    CUDA_CHECK(cudaMalloc(&impl->d_rng, sizeof(curandState) * (size_t)nBlocks * rngPerBlock));
    init_rng_kernel<<<(nBlocks * rngPerBlock + 255) / 256, 256>>>(impl->d_rng, /*seed=*/4242ull, nBlocks * rngPerBlock);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Per-impl CUDA stream for multi-chain concurrency.
    CUDA_CHECK(cudaStreamCreate(&impl->stream));

    impl->initialized = true;
    return *impl;
}

// Release a caller's GPU state. Free per-impl buffers + decrement the d_Q refcount;
// when d_Q refcount hits zero, free the device Q. Called from SnpEffects destructor.
void sbrc_gpu_release(const void* caller_identity) {
    if (!sbrc_gpu_enabled) return;
    // Fast no-lock check: if the map is empty, skip. Avoids touching mutex during
    // global-static-destructor teardown when other globals (the mutex itself) may
    // already be invalid.
    if (g_sbrc_states.empty()) return;
    std::lock_guard<std::mutex> lk(g_state_mutex);
    auto it = g_sbrc_states.find(caller_identity);
    if (it == g_sbrc_states.end()) return;

    SbrcGpuImpl* p = it->second;
    if (!p) { g_sbrc_states.erase(it); return; }

    // Find the QCacheEntry that holds this d_Q and decrement its refcount.
    if (p->d_Q) {
        for (auto qit = g_q_cache.begin(); qit != g_q_cache.end(); ) {
            if (qit->second.d_Q == p->d_Q) {
                qit->second.refcount -= 1;
                if (qit->second.refcount <= 0) {
                    std::fprintf(stderr, "[sbrc_gpu] freeing d_Q (%.2f GB; last ref dropped)\n",
                                 qit->second.totalQElems * 4.0 / 1e9);
                    cudaFree(qit->second.d_Q);
                    qit = g_q_cache.erase(qit);
                } else {
                    std::fprintf(stderr, "[sbrc_gpu] decremented d_Q refcount to %d\n",
                                 qit->second.refcount);
                    ++qit;
                }
                break;   // each d_Q appears in cache at most once
            } else {
                ++qit;
            }
        }
    }

    // Free per-impl device buffers
    if (p->d_beta)       cudaFree(p->d_beta);
    if (p->d_pip)        cudaFree(p->d_pip);
    if (p->d_wcorr)      cudaFree(p->d_wcorr);
    if (p->d_what)       cudaFree(p->d_what);
    if (p->d_membership) cudaFree(p->d_membership);
    if (p->d_perm)       cudaFree(p->d_perm);
    if (p->d_urnd)       cudaFree(p->d_urnd);
    if (p->d_nrnd)       cudaFree(p->d_nrnd);
    if (p->d_meta)       cudaFree(p->d_meta);
    if (p->d_snpPi)      cudaFree(p->d_snpPi);
    if (p->d_gamma)      cudaFree(p->d_gamma);
    if (p->d_deltaPi)    cudaFree(p->d_deltaPi);
    if (p->d_badSnps)    cudaFree(p->d_badSnps);
    if (p->d_rng)        cudaFree(p->d_rng);

    // Free pinned host buffers
    if (p->h_snpPi_pinned)      cudaFreeHost(p->h_snpPi_pinned);
    if (p->h_deltaPi_pinned)    cudaFreeHost(p->h_deltaPi_pinned);
    if (p->h_wcorr_pinned)      cudaFreeHost(p->h_wcorr_pinned);
    if (p->h_what_pinned)       cudaFreeHost(p->h_what_pinned);
    if (p->h_membership_pinned) cudaFreeHost(p->h_membership_pinned);

    delete p;
    g_sbrc_states.erase(it);
}

// ──────────────────────────────────────────────────────────────────────────
// Milestone-3a: snpP = Φ(annoMat * α) GPU dispatch
// ──────────────────────────────────────────────────────────────────────────

struct SbrcAnnoImpl {
    int numSnps = 0;
    int numAnno = 0;
    float* d_annoMat = nullptr;    // numSnps × numAnno col-major (shared via g_annomat_cache)
    bool owns_annoMat = false;     // false → d_annoMat is shared, do not free in destructor
    float* d_alphai = nullptr;     // numAnno scratch
    float* d_y = nullptr;          // numSnps scratch (GEMV result / latent)
    float* d_zi = nullptr;         // numSnps scratch (M3a-2 latent input)
    float* d_snpP = nullptr;       // numSnps scratch (Φ-applied)
    float* h_snpP_pinned = nullptr;
    float* h_y_pinned = nullptr;
    curandState* d_rng = nullptr;  // for M3a-2 latent generation; lazily allocated
    int rng_n = 0;
    // M3a-3 Gibbs sweep buffers (allocated lazily on first sbrc_gpu_anno_gibbs_sweep_apply)
    // (removed d_alphai_full — was a duplicate buffer; M3a-3 now uses impl->d_alphai directly)
    float* d_annoDiagi = nullptr;
    int*   d_shuffled = nullptr;
    float* d_nrnd_gibbs = nullptr;
    float* d_ssq = nullptr;
    bool gibbs_initialized = false;
    cublasHandle_t cublas = nullptr;
    bool initialized = false;
    bool gpu_failed = false;
};

static std::unordered_map<const void*, SbrcAnnoImpl*> g_anno_states;
static std::mutex g_anno_mutex;

// Refcounted cache for the read-only annoMat device buffer. Without this each
// chain allocated its own 5.5 GB copy, and at 4 chains × 5.5 GB + 69 GB Q the
// 80 GB HBM3 budget on a single H100 overflowed — 2 of 4 chains fell back to
// CPU snpP, costing ~10% of wall time. annoMat is identical across chains for
// the same Data object so we key by (data pointer, total elements) and share.
struct AnnoMatCacheEntry {
    float* d_annoMat = nullptr;
    size_t elems = 0;
    int refcount = 0;
};
struct AnnoMatKey {
    const float* ptr;
    size_t elems;
    bool operator==(const AnnoMatKey& o) const { return ptr == o.ptr && elems == o.elems; }
};
struct AnnoMatKeyHash {
    size_t operator()(const AnnoMatKey& k) const {
        return std::hash<const float*>()(k.ptr) ^ std::hash<size_t>()(k.elems);
    }
};
static std::unordered_map<AnnoMatKey, AnnoMatCacheEntry, AnnoMatKeyHash> g_annomat_cache;

// Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
__global__ void normal_cdf_kernel(const float* __restrict__ y, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = 0.5f * (1.0f + erff(y[idx] * 0.70710678118654752f));   // 1/√2
}

// fwd decl
static SbrcAnnoImpl& get_or_init_anno(const void* caller_id, const MatrixXf& annoMat);

// Initialize cuRAND state for the latent kernel.
__global__ void init_anno_rng_kernel(curandState* states, uint64_t seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) curand_init(seed, idx, 0, &states[idx]);
}

// M3a-2 latent + GEMV-residual kernel:
//   mean = annoMat * alphai (already done via cuBLAS, in d_y on entry)
//   for j: sample z_j ~ TruncN(mean_j, 1, 0, sign by zi_j); y_j = z_j - mean_j
//
// CPU uses Stat::TruncatedNormal::sample_lower/upper_truncated (model.cpp:6670-6671):
// inverse-CDF method for moderate truncation; tail-rejection for |a-mean|>5σ. Here
// sd=1, threshold=0, mean is computed by the prior GEMV.
__global__ void anno_latent_kernel(const float* __restrict__ mean,   // in: numSnps
                                    const float* __restrict__ zi,    // in: numSnps (0 or 1)
                                    float* __restrict__ y,           // out: numSnps (z - mean)
                                    curandState* __restrict__ states,
                                    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curandState s = states[idx % gridDim.x / blockDim.x * blockDim.x + threadIdx.x];   // share state by block
    // Simpler: one state per thread index modulo grid×blockDim. To avoid complex sharing,
    // just use idx directly bounded by states_n.
    s = states[idx & 0x3FFFF];   // 256k states max (cycled)
    float mu = mean[idx];
    float z;
    if (zi[idx] != 0.0f) {
        // lower-truncated at 0: x >= 0
        // alpha = (0 - mu)/1 = -mu
        // alpha_cdf = Φ(-mu); u ~ U(0,1); x = alpha_cdf + (1-alpha_cdf)*u; z = mu + Φ⁻¹(x)
        float alpha_cdf = 0.5f * (1.0f + erff(-mu * 0.70710678118654752f));
        float u = curand_uniform(&s);
        float x = alpha_cdf + (1.0f - alpha_cdf) * u;
        // clamp to avoid numerical issues at endpoints
        if (x < 1e-7f) x = 1e-7f;
        if (x > 1.0f - 1e-7f) x = 1.0f - 1e-7f;
        z = mu + normcdfinvf(x);
    } else {
        // upper-truncated at 0: x <= 0
        float beta_cdf = 0.5f * (1.0f + erff(-mu * 0.70710678118654752f));
        float u;
        do { u = curand_uniform(&s); } while (u <= 0.0f);
        float x = beta_cdf * u;
        if (x < 1e-7f) x = 1e-7f;
        if (x > 1.0f - 1e-7f) x = 1.0f - 1e-7f;
        z = mu + normcdfinvf(x);
    }
    states[idx & 0x3FFFF] = s;
    y[idx] = z - mu;
}

// M3a-3 Gibbs sweep kernel — one block does the full numAnno-iteration sweep over
// annoMat columns. y stays in HBM (too big for shared); parallel reduction across
// threads in the block for dot/axpy ops.
__global__ void anno_gibbs_sweep_kernel(
    const float* __restrict__ annoMat,    // m × numAnno col-major
    float* __restrict__ y,                // m, in-place
    float* __restrict__ alphai,           // numAnno, in-place
    const float* __restrict__ annoDiagi,  // numAnno
    float sigmaSq_i,
    int m,
    int numAnno,
    const int* __restrict__ shuffled_idx, // numAnno (entries are k values; index 0 = intercept skipped)
    const float* __restrict__ nrnd,       // numAnno standard normal samples
    float* __restrict__ ssq_out)          // 1 scalar (out)
{
    extern __shared__ float smem[];
    float* sshared = smem;                // scratch for reductions
    float* sdelta  = smem + 64;           // 1 slot for the per-k delta
    float* sssq    = smem + 65;           // 1 slot for ssq accumulator

    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const unsigned mask = 0xffffffff;

    if (tid == 0) sssq[0] = 0.0f;
    __syncthreads();

    for (int t = 0; t < numAnno; ++t) {
        int k = shuffled_idx[t];
        if (k == 0) continue;   // intercept handled on CPU (uses different formula)
        const float* col = annoMat + (long long)k * m;
        float oldSample = alphai[k];

        // Parallel dot: col · y — accumulate in DOUBLE to avoid chain drift over
        // 188 sequential Gibbs steps (FP32 accumulation drifts ~1e-3 per step which
        // compounds enough to shift mixture component assignments after numAnno iters).
        double partial = 0.0;
        for (int r = tid; r < m; r += blockSize)
            partial += (double)col[r] * (double)y[r];

        // Warp reduce in double precision via shuffle
        unsigned long long pll = __double_as_longlong(partial);
        // shfl_xor for doubles via 2x int shuffles
        auto warp_reduce_d = [&](double v) {
            unsigned long long bits = __double_as_longlong(v);
            for (int off = 16; off > 0; off >>= 1) {
                int lo = __shfl_xor_sync(mask, (int)(bits & 0xffffffffu), off);
                int hi = __shfl_xor_sync(mask, (int)(bits >> 32), off);
                unsigned long long other = ((unsigned long long)(unsigned)hi << 32) | (unsigned)lo;
                v += __longlong_as_double(other);
                bits = __double_as_longlong(v);
            }
            return v;
        };
        partial = warp_reduce_d(partial);

        int warpId = tid >> 5;
        int lane   = tid & 31;
        // Use sshared as a double[32] (32 warps max for 1024 threads)
        double* sshared_d = (double*)sshared;
        if (lane == 0) sshared_d[warpId] = partial;
        __syncthreads();

        if (warpId == 0) {
            int nwarps = blockSize >> 5;
            double v = (tid < nwarps) ? sshared_d[tid] : 0.0;
            v = warp_reduce_d(v);
            if (tid == 0) {
                double dot = v;
                double rhs = dot + (double)annoDiagi[k] * (double)oldSample;
                double denom = (double)annoDiagi[k] + 1.0 / (double)sigmaSq_i;
                // Guard against pathological denom (sigmaSq_i 0/negative or annoDiagi[k]
                // == -1/sigmaSq_i). CPU's Normal::sample(ahat, invLhs) silently produces
                // NaN here too — but the CPU chain's NaN are bounded; mine apparently
                // drift more. Fall back to keeping the old sample if math would NaN.
                float new_alpha;
                if (denom <= 0.0 || !isfinite(denom) || !isfinite(rhs)) {
                    new_alpha = oldSample;
                } else {
                    double invLhs = 1.0 / denom;
                    double ahat = invLhs * rhs;
                    double draw = ahat + (double)nrnd[t] * sqrt(invLhs);
                    new_alpha = isfinite(draw) ? (float)draw : oldSample;
                }
                alphai[k] = new_alpha;
                sdelta[0] = oldSample - new_alpha;
                sssq[0] += new_alpha * new_alpha;
            }
        }
        __syncthreads();

        float delta = sdelta[0];
        for (int r = tid; r < m; r += blockSize) {
            y[r] += col[r] * delta;
        }
        __syncthreads();
    }

    if (tid == 0) ssq_out[0] = sssq[0];
}

// M3a-3 v4: cuBLAS-driven Gibbs sweep. Replaces the single-block custom kernel
// (which only filled 1 of 132 SMs and ran 2.5× slower than CPU at full Imputed).
//
// For each k in shuffled_idx:
//   dot     = annoMat[:,k] · y      (cublasSdot, multi-block reduction)
//   new_alpha = ahat + nrnd[t] * sqrt(invLhs)   (host arithmetic)
//   y      += (oldSample - new_alpha) * annoMat[:,k]   (cublasSaxpy)
//
// RNG draws come from the caller (so chain dynamics match CPU's snorm() sequence
// exactly). The per-k host<->device sync from cublasSdot host-pointer mode is
// the only stall; with ~30 µs per dot on H100 and ~25 µs per axpy, a 187-iter
// sweep is ~10 ms vs ~500 ms on CPU at full Imputed (~45× speedup expected).
bool sbrc_gpu_anno_gibbs_sweep_apply(const void* caller_identity,
                                      const MatrixXf& annoMat,
                                      VectorXf& y,
                                      VectorXf& alphai,
                                      const VectorXf& annoDiagi,
                                      float sigmaSq_i,
                                      const std::vector<int>& shuffled_idx,
                                      const std::vector<float>& nrnd,
                                      float& ssq_out) {
    if (!sbrc_gpu_enabled) return false;
    SbrcAnnoImpl* impl;
    {
        std::lock_guard<std::mutex> lk(g_anno_mutex);
        SbrcAnnoImpl& impl_ref = get_or_init_anno(caller_identity, annoMat);
        impl = &impl_ref;
    }
    if (impl->gpu_failed) return false;
    if (impl->cublas == nullptr) return false;   // need cuBLAS handle
    if (y.size() != impl->numSnps) return false;
    if (alphai.size() != impl->numAnno || annoDiagi.size() != impl->numAnno) return false;
    if ((int)shuffled_idx.size() != impl->numAnno - 1) return false;
    if ((int)nrnd.size() != impl->numAnno - 1) return false;

    const int m = impl->numSnps;

    // Defensive: scrub NaN/Inf from inputs so the cuBLAS reduction never sees them.
    // A single NaN in y propagates through every subsequent k via the dot product.
    auto scrub_nan = [](float* p, int n) {
        for (int i = 0; i < n; ++i) if (!std::isfinite(p[i])) p[i] = 0.0f;
    };
    scrub_nan(y.data(), m);

    // Upload y to device once. annoMat (column-major) is already resident in
    // impl->d_annoMat from the M3a snpP initialization.
    CUDA_CHECK(cudaMemcpy(impl->d_y, y.data(), sizeof(float) * m, cudaMemcpyHostToDevice));

    // Use host-pointer mode (default) so cublasSdot result lands on host and we
    // can branch on it within this thread without writing a device kernel for
    // the alpha-update math. Each Sdot is synchronous; saxpy is async-enqueued.
    cublasSetPointerMode(impl->cublas, CUBLAS_POINTER_MODE_HOST);

    float ssq = 0.0f;
    const int sweep_len = impl->numAnno - 1;
    for (int t = 0; t < sweep_len; ++t) {
        int k = shuffled_idx[t];
        if (k <= 0 || k >= impl->numAnno) continue;   // safety
        float oldSample = alphai[k];
        const float* col_k = impl->d_annoMat + (size_t)k * m;

        float dot = 0.0f;
        cublasStatus_t st = cublasSdot(impl->cublas, m, col_k, 1, impl->d_y, 1, &dot);
        if (st != CUBLAS_STATUS_SUCCESS) return false;

        float denom = annoDiagi[k] + 1.0f / sigmaSq_i;
        float new_alpha;
        if (!(denom > 0.0f) || !std::isfinite(denom) || !std::isfinite(dot)) {
            // Pathological math: keep oldSample so chain stays alive.
            new_alpha = oldSample;
        } else {
            float invLhs = 1.0f / denom;
            float ahat = invLhs * (dot + annoDiagi[k] * oldSample);
            float draw = ahat + nrnd[t] * sqrtf(invLhs);
            new_alpha = std::isfinite(draw) ? draw : oldSample;
        }
        alphai[k] = new_alpha;
        ssq += new_alpha * new_alpha;

        float delta = oldSample - new_alpha;
        if (delta != 0.0f) {
            st = cublasSaxpy(impl->cublas, m, &delta, col_k, 1, impl->d_y, 1);
            if (st != CUBLAS_STATUS_SUCCESS) return false;
        }
    }

    // Final sync — flush the queued axpys before downloading y.
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(y.data(), impl->d_y, sizeof(float) * m, cudaMemcpyDeviceToHost));
    ssq_out = ssq;
    return true;
}

bool sbrc_gpu_anno_gemv_latent_apply(const void* caller_identity,
                                     const MatrixXf& annoMat,
                                     const VectorXf& alphai,
                                     const VectorXf& zi,
                                     VectorXf& y) {
    if (!sbrc_gpu_enabled) return false;
    SbrcAnnoImpl* impl;
    {
        std::lock_guard<std::mutex> lk(g_anno_mutex);
        SbrcAnnoImpl& impl_ref = get_or_init_anno(caller_identity, annoMat);
        impl = &impl_ref;
    }
    if (impl->gpu_failed) return false;
    if (alphai.size() != impl->numAnno || zi.size() != impl->numSnps) return false;

    // Lazily allocate + init RNG states (256k pool)
    if (!impl->d_rng) {
        impl->rng_n = 1 << 18;   // 256k states
        if (cudaMalloc(&impl->d_rng, sizeof(curandState) * impl->rng_n) != cudaSuccess) {
            std::fprintf(stderr, "[sbrc_gpu] anno latent cuRAND alloc failed; CPU fallback.\n");
            return false;
        }
        init_anno_rng_kernel<<<(impl->rng_n + 255)/256, 256>>>(impl->d_rng, 0xa110ce, impl->rng_n);
    }

    // Upload alphai + zi
    CUDA_CHECK(cudaMemcpy(impl->d_alphai, alphai.data(),
                          sizeof(float) * impl->numAnno, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(impl->d_zi, zi.data(),
                          sizeof(float) * impl->numSnps, cudaMemcpyHostToDevice));
    // mean = annoMat * alphai → d_y
    const float one = 1.0f, zero = 0.0f;
    if (cublasSgemv(impl->cublas, CUBLAS_OP_N, impl->numSnps, impl->numAnno,
                    &one, impl->d_annoMat, impl->numSnps,
                    impl->d_alphai, 1, &zero, impl->d_y, 1) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    // Latent kernel: in-place overwrite d_y with z - mean. But we need both
    // mean (in d_y) and the output (z - mean) — overwrite is safe since each
    // thread reads d_y[idx] then writes y[idx].
    int threads = 256;
    int blocks = (impl->numSnps + threads - 1) / threads;
    anno_latent_kernel<<<blocks, threads>>>(impl->d_y, impl->d_zi, impl->d_y, impl->d_rng, impl->numSnps);
    CUDA_CHECK(cudaMemcpy(impl->h_y_pinned, impl->d_y,
                          sizeof(float) * impl->numSnps, cudaMemcpyDeviceToHost));
    if (y.size() != impl->numSnps) y.resize(impl->numSnps);
    std::memcpy(y.data(), impl->h_y_pinned, sizeof(float) * impl->numSnps);
    return true;
}

static SbrcAnnoImpl& get_or_init_anno(const void* caller_id, const MatrixXf& annoMat) {
    auto it = g_anno_states.find(caller_id);
    if (it != g_anno_states.end()) return *it->second;

    SbrcAnnoImpl* impl = new SbrcAnnoImpl();
    g_anno_states[caller_id] = impl;
    impl->numSnps = (int)annoMat.rows();
    impl->numAnno = (int)annoMat.cols();
    const size_t annoElems = (size_t)impl->numSnps * impl->numAnno;
    const size_t annoBytes = sizeof(float) * annoElems;

    // Try the shared cache first — annoMat is identical across chains for the
    // same Data object, so we can hold one device copy refcounted across chains.
    AnnoMatKey key{annoMat.data(), annoElems};
    auto cit = g_annomat_cache.find(key);
    if (cit != g_annomat_cache.end()) {
        impl->d_annoMat = cit->second.d_annoMat;
        impl->owns_annoMat = false;
        cit->second.refcount += 1;
        std::fprintf(stderr,
            "[sbrc_gpu] reusing cached d_annoMat (%.2f GB) for caller %p (refcount=%d)\n",
            annoBytes / 1e9, caller_id, cit->second.refcount);
    } else {
        std::fprintf(stderr, "[sbrc_gpu] anno init: numSnps=%d  numAnno=%d  size=%.2f GB\n",
                     impl->numSnps, impl->numAnno, annoBytes / 1e9);
        cudaError_t err = cudaMalloc(&impl->d_annoMat, annoBytes);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "[sbrc_gpu] anno cudaMalloc(%.2f GB) failed; CPU fallback for snpP.\n",
                         annoBytes / 1e9);
            cudaGetLastError();
            impl->gpu_failed = true;
            impl->initialized = true;
            return *impl;
        }
        impl->owns_annoMat = true;
        // Upload annoMat (one-time per Data object)
        CUDA_CHECK(cudaMemcpy(impl->d_annoMat, annoMat.data(), annoBytes, cudaMemcpyHostToDevice));
        AnnoMatCacheEntry e;
        e.d_annoMat = impl->d_annoMat;
        e.elems = annoElems;
        e.refcount = 1;
        g_annomat_cache[key] = e;
    }
    CUDA_CHECK(cudaMalloc(&impl->d_alphai, sizeof(float) * impl->numAnno));
    CUDA_CHECK(cudaMalloc(&impl->d_y, sizeof(float) * impl->numSnps));
    CUDA_CHECK(cudaMalloc(&impl->d_zi, sizeof(float) * impl->numSnps));
    CUDA_CHECK(cudaMalloc(&impl->d_snpP, sizeof(float) * impl->numSnps));
    CUDA_CHECK(cudaMallocHost(&impl->h_snpP_pinned, sizeof(float) * impl->numSnps));
    CUDA_CHECK(cudaMallocHost(&impl->h_y_pinned, sizeof(float) * impl->numSnps));
    if (cublasCreate(&impl->cublas) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "[sbrc_gpu] cublasCreate failed; CPU fallback for snpP.\n");
        impl->gpu_failed = true;
    }
    impl->initialized = true;
    return *impl;
}

bool sbrc_gpu_anno_snpP_apply(const void* caller_identity,
                              const MatrixXf& annoMat,
                              const VectorXf& alphai,
                              VectorXf& snpP_col) {
    if (!sbrc_gpu_enabled) return false;
    SbrcAnnoImpl* impl;
    {
        std::lock_guard<std::mutex> lk(g_anno_mutex);
        SbrcAnnoImpl& impl_ref = get_or_init_anno(caller_identity, annoMat);
        impl = &impl_ref;
    }
    if (impl->gpu_failed) return false;
    if (alphai.size() != impl->numAnno) return false;

    // Upload alphai
    CUDA_CHECK(cudaMemcpy(impl->d_alphai, alphai.data(),
                          sizeof(float) * impl->numAnno, cudaMemcpyHostToDevice));
    // y = annoMat * alphai (col-major, no transpose; m rows × numAnno cols)
    const float one = 1.0f, zero = 0.0f;
    cublasStatus_t st = cublasSgemv(impl->cublas, CUBLAS_OP_N,
                                    impl->numSnps, impl->numAnno,
                                    &one, impl->d_annoMat, impl->numSnps,
                                    impl->d_alphai, 1,
                                    &zero, impl->d_y, 1);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "[sbrc_gpu] cublasSgemv failed st=%d; CPU fallback for this snpP call.\n", (int)st);
        return false;
    }
    // snpP_col = Φ(y)
    int threads = 256, blocks = (impl->numSnps + threads - 1) / threads;
    normal_cdf_kernel<<<blocks, threads>>>(impl->d_y, impl->d_snpP, impl->numSnps);
    CUDA_CHECK(cudaMemcpy(impl->h_snpP_pinned, impl->d_snpP,
                          sizeof(float) * impl->numSnps, cudaMemcpyDeviceToHost));

    if (snpP_col.size() != impl->numSnps) snpP_col.resize(impl->numSnps);
    std::memcpy(snpP_col.data(), impl->h_snpP_pinned, sizeof(float) * impl->numSnps);
    return true;
}

void sbrc_gpu_anno_release(const void* caller_identity) {
    if (!sbrc_gpu_enabled) return;
    std::lock_guard<std::mutex> lk(g_anno_mutex);
    auto it = g_anno_states.find(caller_identity);
    if (it == g_anno_states.end()) return;
    SbrcAnnoImpl* p = it->second;
    if (p) {
        if (p->d_annoMat) {
            // Every impl that called get_or_init_anno bumped refcount (whether it
            // allocated or reused). So unconditionally decrement here.
            for (auto cit = g_annomat_cache.begin(); cit != g_annomat_cache.end(); ) {
                if (cit->second.d_annoMat == p->d_annoMat) {
                    cit->second.refcount -= 1;
                    if (cit->second.refcount <= 0) {
                        std::fprintf(stderr,
                            "[sbrc_gpu] freeing d_annoMat (%.2f GB; last ref dropped)\n",
                            cit->second.elems * sizeof(float) / 1e9);
                        cudaFree(cit->second.d_annoMat);
                        cit = g_annomat_cache.erase(cit);
                    } else {
                        ++cit;
                    }
                    break;
                } else {
                    ++cit;
                }
            }
        }
        if (p->d_alphai)  cudaFree(p->d_alphai);
        if (p->d_y)       cudaFree(p->d_y);
        if (p->d_zi)      cudaFree(p->d_zi);
        if (p->d_snpP)    cudaFree(p->d_snpP);
        if (p->d_rng)     cudaFree(p->d_rng);
        // d_alphai_full removed; impl->d_alphai is freed elsewhere
        if (p->d_annoDiagi)   cudaFree(p->d_annoDiagi);
        if (p->d_shuffled)    cudaFree(p->d_shuffled);
        if (p->d_nrnd_gibbs)  cudaFree(p->d_nrnd_gibbs);
        if (p->d_ssq)         cudaFree(p->d_ssq);
        if (p->h_snpP_pinned) cudaFreeHost(p->h_snpP_pinned);
        if (p->h_y_pinned)    cudaFreeHost(p->h_y_pinned);
        if (p->cublas)    cublasDestroy(p->cublas);
        delete p;
    }
    g_anno_states.erase(it);
}

// Public query: did this caller's GPU dispatch succeed?
bool sbrc_gpu_caller_ok(const void* caller_identity) {
    if (!sbrc_gpu_enabled) return false;
    std::lock_guard<std::mutex> lk(g_state_mutex);
    auto it = g_sbrc_states.find(caller_identity);
    if (it == g_sbrc_states.end()) return false;
    return !it->second->gpu_failed;
}

// ─── Public dispatch function ──────────────────────────────────────────────
void sbrc_gpu_dispatch_sample_from_fc_eigen(
    const void* caller_identity,
    std::vector<VectorXf>& wcorrBlocks,
    const std::vector<MatrixXf>& Qblocks,
    std::vector<VectorXf>& whatBlocks,
    const std::vector<LDBlockInfo*>& keptLdBlockInfoVec,
    const VectorXf& nGWASblocks,
    const VectorXf& vareBlocks,
    const MatrixXf& snpPi,
    const VectorXf& gamma,
    float varg,
    bool hsqPercModel,
    float sigmaSq,
    int ndist,
    int totalSnps,
    const VectorXi& badSnps,
    VectorXf& values,
    VectorXf& pip,
    VectorXi& membership_eig,
    MatrixXf& z,
    VectorXf& fcMean,
    float& sumSq,
    float& wtdSumSq,
    unsigned& numNonZeros,
    VectorXf& nnzPerBlk,
    VectorXf& ssqBlocks,
    ArrayXf& numSnpMix,
    std::vector<std::vector<unsigned>>& snpset,
    MatrixXf& deltaPi_values)
{
    const int nBlocks = (int)Qblocks.size();
    SbrcGpuImpl* impl;
    {
        SbrcGpuImpl& impl_ref = get_or_init_state(caller_identity, nBlocks, Qblocks, wcorrBlocks,
                                                  keptLdBlockInfoVec, totalSnps, ndist);
        impl = &impl_ref;
    }
    // From here on we operate on *impl* without holding g_state_mutex. Other callers
    // can init their own impls concurrently; they're separate map entries.
    std::lock_guard<std::mutex> sweep_lk(impl->sweep_mutex);

    if (impl->gpu_failed) {
        // GPU init OOMed earlier; skip dispatch (signal via caller_ok query) and
        // leave outputs untouched. The caller will fall back to the CPU path.
        return;
    }

    auto t_call_start = clock_now::now();
    auto t_upload_start = t_call_start;
    static int call_n = 0;
    if (call_n < 3) std::fprintf(stderr, "[sbrc_gpu] sweep call %d start: nBlocks=%d totalSnps=%d ndist=%d wcorrBlocks.size=%zu\n",
                                  call_n, nBlocks, totalSnps, ndist, wcorrBlocks.size());

    // Update per-block vareDn = nGWAS / vare and push meta
    for (int i = 0; i < nBlocks; ++i) {
        float ng = nGWASblocks[i];
        float ve = vareBlocks[i];
        impl->h_meta[i].vareDn = (ve > 0.0f) ? (ng / ve) : 0.0f;
    }
    CUDA_CHECK(cudaMemcpy(impl->d_meta, impl->h_meta.data(),
                          sizeof(BlockMeta) * nBlocks, cudaMemcpyHostToDevice));

    // Re-upload wcorrBlocks (host is authoritative; mutated externally every 100 iter)
    // — pack into pinned buffer in parallel, then a single async H→D copy.
    {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nBlocks; ++i) {
            int qi = impl->h_meta[i].q;
            int qOff_i = impl->h_meta[i].qOffset;
            std::memcpy(impl->h_wcorr_pinned + qOff_i, wcorrBlocks[i].data(), sizeof(float) * qi);
        }
        CUDA_CHECK(cudaMemcpyAsync(impl->d_wcorr, impl->h_wcorr_pinned,
                                   sizeof(float) * impl->totalQDim, cudaMemcpyHostToDevice, impl->stream));
    }

    // Upload β. Caller's `values` is sized to global totalSnps; on first call β is 0.
    if (values.size() == totalSnps) {
        CUDA_CHECK(cudaMemcpyAsync(impl->d_beta, values.data(), sizeof(float) * totalSnps, cudaMemcpyHostToDevice, impl->stream));
    } else {
        CUDA_CHECK(cudaMemsetAsync(impl->d_beta, 0, sizeof(float) * totalSnps, impl->stream));
    }

    // Upload badSnps mask if caller provided one.
    if (badSnps.size() == totalSnps) {
        CUDA_CHECK(cudaMemcpyAsync(impl->d_badSnps, badSnps.data(),
                                   sizeof(int) * totalSnps, cudaMemcpyHostToDevice, impl->stream));
    }

    // snpPi transpose col-major (Eigen) → row-major pinned host → async H→D on impl->stream.
    const float* snpPi_data = snpPi.data();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < totalSnps; ++i) {
        float* dst = impl->h_snpPi_pinned + (size_t)i * ndist;
        for (int k = 0; k < ndist; ++k) dst[k] = snpPi_data[(size_t)i + (size_t)k * totalSnps];
    }
    CUDA_CHECK(cudaMemcpyAsync(impl->d_snpPi, impl->h_snpPi_pinned,
                               sizeof(float) * (size_t)totalSnps * ndist, cudaMemcpyHostToDevice, impl->stream));

    // gamma
    CUDA_CHECK(cudaMemcpyAsync(impl->d_gamma, gamma.data(), sizeof(float) * ndist, cudaMemcpyHostToDevice, impl->stream));

    CUDA_CHECK(cudaStreamSynchronize(impl->stream));   // ensure upload memcpys are done before timing kernel
    auto t_kernel_start = clock_now::now();

    // Kernel launches on impl->stream so multi-chain runs can overlap on GPU.
    const int threads = 512;
    pregen_rng_kernel<<<nBlocks, threads, 0, impl->stream>>>(
        impl->d_rng, totalSnps, impl->d_urnd, impl->d_nrnd, impl->d_perm,
        impl->d_meta, nBlocks);

    int nwarps = threads / 32;
    // K1a: 3*maxQ shared mem (wcorr + what + Qi staging). For maxQ=3993, ~48 KB.
    size_t shmemBytes = sizeof(float) * (3 * impl->maxQ + nwarps);
    block_sweep_kernel<<<nBlocks, threads, shmemBytes, impl->stream>>>(
        impl->d_meta, impl->d_Q, impl->d_beta, impl->d_wcorr, impl->d_what,
        impl->d_pip, impl->d_membership, impl->d_deltaPi, impl->d_badSnps,
        impl->d_perm, impl->d_urnd, impl->d_nrnd,
        impl->d_snpPi, impl->d_gamma,
        ndist, sigmaSq, varg, hsqPercModel ? 1 : 0);

    CUDA_CHECK(cudaStreamSynchronize(impl->stream));
    auto t_download_start = clock_now::now();
    if (call_n < 3) std::fprintf(stderr, "[sbrc_gpu] sweep call %d kernel done\n", call_n);

    if (values.size() != totalSnps) values.resize(totalSnps);
    if (pip.size()    != totalSnps) pip.resize(totalSnps);
    CUDA_CHECK(cudaMemcpyAsync(values.data(), impl->d_beta, sizeof(float) * totalSnps, cudaMemcpyDeviceToHost, impl->stream));
    CUDA_CHECK(cudaMemcpyAsync(pip.data(),    impl->d_pip,  sizeof(float) * totalSnps, cudaMemcpyDeviceToHost, impl->stream));
    CUDA_CHECK(cudaMemcpyAsync(impl->h_membership_pinned, impl->d_membership,
                               sizeof(int) * totalSnps, cudaMemcpyDeviceToHost, impl->stream));
    CUDA_CHECK(cudaMemcpyAsync(impl->h_deltaPi_pinned, impl->d_deltaPi,
                               sizeof(float) * (size_t)totalSnps * ndist, cudaMemcpyDeviceToHost, impl->stream));
    CUDA_CHECK(cudaMemcpyAsync(impl->h_wcorr_pinned, impl->d_wcorr,
                               sizeof(float) * impl->totalQDim, cudaMemcpyDeviceToHost, impl->stream));
    CUDA_CHECK(cudaMemcpyAsync(impl->h_what_pinned, impl->d_what,
                               sizeof(float) * impl->totalQDim, cudaMemcpyDeviceToHost, impl->stream));
    CUDA_CHECK(cudaStreamSynchronize(impl->stream));

    const int* membership_buf = impl->h_membership_pinned;

    // membership_eig: VectorXi indexed
    membership_eig.resize(totalSnps);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < totalSnps; ++i) membership_eig(i) = membership_buf[i];

    // deltaPi: row-major pinned → col-major Eigen MatrixXf, OMP-parallel transpose
    if (deltaPi_values.rows() != totalSnps || deltaPi_values.cols() != ndist)
        deltaPi_values.resize(totalSnps, ndist);
    {
        float* dst_data = deltaPi_values.data();   // col-major: dst(i,k) = dst_data[i + k*totalSnps]
        const float* src = impl->h_deltaPi_pinned;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < totalSnps; ++i) {
            const float* row = src + (size_t)i * ndist;
            for (int k = 0; k < ndist; ++k) dst_data[(size_t)i + (size_t)k * totalSnps] = row[k];
        }
    }

    auto t_agg_start = clock_now::now();
    if (call_n < 3) std::fprintf(stderr, "[sbrc_gpu] sweep call %d before agg: values.size=%lld pip.size=%lld z.rows=%lld z.cols=%lld dpv.rows=%lld dpv.cols=%lld\n",
                                  call_n, (long long)values.size(), (long long)pip.size(),
                                  (long long)z.rows(), (long long)z.cols(),
                                  (long long)deltaPi_values.rows(), (long long)deltaPi_values.cols());

    // wcorrBlocks / whatBlocks: unpack from pinned buffers in parallel
    if ((int)wcorrBlocks.size() != nBlocks) wcorrBlocks.resize(nBlocks);
    if ((int)whatBlocks.size()  != nBlocks) whatBlocks.resize(nBlocks);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nBlocks; ++i) {
        int qi = impl->h_meta[i].q;
        int qOff_i = impl->h_meta[i].qOffset;
        if ((int)wcorrBlocks[i].size() != qi) wcorrBlocks[i].resize(qi);
        if ((int)whatBlocks[i].size()  != qi) whatBlocks[i].resize(qi);
        std::memcpy(wcorrBlocks[i].data(), impl->h_wcorr_pinned + qOff_i, sizeof(float) * qi);
        std::memcpy(whatBlocks[i].data(),  impl->h_what_pinned  + qOff_i, sizeof(float) * qi);
    }

    // Host-side aggregation: OMP-parallel over blocks with per-thread snpset accumulators.
    nnzPerBlk.setZero(nBlocks);
    ssqBlocks.setZero(nBlocks);
    z.setZero(totalSnps, ndist - 1);

    // Per-thread accumulators
    int nThreads = 1;
    #ifdef _OPENMP
        nThreads = omp_get_max_threads();
    #endif
    std::vector<std::vector<std::vector<unsigned>>> thread_snpset(nThreads,
        std::vector<std::vector<unsigned>>(ndist));
    std::vector<std::vector<float>> thread_numSnpMix(nThreads, std::vector<float>(ndist, 0.0f));
    std::vector<double> thread_sumSq(nThreads, 0.0);
    std::vector<double> thread_wtdSumSq(nThreads, 0.0);
    std::vector<unsigned> thread_nnz(nThreads, 0u);

    #pragma omp parallel for schedule(dynamic, 4)
    for (int blk = 0; blk < nBlocks; ++blk) {
        int tid = 0;
        #ifdef _OPENMP
            tid = omp_get_thread_num();
        #endif
        unsigned blockStart = keptLdBlockInfoVec[blk]->startSnpIdx;
        unsigned blockEnd   = keptLdBlockInfoVec[blk]->endSnpIdx;
        float local_ssq = 0.0f; int local_nnz = 0;
        for (unsigned i = blockStart; i <= blockEnd; ++i) {
            int delta = membership_buf[i];
            // Count every non-bad SNP (incl. delta=0) in numSnpMix — the R sampler reads
            // this as snpStore and feeds it to Pis.sampleFromFC. Without the delta=0 count,
            // the prior for zero-effect collapses and the chain diverges.
            thread_numSnpMix[tid][delta] += 1.0f;
            if (delta == 0) continue;
            float b = values[i];
            local_ssq += b * b;
            thread_sumSq[tid]    += b * b;
            thread_wtdSumSq[tid] += (b * b) / gamma(delta);
            ++local_nnz;
            thread_nnz[tid] += 1u;
            thread_snpset[tid][delta].push_back(i);
            for (int k2 = 0; k2 < delta; ++k2) z(i, k2) = 1.0f;
        }
        ssqBlocks(blk) = local_ssq;
        nnzPerBlk(blk) = (float)local_nnz;
    }

    // Reduce thread-locals
    sumSq = 0.0f; wtdSumSq = 0.0f; numNonZeros = 0;
    numSnpMix.setZero(ndist);
    snpset.assign(ndist, std::vector<unsigned>());
    for (int t = 0; t < nThreads; ++t) {
        sumSq        += (float)thread_sumSq[t];
        wtdSumSq     += (float)thread_wtdSumSq[t];
        numNonZeros  += thread_nnz[t];
        for (int k = 0; k < ndist; ++k) {
            numSnpMix(k) += thread_numSnpMix[t][k];
            // append per-thread snpset
            auto& dst = snpset[k];
            auto& src_v = thread_snpset[t][k];
            dst.insert(dst.end(), src_v.begin(), src_v.end());
        }
    }

    fcMean.setZero(totalSnps);   // not computed on GPU; downstream uses are gated
    if (call_n < 3) std::fprintf(stderr, "[sbrc_gpu] sweep call %d agg done\n", call_n);
    ++call_n;

    auto t_end = clock_now::now();
    auto dur = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
    g_dispatch_timing.upload   += dur(t_upload_start,   t_kernel_start);
    g_dispatch_timing.kernel   += dur(t_kernel_start,   t_download_start);
    g_dispatch_timing.download += dur(t_download_start, t_agg_start);
    g_dispatch_timing.agg      += dur(t_agg_start,      t_end);
    g_dispatch_timing.total    += dur(t_call_start,     t_end);
    g_dispatch_timing.calls    += 1;
}
