// CUDA environment smoke only; no SGLang, Mooncake, RDMA or cache acceptance.
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>

__global__ void invert_bytes(unsigned char* bytes, std::size_t count) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
        bytes[index] = static_cast<unsigned char>(255 - bytes[index]);
}

int main() {
    constexpr std::size_t bytes = 4 * 1024 * 1024;
    constexpr std::size_t budget = 1024ULL * 1024 * 1024;
    unsigned char *source = nullptr, *destination = nullptr, *device = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t ready = nullptr;
    cudaDeviceProp properties{};
    std::size_t free_before = 0, total = 0, free_after = 0;
    int count = 0, driver = 0, runtime = 0;
    bool passed = false;
    auto ok = [](cudaError_t result, const char* operation) {
        if (result == cudaSuccess) return true;
        std::fprintf(stderr, "%s failed: %s\n", operation,
                     cudaGetErrorString(result));
        return false;
    };
#define CHECK(operation) \
    if (!ok((operation), #operation)) goto cleanup
    CHECK(cudaGetDeviceCount(&count));
    if (count != 1) {
        std::fprintf(stderr, "Require exactly one visible GPU; got %d\n",
                     count);
        goto cleanup;
    }
    CHECK(cudaSetDevice(0));
    CHECK(cudaGetDeviceProperties(&properties, 0));
    if (properties.major != 12 || properties.minor != 0) {
        std::fprintf(stderr, "Expected verified sm120 device\n");
        goto cleanup;
    }
    CHECK(cudaDriverGetVersion(&driver));
    CHECK(cudaRuntimeGetVersion(&runtime));
    CHECK(cudaMemGetInfo(&free_before, &total));
    if (total - free_before + bytes > budget) {
        std::fprintf(stderr,
                     "Context/device memory already exceeds 1 GiB budget\n");
        goto cleanup;
    }
    CHECK(cudaHostAlloc(reinterpret_cast<void**>(&source), bytes,
                        cudaHostAllocDefault));
    CHECK(cudaHostAlloc(reinterpret_cast<void**>(&destination), bytes,
                        cudaHostAllocDefault));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&device), bytes));
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    for (std::size_t i = 0; i < bytes; ++i)
        source[i] = static_cast<unsigned char>((i * 17 + 5) & 255);
    CHECK(
        cudaMemcpyAsync(device, source, bytes, cudaMemcpyHostToDevice, stream));
    invert_bytes<<<(bytes + 255) / 256, 256, 0, stream>>>(device, bytes);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(ready, stream));
    {
        const cudaError_t observed = cudaEventQuery(ready);
        // A tiny kernel may already finish. Neither outcome is invented.
        if (observed != cudaSuccess && observed != cudaErrorNotReady) {
            ok(observed, "cudaEventQuery before synchronization");
            goto cleanup;
        }
        std::printf("event_before_sync=%s\n",
                    observed == cudaSuccess ? "ready" : "not_ready");
    }
    CHECK(cudaEventSynchronize(ready));
    CHECK(cudaEventQuery(ready));
    CHECK(cudaMemcpyAsync(destination, device, bytes, cudaMemcpyDeviceToHost,
                          stream));
    CHECK(cudaStreamSynchronize(stream));
    for (std::size_t i = 0; i < bytes; ++i) {
        if (destination[i] != static_cast<unsigned char>(255 - source[i])) {
            std::fprintf(stderr, "Byte verification failed at %zu\n", i);
            goto cleanup;
        }
    }
    CHECK(cudaMemGetInfo(&free_after, &total));
    if (total - free_after > budget) {
        std::fprintf(stderr,
                     "Observed context/device memory exceeds 1 GiB budget\n");
        goto cleanup;
    }
    std::printf(
        "driver_api=%d runtime=%d sm=%d%d device_bytes=%zu pinned_bytes=%zu "
        "observed_used_bytes=%zu\n",
        driver, runtime, properties.major, properties.minor, bytes, 2 * bytes,
        total - free_after);
    passed = true;
cleanup:
    // Do not free a host source while an unsuccessful stream drain leaves it
    // unsafe. The owning test process exits on failure; this is not a shutdown
    // safety proof.
    if (stream && !ok(cudaStreamSynchronize(stream), "cleanup stream drain"))
        return 1;
    if (ready && !ok(cudaEventDestroy(ready), "cudaEventDestroy"))
        passed = false;
    if (stream && !ok(cudaStreamDestroy(stream), "cudaStreamDestroy"))
        passed = false;
    if (device && !ok(cudaFree(device), "cudaFree")) passed = false;
    if (destination &&
        !ok(cudaFreeHost(destination), "cudaFreeHost destination"))
        passed = false;
    if (source && !ok(cudaFreeHost(source), "cudaFreeHost source"))
        passed = false;
    std::printf("cuda_environment_smoke=%s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
#undef CHECK
}
