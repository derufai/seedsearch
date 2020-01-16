// compile with: nvcc -ccbin=g++-5 -O3 -std=c++11 -arch compute_60 -x cu seedsearch.cpp -o seedsearch
// edit the flat_chunks variable with the x, z chunk coordinates of ocean monuments

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <functional>
#include <fstream>
#include <stdio.h>
#include <cassert>
#include <vector>
#include <algorithm>
#include <chrono>
#include <inttypes.h>

#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/execution_policy.h>

typedef int64_t lng;
typedef uint64_t ulng;

//2^28 checks per launch means we need 2^20 launches
const int lg_launches = 20;
const int lg_blocks = 23;
const int lg_threads = 5; //fits in a warp perfectly!
const int num_launches = 1 << lg_launches;
const int num_blocks = 1 << lg_blocks;
const int num_threads = 1 << lg_threads;

const int spacing = 32;
const int separation = 5;
const int gap = spacing - separation;
const lng struct_seed = 10387313;

const lng mask = (1LL << 48) - 1LL;
const lng mask32 = (1LL << 32) - 1LL;
const lng mask16 = (1LL << 16) - 1LL;

const lng lcg1 = 0x5DEECE66DLL;
const lng lcg2 = 0xBLL;
const lng lcg3 = 341873128712LL;
const lng lcg4 = 132897987541LL;
const lng inv_lcg1 = 0xDFE05BCB1365LL;

const bool check_all = true;

__device__ __host__ lng set_seed(lng seed) {
    return (seed ^ lcg1) & mask;
}

__device__ __host__ lng update_seed(lng seed) {
    return (seed * lcg1 + lcg2) & mask;
}

__device__ __host__ int nextInt(lng &seed) {
    int bits;
    int val;
    do {
        seed = update_seed(seed);
        //unsigned to make this an arithmetic shift
        bits = (int) ((ulng)seed >> (48-31));
        val = bits % gap;
    } while (bits-val+gap-1 < 0);
    return val;
}

__device__ __host__ void rand4_for_seed(int arg1, int arg2, lng seed, int * buffer) {
    //first compute the input seed
    lng i = (lng)arg1 * lcg3 + (lng)arg2 * lcg4 + struct_seed + seed;
    seed = set_seed(i);

    for (int j = 0; j < 4; j++) {
        buffer[j] = nextInt(seed);
    }
}

__device__ __host__ bool check_chunk(int x, int z, lng world_seed) {
    int i = x;
    int j = z;

    if (x < 0) {
        x -= spacing - 1;
    }
    if (z < 0) {
        z -= spacing - 1;
    }

    //rounding should be the same as in java
    int k = x/spacing;
    int l = z/spacing;

    int rand4[4];
    rand4_for_seed(k, l, world_seed, rand4);

    k *= spacing;
    l *= spacing;

    k += (rand4[0] + rand4[1])/2;
    l += (rand4[2] + rand4[3])/2;
    return (i == k) && (j == l);
}

__device__ __host__ bool check_all_chunks(lng world_seed) {

    const int flat_chunks[] = {
        46, 48,
        -50, -47,
        -557, -115,
        -726, -115,
        41, 79,
        81, 229,
        70, 394,
    };
    for (unsigned int i = 0; i < sizeof(flat_chunks)/sizeof(int)/2; i++) {
        if (!check_chunk(flat_chunks[i*2], flat_chunks[i*2+1], world_seed)) {
            return false;
        }
    }
    return true;
}

__global__ void run_launch(int launch, bool * results) {
    lng seed = ((lng)launch << (lg_blocks+lg_threads)) + ((lng)blockIdx.x << lg_threads) + threadIdx.x;
    bool result = check_all_chunks(seed);
    bool any_found = __any(result);

    if (threadIdx.x == 0) {
        results[blockIdx.x] = any_found;
    }
}

lng seed_search() {

    lng compatible_seed = -1;
    
    bool *block_results;
    bool max_elem;
    cudaMalloc(&block_results, sizeof(bool)*num_blocks);

    auto t0 = std::chrono::high_resolution_clock::now();

    //can start at 624635 for debugging
    printf("num_launches:",num_launches);
    for (int launch = 0; launch < num_launches; launch++) {
        if (!(launch % (1 << 10))) {
            auto t1 = std::chrono::high_resolution_clock::now();
            long d0 = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
            t0 = t1;
            /*
            printf("seed %" PRIi64 ", launch %d, time %ld ms\n",
                   ((lng)launch) << (lg_blocks + lg_threads),
                   launch, d0);
            */
        }

        run_launch<<<num_blocks,num_threads>>>(launch, block_results);
        
        thrust::device_vector<bool> thrust_results(block_results, block_results+num_blocks); 
        auto max_idx = thrust::max_element(thrust_results.begin(), thrust_results.end());
        int index = max_idx - thrust_results.begin();

        cudaMemcpy(&max_elem, block_results + index, sizeof(bool), cudaMemcpyDeviceToHost);
        
        if (max_elem) {
            lng seedstart = ((lng)launch << (lg_blocks+lg_threads)) + ((lng)index << lg_threads);
            for (lng seed = seedstart; seed < seedstart+num_threads; seed++) {
                if (check_all_chunks(seed)) {
                    if (!check_all) {
                        return seed;
                    } else if (compatible_seed == -1) {
                        printf("found candidate %" PRIi64 "\n", seed);
                        compatible_seed = seed;
                    } else {
                        printf("%" PRIi64 "\n", compatible_seed);
                        printf("%" PRIi64 "\n", seed);
                        return -2;
                    }
                }
            }
        }
    }
    return compatible_seed;
}

void extend_seed(lng lower48) {
    //assert(((lcg1 * inv_lcg1) & mask) == 1LL);
    
    lng lower32 = lower48 & mask32;
    lng upper32lower16 = (lower48 >> 32) & mask16;

    for (lng i = 0; i < (1LL << 16); i++) {
        lng seed2 = (lower32 << 16) | i;
        lng seed1 = ((seed2-lcg2) * inv_lcg1) & mask;
        lng upper32 = seed1 >> 16;
        if (upper32lower16 == (upper32 & mask16)) {
            lng candidate = (upper32 << 32) | lower32;
            printf("candidate 64 bit extension: %" PRIi64 "\n", candidate);            
        }
    }
}


int main(int argc, char **argv)
{
    lng out = seed_search();
    //lng out = 167675086332377LL;
    //lng out = 255383536937374LL;
    
    if (out == -1) {
        printf("no compatible seeds found\n");
    } else if (out == -2) {
        printf("too many compatible seeds\n");
    } else {
        printf("lower 48 bits are %" PRIi64 "\n", out);
        extend_seed((lng) out);
    }
    return 0;
}