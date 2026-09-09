// SPDX-License-Identifier: Apache-2.0
// Runs the SAME planner template as the AIV kernel, with checked host storage.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>
#include "csrc/attention/turbo_quant_sparse_flash_attention/turbo_quant_group_plan.h"

using namespace tq_group;
static void Require(bool condition, const char *message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}
struct Storage {
    uint32_t k;
    std::vector<int32_t> raw, keys, positions, ids, owners, publishedIds, publishedOwners;
    std::array<std::array<int32_t, DESC_WORDS>, MAX_GROUP> desc{};
    explicit Storage(uint32_t width)
        : k(width), raw(MAX_GROUP * k, -1), keys(HASH_SIZE, -1), positions(HASH_SIZE),
          ids(QUAD_CAP), owners(QUAD_CAP), publishedIds(MAX_GROUP * k, -99),
          publishedOwners(MAX_GROUP * k, -99) {}
    void ResetHash() { std::fill(keys.begin(), keys.end(), -1); }
    int32_t ReadIndex(uint32_t i) { return raw.at(i); }
    int32_t HashKey(uint32_t i) { return keys.at(i); }
    uint32_t HashPosition(uint32_t i) { return positions.at(i); }
    uint32_t Owner(uint32_t i) { return owners.at(i); }
    void SetHash(uint32_t i, int32_t id, uint32_t pos) { keys.at(i) = id; positions.at(i) = pos; }
    void SetId(uint32_t i, int32_t id) { ids.at(i) = id; }
    void SetOwner(uint32_t i, uint32_t bit) { owners.at(i) = bit; }
    void Describe(uint32_t q, int32_t kind, uint32_t size, uint32_t length)
    {
        desc.at(q).fill(0);
        desc.at(q)[KIND] = kind;
        desc.at(q)[SIZE] = size;
        desc.at(q)[LENGTH] = length;
    }
    void Publish(uint32_t first, uint32_t length)
    {
        for (uint32_t i = 0; i < length; ++i) {
            publishedIds.at(first * k + i) = ids.at(i);
            publishedOwners.at(first * k + i) = owners.at(i);
        }
    }
};

static std::vector<int32_t> Effective(const Storage &s, uint32_t q, int64_t kv, int64_t len,
                                    uint32_t local, uint32_t mode)
{
    const int64_t end = mode == 3 ? kv - len + local + q + 1 : kv;
    std::vector<int32_t> result;
    for (uint32_t j = 0; j < s.k; ++j) {
        const int32_t id = s.raw.at(q * s.k + j);
        if (id >= 0 && id < kv && id < end) {
            result.push_back(id);
        }
    }
    return result;
}
// Independent scalar attention with nonuniform logits/values; multiplicity is
// preserved. The planner is exact in selection, not bitwise summation order.
static double Attention(const std::vector<int32_t> &ids, uint32_t query)
{
    double numerator = 0, denominator = 0;
    for (int32_t id : ids) {
        const double score = std::sin(double(id) * .13 + query);
        const double value = std::cos(double(id) * .07 - query);
        const double weight = std::exp(score);
        numerator += weight * value;
        denominator += weight;
    }
    return denominator == 0 ? 0 : numerator / denominator;
}
static void Validate(const Storage &s, uint32_t count, int64_t kv, int64_t len,
                     uint32_t local, uint32_t mode)
{
    std::array<uint32_t, MAX_GROUP> writes{};
    for (uint32_t first = 0; first < count; ++first) {
        const auto &d = s.desc[first];
        if (d[KIND] == SKIP) {
            continue;
        }
        const uint32_t size = d[SIZE];
        Require(size == 1 || size == 2 || size == 4, "bad group size");
        Require(first + size <= count, "crossed window/request boundary");
        Require(d[LENGTH] <= int32_t(size * s.k), "union exceeds its reserved query range");
        for (uint32_t q = 0; q < size; ++q) {
            ++writes[first + q];
            auto expected = Effective(s, first + q, kv, len, local, mode);
            std::vector<int32_t> actual;
            if (d[KIND] == LEGACY) {
                Require(size == 1, "legacy is not a singleton");
                actual = expected;
            } else if (d[KIND] == GROUP) {
                std::set<int32_t> unique;
                for (int32_t j = 0; j < d[LENGTH]; ++j) {
                    const uint32_t offset = first * s.k + j;
                    const int32_t id = s.publishedIds.at(offset);
                    const uint32_t owner = s.publishedOwners.at(offset);
                    Require(owner > 0 && owner < (1U << size), "bad owner bits");
                    Require(unique.insert(id).second, "duplicate union ID");
                    if (owner & (1U << q)) {
                        actual.push_back(id);
                    }
                }
            } else {
                Require(d[KIND] == ZERO && expected.empty(), "incorrect zero task");
            }
            Require(std::abs(Attention(expected, first + q) - Attention(actual, first + q)) < 1e-12,
                    "masked attention changed");
            std::sort(expected.begin(), expected.end());
            std::sort(actual.begin(), actual.end());
            Require(expected == actual, "membership differs from original effective TopK");
        }
    }
    for (uint32_t q = 0; q < count; ++q) {
        Require(writes[q] == 1, "query omitted or written twice");
    }
}
static void Run(Storage &s, uint32_t count, int64_t kv, int64_t len, uint32_t local, uint32_t mode,
                uint32_t quadCap = QUAD_CAP, uint32_t pairCap = PAIR_CAP)
{
    PlanWindow(s, count, s.k, kv, len, local, mode, quadCap, pairCap);
    Validate(s, count, kv, len, local, mode);
}
static void FillRange(Storage &s, uint32_t q, int32_t start)
{
    for (uint32_t j = 0; j < s.k; ++j) {
        s.raw[q * s.k + j] = start + j;
    }
}
int main()
{
    try {
        Storage small(8);
        for (uint32_t q = 0; q < MAX_GROUP; ++q) FillRange(small, q, 0);
        // Includes zero-length requests, long nonmultiples, mixed batches and
        // the exact same storage reused for subsequent replay-like plans.
        for (uint32_t length : {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 9U, 127U, 129U, 4095U, 4097U}) {
            for (uint32_t first = 0; first < length; first += MAX_GROUP) {
                Run(small, std::min(MAX_GROUP, length - first), 8192, length, first, 3);
            }
        }
        Run(small, 4, 0, 4, 0, 3);
        Run(small, 4, 2, 4, 0, 3); // early queries have no visible KV
        Run(small, 3, 10, 3, 0, 0);
        Require(small.desc[0][SIZE] == 2 && small.desc[2][SIZE] == 1, "3 != 2+1");

        Storage full(MAX_TOPK);
        for (uint32_t q = 0; q < MAX_GROUP; ++q) FillRange(full, q, 0);
        Run(full, 4, 20000, 4, 0, 3);
        Require(full.desc[0][SIZE] == 4 && full.desc[0][LENGTH] == 2048, "identical sets not grouped");
        FillRange(full, 2, 4096); FillRange(full, 3, 4096);
        Run(full, 4, 20000, 4, 0, 3);
        Require(full.desc[0][SIZE] == 2 && full.desc[2][SIZE] == 2, "quad did not fall back to pairs");
        for (uint32_t q = 0; q < MAX_GROUP; ++q) FillRange(full, q, q * 4096);
        Run(full, 4, 20000, 4, 0, 0);
        for (const auto &d : full.desc) Require(d[KIND] == LEGACY, "disjoint sets did not fall back");
        FillRange(full, 0, 0); FillRange(full, 1, 512);
        Run(full, 2, 20000, 2, 0, 0);
        Require(full.desc[0][LENGTH] == PAIR_CAP, "pair cap boundary");
        full.raw[full.k] = 19000; // replace an overlap, adding one unique ID
        Run(full, 2, 20000, 2, 0, 0);
        Require(full.desc[0][KIND] == LEGACY, "pair cap+1 was truncated");
        for (uint32_t q = 0; q < MAX_GROUP; ++q) FillRange(full, q, (q % 2) * 1024);
        Run(full, 4, 20000, 4, 0, 0);
        Require(full.desc[0][LENGTH] == QUAD_CAP, "quad cap boundary");
        full.raw.back() = 19000;
        Run(full, 4, 20000, 4, 0, 0);
        Require(full.desc[0][SIZE] != 4, "quad cap+1 was truncated");

        Storage collision(64);
        for (uint32_t q = 0; q < MAX_GROUP; ++q) {
            for (uint32_t j = 0; j < collision.k; ++j) collision.raw[q * collision.k + j] = 16777217 + j * HASH_SIZE;
        }
        Run(collision, 4, INT32_MAX, 4, 0, 0);
        Require(collision.desc[0][SIZE] == 4, "integer precision/hash collision failure");
        collision.raw[1] = collision.raw[0];
        Run(collision, 4, INT32_MAX, 4, 0, 0);
        Require(collision.desc[0][KIND] == LEGACY, "duplicate multiplicity was lost");

        std::mt19937 rng(20260909);
        for (uint32_t trial = 0; trial < 2000; ++trial) {
            const uint32_t count = 1 + rng() % MAX_GROUP;
            const uint32_t mode = trial % 2 == 0 ? 0 : 3;
            for (auto &id : small.raw) id = int32_t(rng() % 24) - 3;
            Run(small, count, trial % 5 == 0 ? 0 : 20, count + 3, 3, mode, 12, 10);
        }
        // 512-column boundary/full-mask example: each query owns one whole tile.
        Storage tiles(512);
        for (uint32_t q = 0; q < MAX_GROUP; ++q) FillRange(tiles, q, q * 512);
        Run(tiles, 4, 8192, 4, 0, 0);
        Require(tiles.desc[0][SIZE] == 4 && tiles.desc[0][LENGTH] == 2048, "tile membership case");
        std::cout << "PASS: planner membership, numeric reference, tails, replay reuse, hash collisions, "
                     "duplicates, empty rows, capacity boundaries and 2000 randomized cases\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
