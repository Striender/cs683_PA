#include <iostream>
#include <vector>
#include <immintrin.h>
#include <chrono>
#include <cstdlib>
#include <cmath>

using namespace std;
using namespace std::chrono;

const int embedding_dim = 8; // multiple of 8 for easy printing
const float EPS = 1e-5;
int prefetch_distance = 8;

// === Your SIMD function exactly as-is ===
long long run_with_simd(vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {
   
    auto start = high_resolution_clock::now();
    vector<vector<float>> output;

    for (size_t i = 0; i < offsets.size(); ++i) {
        int start_idx = offsets[i];
        int end_idx = (i + 1 < offsets.size()) ? offsets[i + 1] : input.size();

        vector<float> bag_embedding(embedding_dim, 0.0f);

        int d = 0;
        for (; d <= embedding_dim - 8; d += 8)
        {
            __m256 sum = _mm256_setzero_ps();
            for (int j = start_idx; j < end_idx; ++j)
            {
                const float *data_ptr = &embedding_table[input[j] * embedding_dim];
                __m256 data = _mm256_loadu_ps(&data_ptr[d]);
                sum = _mm256_add_ps(sum, data);
            }
            _mm256_storeu_ps(&bag_embedding[d], sum);
        }
            
        output.push_back(bag_embedding);
    
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
     //cout<<"\nembedding size "<<embedding_dim<<endl;
    cout << "\nTime with Simd: " << duration.count() << " microseconds.";

    cout << "Simd Output:\n";
    for (auto &vec : output) {
        for (float x : vec) cout << x << " ";
        cout << "\n";
    }
    
    return duration.count();
}

// === Your Naive function exactly as-is ===
long long naive_emb(vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {

    auto start = high_resolution_clock::now();
    vector<vector<float>> output;

    for (size_t i = 0; i < offsets.size(); ++i) {
        int start_idx = offsets[i];
        int end_idx = (i + 1 < offsets.size()) ? offsets[i + 1] : input.size();

        vector<float> bag_embedding(embedding_dim, 0.0f);

        for (int j = start_idx; j < end_idx; ++j) {
            float* data_ptr = &embedding_table[input[j] * embedding_dim];
            for (int d = 0; d < embedding_dim; ++d) {
                bag_embedding[d] += data_ptr[d];
            }
        }

        output.push_back(bag_embedding);
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "\nTime WITHOUT software prefetching: " << duration.count() << " microseconds.";
    
     cout << "\nSimd Output:\n";
    for (auto &vec : output) {
        for (float x : vec) cout << x << " ";
        cout << "\n";
    }

    return duration.count();
}

long long run_with_prefetching_simd(const vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {

    auto start = high_resolution_clock::now();
     vector<vector<float>> output;

    for (size_t i = 0; i < offsets.size(); ++i) {
        int start_idx = offsets[i];
        int end_idx = (i + 1 < offsets.size()) ? offsets[i + 1] : input.size();

        vector<float> bag_embedding(embedding_dim, 0.0f);
            int d = 0;
        for (; d <= embedding_dim - 8; d += 8)
        {
            __m256 sum = _mm256_setzero_ps();
            for (int j = start_idx; j < end_idx; ++j)
            { if (j + prefetch_distance < end_idx) {
                _mm_prefetch((const char*)&embedding_table[input[j + prefetch_distance] * embedding_dim], _MM_HINT_T2);
            }
                const float *data_ptr = &embedding_table[input[j] * embedding_dim];
                __m256 data = _mm256_loadu_ps(&data_ptr[d]);
                sum = _mm256_add_ps(sum, data);
            }
            _mm256_storeu_ps(&bag_embedding[d], sum);
        }
        for (; d < embedding_dim; ++d)
        {
            float sum = 0.0f;
            for (int j = start_idx; j < end_idx; ++j)
            {
                const float *data_ptr = &embedding_table[input[j] * embedding_dim];
                sum += data_ptr[d];
            }
            bag_embedding[d] = sum;
        }
        
        output.push_back(bag_embedding);
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "\nTime WITH software prefetching + Simd: " << duration.count() << " microseconds.";

 cout << "\nSimd + prefetching Output:\n";
    for (auto &vec : output) {
        for (float x : vec) cout << x << " ";
        cout << "\n";
    }


    return duration.count();
}

// Utility to compare embeddings
bool compare_outputs(const vector<vector<float>>& a, const vector<vector<float>>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != b[i].size()) return false;
        for (size_t j = 0; j < a[i].size(); ++j) {
            if (fabs(a[i][j] - b[i][j]) > EPS) return false;
        }
    }
    return true;
}

int main() {
    int num_embeddings = 5;
    int num_indices = 6;

    vector<float> embedding_table(num_embeddings * embedding_dim);
    vector<int> input(num_indices);
    vector<int> offsets = {0, 2, 4, 6};

    // Fill with small random numbers
    for (auto &v : embedding_table) v = rand() % 5; // 0..4
    for (auto &v : input) v = rand() % num_embeddings;

    // Print embedding table & input
    cout << "Embedding table:\n";
    for (int i = 0; i < num_embeddings; ++i) {
        for (int d = 0; d < embedding_dim; ++d) cout << embedding_table[i * embedding_dim + d] << " ";
        cout << "\n";
    }

    cout << "\nInput indices:\n";
    for (auto x : input) cout << x << " ";
    cout << "\n";

    // Run both functions
    naive_emb(embedding_table, input, offsets);
    run_with_simd(embedding_table, input, offsets);
    run_with_prefetching_simd(embedding_table, input, offsets);

    return 0;
}
