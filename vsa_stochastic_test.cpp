#include "vsa_stochastic.h"
#include <iostream>
#include <chrono>
#include <iomanip>

// Helper to generate a random ternary hypervector
void make_random_hv(uint64_t* seed, vsa_hv_t* h) {
    uint64_t x = *seed;
    for (size_t w = 0; w < VSA_WORDS; ++w) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17; h->sign[w] = x;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17; h->zero[w] = x;
    }
    *seed = x;
}

// Add noise to a hypervector to simulate a noisy training signal (e.g., sensor noise)
void add_noise(const vsa_hv_t* in, vsa_hv_t* out, int noise_percent) {
    memcpy(out, in, sizeof(vsa_hv_t));
    for (size_t d = 0; d < STOCHASTIC_DIM; ++d) {
        if ((rand() % 100) < noise_percent) {
            size_t w = d / 64;
            size_t bit = d % 64;
            // Flip active state or sign state
            if (rand() % 2) {
                out->zero[w] ^= (1ULL << bit); // toggle active
            } else {
                out->sign[w] ^= (1ULL << bit); // toggle sign
            }
        }
    }
}

int main() {
    std::cout << "================ VSA STOCHASTIC TRAINING TEST ================\n";
    
    // Seed random generator
    uint64_t seed = 0x123456789ABCDEF0ULL;
    srand(42);
    
    // 1. Generate target knowledge hypervector (the "ideal state" the model must learn)
    vsa_hv_t target_hv;
    make_random_hv(&seed, &target_hv);
    int32_t target_active = vsa_active_count(&target_hv);
    std::cout << "Target hypervector generated: active dimensions = " << target_active << "/" << STOCHASTIC_DIM << "\n";
    
    // 2. Initialize 5-bit stochastic hypervector (the learner)
    vsa_stochastic_hv_t learner;
    vsa_stochastic_init(&learner);
    
    // 3. Training Loop Simulation
    const int total_steps = 1000;
    const int report_every = 100;
    const int probability_threshold = 40; // 40% chance of confidence nudge on signal
    const int noise_percent = 15;          // 15% noise injected into training signals
    
    std::cout << "\nStarting Stochastic Training Loop...\n";
    std::cout << "Parameters: Steps=" << total_steps 
              << ", ProbThreshold=" << probability_threshold << "%"
              << ", TrainingSignalNoise=" << noise_percent << "%\n\n";
              
    std::cout << "Step\tActive Dims\tDot Product w/ Target\tAccuracy %\n";
    std::cout << "--------------------------------------------------------\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step <= total_steps; ++step) {
        // Measure and print accuracy periodically
        if (step % report_every == 0) {
            vsa_hv_t current_inference_hv;
            vsa_stochastic_quantize(&learner, &current_inference_hv);
            int32_t dot = vsa_dot(&current_inference_hv, &target_hv);
            int32_t current_active = vsa_active_count(&current_inference_hv);
            
            // Accuracy is the fraction of matching active dimensions relative to the target's active dimensions
            double accuracy = 0.0;
            if (target_active > 0) {
                accuracy = (double)dot / target_active * 100.0;
                if (accuracy < 0.0) accuracy = 0.0;
            }
            
            std::cout << step << "\t" 
                      << current_active << "\t\t" 
                      << dot << "\t\t\t" 
                      << std::fixed << std::setprecision(2) << accuracy << "%\n";
        }
        
        if (step == total_steps) break;
        
        // Generate a noisy version of the target hypervector as our training signal (gradient)
        vsa_hv_t training_signal;
        add_noise(&target_hv, &training_signal, noise_percent);
        
        // Update the 5-bit learner stochastic parameters
        vsa_stochastic_update(&learner, &training_signal, probability_threshold);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    double time_per_update_us = (double)total_duration_us / total_steps;
    
    std::cout << "--------------------------------------------------------\n";
    std::cout << "Stochastic Training Complete!\n";
    std::cout << "Total training execution time: " << total_duration_us << " microseconds\n";
    std::cout << "Average CPU time per 10,240-D parameter update: " 
              << std::fixed << std::setprecision(3) << time_per_update_us << " microseconds (" 
              << (time_per_update_us * 1000.0) << " nanoseconds)\n";
              
    std::cout << "==============================================================\n";
    return 0;
}
