#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

class PeakRssMonitor {
public:
    PeakRssMonitor() {
        const std::uint64_t initial_rss_bytes = readCurrentRssBytes();
        if (initial_rss_bytes == 0) {
            throw std::runtime_error("Unable to read VmRSS from /proc/self/status");
        }

        peak_rss_bytes_.store(initial_rss_bytes, std::memory_order_relaxed);
        running_.store(true, std::memory_order_release);
        sampler_ = std::thread([this]() { sampleLoop(); });
    }

    ~PeakRssMonitor() {
        stop();
    }

    double stopGb() {
        stop();
        return static_cast<double>(peak_rss_bytes_.load(std::memory_order_relaxed)) /
               1000000000.0;
    }

private:
    static std::uint64_t readCurrentRssBytes() {
        std::ifstream status_file("/proc/self/status");
        std::string label;
        while (status_file >> label) {
            if (label == "VmRSS:") {
                std::uint64_t rss_kb = 0;
                std::string unit;
                if (status_file >> rss_kb >> unit && unit == "kB") {
                    return rss_kb * 1024;
                }
                return 0;
            }
            status_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
        return 0;
    }

    void sampleOnce() {
        const std::uint64_t current_rss_bytes = readCurrentRssBytes();
        std::uint64_t observed_peak = peak_rss_bytes_.load(std::memory_order_relaxed);
        while (current_rss_bytes > observed_peak &&
               !peak_rss_bytes_.compare_exchange_weak(
                   observed_peak, current_rss_bytes, std::memory_order_relaxed)) {
        }
    }

    void sampleLoop() {
        while (running_.load(std::memory_order_acquire)) {
            sampleOnce();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        sampleOnce();
    }

    void stop() {
        if (running_.exchange(false, std::memory_order_acq_rel) && sampler_.joinable()) {
            sampler_.join();
        }
    }

    std::atomic<bool> running_{false};
    std::atomic<std::uint64_t> peak_rss_bytes_{0};
    std::thread sampler_;
};

inline void printPeakRss(const std::string& label, double peak_rss_gb) {
    std::ostringstream message;
    message << std::fixed << std::setprecision(6)
            << label << ": " << peak_rss_gb << " GB";
    std::cout << message.str() << std::endl;
}
