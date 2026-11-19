// Lock-free single-producer / single-consumer ring. One thread calls tryPush, one other thread
// calls tryPop; neither ever blocks or takes a lock. Capacity is rounded up to a power of two.
#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <new>
#include <utility>

namespace studio {

template <typename T>
class SpscRing {
public:
    explicit SpscRing(std::size_t capacity)
        : m_cap(roundUpPow2(capacity < 2 ? 2 : capacity)), m_mask(m_cap - 1), m_slots(new T[m_cap])
    {
    }
    SpscRing(const SpscRing &) = delete;
    SpscRing &operator=(const SpscRing &) = delete;

    std::size_t capacity() const { return m_cap; }

    // Producer thread only. False when full (the value is left untouched).
    bool tryPush(T &&v)
    {
        const std::size_t head = m_head.load(std::memory_order_relaxed);
        if (head - m_tailCache >= m_cap) {
            m_tailCache = m_tail.load(std::memory_order_acquire);
            if (head - m_tailCache >= m_cap)
                return false;
        }
        m_slots[head & m_mask] = std::move(v);
        m_head.store(head + 1, std::memory_order_release);
        return true;
    }
    bool tryPush(const T &v)
    {
        T copy = v;
        return tryPush(std::move(copy));
    }

    // Consumer thread only. False when empty.
    bool tryPop(T &out)
    {
        const std::size_t tail = m_tail.load(std::memory_order_relaxed);
        if (tail == m_headCache) {
            m_headCache = m_head.load(std::memory_order_acquire);
            if (tail == m_headCache)
                return false;
        }
        out = std::move(m_slots[tail & m_mask]);
        m_slots[tail & m_mask] = T();  // release whatever the slot held (strings, buffers)
        m_tail.store(tail + 1, std::memory_order_release);
        return true;
    }

    // Approximate from any thread; exact from the consumer when the producer is idle.
    std::size_t sizeApprox() const
    {
        return m_head.load(std::memory_order_acquire) - m_tail.load(std::memory_order_acquire);
    }
    bool emptyApprox() const { return sizeApprox() == 0; }

private:
    static std::size_t roundUpPow2(std::size_t n)
    {
        std::size_t p = 1;
        while (p < n)
            p <<= 1;
        return p;
    }

    static constexpr std::size_t kLine = 64;
    const std::size_t m_cap;
    const std::size_t m_mask;
    std::unique_ptr<T[]> m_slots;
    alignas(kLine) std::atomic<std::size_t> m_head{0};  // written by the producer
    alignas(kLine) std::size_t m_tailCache = 0;         // producer's view of m_tail
    alignas(kLine) std::atomic<std::size_t> m_tail{0};  // written by the consumer
    alignas(kLine) std::size_t m_headCache = 0;         // consumer's view of m_head
};

}  // namespace studio
