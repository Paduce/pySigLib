#pragma once
#include <queue>
#include <list>
#include <mutex>
#include <condition_variable>

template <class _Ty, class _Container = std::queue<_Ty, std::list<_Ty>>>
class thread_safe_queue {
    std::mutex m;
    std::condition_variable c;
    _Container q;

public:
    using value_type = typename _Container::value_type;
    //using reference = typename _Container::reference;
    //using const_reference = typename _Container::const_reference;
    //using size_type = typename _Container::size_type;
    //using container_type = _Container;

    static_assert(std::is_same_v<_Ty, value_type>,
        "thread_safe_queue template parameters have inconsistent types");

    thread_safe_queue() noexcept {}

    // Test whether the queue is empty
    bool empty() const noexcept(noexcept(q.empty())) {
        return q.empty();
    }

    // Put element at the back of the queue
    void put(const _Ty& elem) {
        {
            std::lock_guard<decltype(m)> lck(m);
            q.push(elem);
        }
        c.notify_one();
    }

    // Put (move) element at the back of the queue
    void put(_Ty&& elem) {
        {
            std::lock_guard<decltype(m)> lck(m);
            q.push(std::move(elem));
        }
        c.notify_one();
    }

    // Try to get the first element without waiting.
    // Return true on success and false otherwise
    bool get(_Ty& elem) {
        return get(elem, std::chrono::seconds(0));
    }

    // Try to get the first element, wait for the specified period of time,
    // if the queue is empty. Return true on success and false otherwise
    template<class Rep, class Period>
    bool get(_Ty& elem, const std::chrono::duration<Rep, Period>& period) {
        std::unique_lock<decltype(m)> lck(m);
        if (c.wait_for(lck, period, [this]()noexcept->bool { return !q.empty(); })) {
            elem = std::move(q.front());
            q.pop();
            return true;
        }
        return false;
    }
}; 
