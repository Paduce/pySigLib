#pragma once
#include "thread_pool_worker.h"
#include "thread_safe_queue.h"
#include <vector>
#include <chrono>

template<typename _Ret = int>
class thread_pool : public thread_pool_base {
public:
	using RET = _Ret;
	using TASK = std::packaged_task<RET(void)>;
	using RESULT = std::future<RET>;

private:
	std::vector<thread_pool_worker<RET(void)>> workers;
	thread_safe_queue<size_t> free_workers;
	size_t current_size;

protected:
	void release(thread_pool_worker_base& w) override {
		free_workers.put(w.get_index_in_the_pool());
	}

public:
	thread_pool(size_t sz) : workers(sz), current_size(0) {}
	thread_pool(const thread_pool&) = delete;
	thread_pool(thread_pool&&) = delete;
	thread_pool& operator=(const thread_pool&) = delete;
	thread_pool& operator=(thread_pool&&) = delete;
	~thread_pool() noexcept {}

	size_t start(size_t sz) {
		if (current_size > 0)
			return 0; // call stop before restarting
		current_size = (sz > capacity()) ? capacity() : sz;
		for (size_t i = 0; i < current_size; ++i) {
			auto& w = workers.at(i);
			w.init(this, i);
			w.start();
			release(w);
		}
		return current_size;
	}

	void stop() {
		for (size_t i = 0; i < current_size; ++i) {
			size_t j;
			while (!free_workers.get(j, std::chrono::seconds(1))); // wait for the next worker-thread to become free
			workers.at(j).stop(); // and stop it
		}
		current_size = 0;
	}

	bool run(TASK& task) {
		return run(task, std::chrono::seconds(0));
	}

	template<class _Rep, class _Period>
	bool run(TASK& task, const std::chrono::duration<_Rep, _Period>& period) {
		size_t i;
		if (free_workers.get(i, period)) {
			workers.at(i).assign(task);
			return true;
		}
		return false;
	}

	bool run(std::list<TASK*>& tasks) {
		return run(tasks, std::chrono::seconds(0));
	}

	template<class _Rep, class _Period>
	bool run(std::list<TASK*>& tasks, const std::chrono::duration<_Rep, _Period>& period) {
		size_t i;
		if (free_workers.get(i, period)) {
			workers[i].assign(tasks);
			return true;
		}
		return false;
	}

	size_t capacity() const noexcept { return workers.size(); }

	size_t size() const noexcept { return current_size; }
};

inline int getMaxThreads() {
	static const int maxThreads = std::thread::hardware_concurrency();
	return maxThreads;
}
