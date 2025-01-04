#pragma once
#include <mutex>
#include <memory>
#include <cassert>

class thread_pool_worker_base;

class thread_pool_base {
	friend class thread_pool_worker_base;
protected:
	virtual void release(thread_pool_worker_base& w) = 0;
public:
	thread_pool_base() noexcept {}
	thread_pool_base(const thread_pool_base&) = delete;
	thread_pool_base(thread_pool_base&&) = delete;
	thread_pool_base& operator=(const thread_pool_base&) = delete;
	thread_pool_base& operator=(thread_pool_base&&) = delete;
	virtual ~thread_pool_base() noexcept {}
};

class thread_pool_worker_base {
protected:
	std::mutex mtx;
	std::condition_variable cond;
	thread_pool_base* parent_thread_pool;
	size_t index_in_the_pool;
	std::unique_ptr<std::thread> thread;

	void release() {
		if (parent_thread_pool)
			parent_thread_pool->release(*this);
	}

public:
	thread_pool_worker_base() noexcept : parent_thread_pool(nullptr),
		index_in_the_pool(0), thread(nullptr) {}
	thread_pool_worker_base(const thread_pool_worker_base&) = delete;
	thread_pool_worker_base(thread_pool_worker_base&&) = delete;
	thread_pool_worker_base& operator=(const thread_pool_worker_base&) = delete;
	thread_pool_worker_base& operator=(thread_pool_worker_base&&) = delete;
	virtual ~thread_pool_worker_base() noexcept {}

	size_t get_index_in_the_pool() const noexcept {
		return index_in_the_pool;
	}

	void init(thread_pool_base* pool, size_t i) noexcept {
		parent_thread_pool = pool;
		index_in_the_pool = i;
	}

};