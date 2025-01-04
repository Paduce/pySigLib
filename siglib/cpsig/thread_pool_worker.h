#pragma once
#include "thread_pool_base.h"
#include <future>
#include <list>

template<class _Func>
class thread_pool_worker : public thread_pool_worker_base {
	std::packaged_task<_Func>* task;
	std::list<std::packaged_task<_Func>*> tasks;

	static std::packaged_task<_Func>* end() noexcept {
		static std::packaged_task<_Func> end;
		return &end;
	}

	static void worker_func(thread_pool_worker* w) {
		if (!w)
			return;
		std::unique_lock<std::mutex> lock(w->mtx);
		for (;;) {
			if (!w->task)
				w->cond.wait(lock, [=]()noexcept->bool { return w->task != nullptr; });
			if (w->task == end())
				break;
			(*w->task)();
			std::for_each(w->tasks.cbegin(), w->tasks.cend(), [](auto t) { if (t) (*t)(); });
			w->task = nullptr;
			w->tasks.clear();
			w->release();
		}
		w->task = nullptr;
		w->tasks.clear();
	}

public:
	thread_pool_worker() noexcept : task(nullptr) {}
	thread_pool_worker(const thread_pool_worker&) = delete;
	thread_pool_worker(thread_pool_worker&&) = delete;
	thread_pool_worker& operator=(const thread_pool_worker&) = delete;
	thread_pool_worker& operator=(thread_pool_worker&&) = delete;
	~thread_pool_worker() noexcept {
		try {
			stop();
		}
		catch (...) {}
	}

	void start() {
		std::lock_guard<std::mutex> lock(mtx);
		thread = std::make_unique<std::thread>(worker_func, this);
	}

	void stop() {
		{
			// wait for the current task to complete (if any)
			std::lock_guard<std::mutex> lock(mtx);
			task = end(); // assign end-task
		}
		cond.notify_one();
		std::this_thread::yield();
		if (thread && thread->joinable())
			thread->join();
		thread.reset(nullptr);
	}

	void assign(std::packaged_task<_Func>& tsk) {
		assert(this->task == nullptr);
		{
			std::lock_guard<std::mutex> lock(mtx);
			this->task = &tsk;
		}
		cond.notify_one();
	}

	void assign(std::list<std::packaged_task<_Func>*>& tsks) {
		assert(this->task == nullptr);
		{
			std::lock_guard<std::mutex> lock(mtx);
			this->task = tsks.front();
			tsks.pop_front();
			this->tasks = tsks;
		}
		cond.notify_one();
	}

};
