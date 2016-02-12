// coffeetray.cpp : Defines the exported functions for the DLL application.
//

#include "pch.h"

__pragma(optimize("",off))
#define COFFEETRAY_API __declspec(dllexport)
#include "coffeetray.h"
#include <boost/algorithm/string.hpp>
#include "google/protobuf/text_format.h"

using namespace boost;
using namespace caffe;

std::vector<int> gpus;

class Buffer {
public:
	Buffer(unsigned int size, void* data)
	:size(size), data(data)
	{}

	operator int() const {
		::_getpid();
		return size == sizeof(int) ? *reinterpret_cast<int*>(data) : -9999;
	}

	operator string() const {
		auto first = reinterpret_cast<const char*>(data);
		return string(first, first + size);
	}

	Buffer& operator = (int other) {
		*reinterpret_cast<int*>(data) = other;
		return *this;
	}

	template <typename T>
	Buffer& operator = (T* other) {
		*reinterpret_cast<T**>(data) = other;
		return *this;
	}

	template <typename T>
	Buffer& operator = (const T* other) {
		*reinterpret_cast<const T**>(data) = other;
		return *this;
	}
	
	unsigned int size;
	void* data;
};

class Coffeetray {
public:
	caffe::SolverParameter solver_param;
	caffe::shared_ptr<caffe::Solver<float> > solver;

	struct NoArgException {};

	bool send(std::string&& command, Buffer&& buffer) {
		try {
			std::vector<std::string> tokens;
			auto eat = [&]{
				tokens.erase(tokens.begin(), tokens.begin() + 1);
			};
			auto peek = [&]{
				if (tokens.size() == 0) {
					throw NoArgException();
				}
				return tokens[0];
			};
			auto match = [&](const std::string& rhs) {
				if (peek() == rhs) {
					eat();
					return true;
				}
				else {
					return false;
				}
			};
			auto layer_commands = [&](caffe::shared_ptr<Net<float>> net) {
				auto layer = solver->net()->layer_by_name(peek());
				eat();
				if (match("reset")) {
					auto mem_layer = boost::static_pointer_cast<caffe::MemoryDataLayer<float>>(layer);
					struct Pack {
						float* data;
						float* label;
						int n;
					};
					auto pack = reinterpret_cast<Pack*>(buffer.data);
					mem_layer->Reset(pack->data, pack->label, pack->n);
					return true;
				}
				return false;
			};
			auto blob_commands = [&](caffe::shared_ptr<Net<float>> net) {
				auto blob = net->blob_by_name(peek());
				eat();
				if (match("mutable_data")) {
					buffer = blob->mutable_cpu_data();
					return true;
				}
				else if (match("mutable_diff")) {
					buffer = blob->mutable_cpu_diff();
					return true;
				}
				else if (match("data")) {
					buffer = blob->cpu_data();
					return true;
				}
				else if (match("diff")) {
					buffer = blob->cpu_diff();
					return true;
				}
				return false;
			};
			boost::split(tokens, command, boost::is_any_of("\t "));
			if (match("solver_param")) {
				if (match("set_array")) {
					return solver_param.ParseFromArray(buffer.data, buffer.size);
				}
				else if (match("set_string")) {
					return google::protobuf::TextFormat::ParseFromString(buffer, &solver_param);
					//return solver_param.ParseFromString(buffer);
				}
				else if (match("get_bytesize")) {
					buffer = solver_param.ByteSize();
					return true;
				}
				else if (match("get")) {
					return solver_param.SerializeToArray(buffer.data, buffer.size);
				}
				else {
					return false;
				}
			}
			else if (match("solver")) {
				if (match("auto_gpu")) {
					if (gpus.size()) {
						solver_param.set_device_id(gpus[0]);
					}
					return true;
				}
				else if (match("create")) {					
					solver.reset(caffe::SolverRegistry<float>::CreateSolver(solver_param));
					return true;
				}
				else if (match("step")) {
					solver->Step(buffer);
					return true;
				}
				else if (match("layer")) {
					return layer_commands(solver->net());
				}
				else if (match("blob")) {
					return blob_commands(solver->net());
				}
			}
		}
		catch (NoArgException&) {
			return false;
		}
		return false;
	}
};

void get_gpus(vector<int>* gpus, const std::string& FLAGS_gpu) {
	if (FLAGS_gpu == "all") {
		int count = 0;
#ifndef CPU_ONLY
		CUDA_CHECK(cudaGetDeviceCount(&count));
#else
		NO_GPU;
#endif
		for (int i = 0; i < count; ++i) {
			gpus->push_back(i);
		}
	}
	else if (FLAGS_gpu.size()) {
		vector<string> strings;
		boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
		for (int i = 0; i < strings.size(); ++i) {
			gpus->push_back(boost::lexical_cast<int>(strings[i]));
		}
	}
	else {
		CHECK_EQ(gpus->size(), 0);
	}
}

static class CustomSink : public google::LogSink {
public:
	void(*callback)(const char*);
	virtual void send(google::LogSeverity severity, const char* full_filename,
		const char* base_filename, int line,
		const struct ::tm* tm_time,
		const char* message, size_t message_len) override{
		callback(ToString(severity, full_filename, line, tm_time, message, message_len).c_str());
	}
} custom_sink;

extern "C" COFFEETRAY_API void coffeetray_init(const char* flags, void(*callback)(const char*)) {
	custom_sink.callback = callback;
	static bool has_registered = false;
	if (!has_registered) {
		has_registered = true;
		google::AddLogSink(&custom_sink);
	}
	get_gpus(&gpus, std::string(flags));
	if (gpus.size() == 0) {
		LOG(INFO) << "Use CPU.";
		Caffe::set_mode(Caffe::CPU);
	}
	else {
		ostringstream s;
		for (int i = 0; i < gpus.size(); ++i) {
			s << (i ? ", " : "") << gpus[i];
		}
		LOG(INFO) << "Using GPUs " << s.str();

		Caffe::SetDevice(gpus[0]);
		Caffe::set_mode(Caffe::GPU);
		Caffe::set_solver_count(gpus.size());
	}
}

extern "C" COFFEETRAY_API void* coffeetray_create()
{
	return new Coffeetray();
}

extern "C" COFFEETRAY_API bool coffeetray_send(void* context, const char* command, unsigned int size, void* data)
{
	auto instance = reinterpret_cast<Coffeetray*>(context);
	return instance->send(command, Buffer(size, data));
}

extern "C" COFFEETRAY_API void coffeetray_destroy(void* context)
{
	auto instance = reinterpret_cast<Coffeetray*>(context);
	delete instance;
}