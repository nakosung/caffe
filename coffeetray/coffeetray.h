#pragma once

#ifndef COFFEETRAY_API
#define COFFEETRAY_API __declspec(dllimport)
#endif

extern "C" COFFEETRAY_API void coffeetray_init(const char* flags, void(*callback)(const char*));
extern "C" COFFEETRAY_API void* coffeetray_create();
extern "C" COFFEETRAY_API bool coffeetray_send(void* context, const char* command, unsigned int size, void* data);
extern "C" COFFEETRAY_API void coffeetray_destroy(void* context);