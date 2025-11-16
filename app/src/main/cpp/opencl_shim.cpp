/**
 * OpenCL Shim Library for Android
 * 
 * This library provides a bridge between llama.cpp (which expects static linking)
 * and Android's runtime-loaded OpenCL library.
 * 
 * It uses dlopen/dlsym to load libOpenCL.so at runtime and exports all OpenCL
 * functions that llama.cpp needs.
 */

#include <dlfcn.h>
#include <string.h>  // memset
#include <unistd.h>  // usleep
#include <android/log.h>
#include <android/dlext.h>  // android_dlopen_ext for namespace bypass

#define LOG_TAG "OpenCLShim"
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define ALOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

// OpenCL types and constants (minimal set needed for function pointers)
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_mem;
typedef void* cl_program;
typedef void* cl_kernel;
typedef void* cl_event;
typedef unsigned int cl_uint;
typedef int cl_int;
typedef unsigned long cl_ulong;
typedef unsigned long cl_bitfield;
typedef size_t cl_size_t;

// OpenCL error codes (minimal set)
#define CL_SUCCESS 0
#define CL_DEVICE_NOT_FOUND -1

// OpenCL function pointer types
#define DECLARE_OPENCL_FUNC(name, ret, ...) \
    typedef ret (*name##_fn)(__VA_ARGS__); \
    static name##_fn name##_ptr = nullptr;

// Core platform/device functions
DECLARE_OPENCL_FUNC(clGetPlatformIDs, cl_int, cl_uint, cl_platform_id*, cl_uint*)
DECLARE_OPENCL_FUNC(clGetPlatformInfo, cl_int, cl_platform_id, cl_uint, cl_size_t, void*, cl_size_t*)
DECLARE_OPENCL_FUNC(clGetDeviceIDs, cl_int, cl_platform_id, cl_bitfield, cl_uint, cl_device_id*, cl_uint*)
DECLARE_OPENCL_FUNC(clGetDeviceInfo, cl_int, cl_device_id, cl_uint, cl_size_t, void*, cl_size_t*)

// Context functions - callback types
typedef void (*cl_context_callback)(const char*, const void*, size_t, void*);
typedef cl_context (*clCreateContext_fn)(const void*, cl_uint, const cl_device_id*, cl_context_callback, void*, cl_int*);
typedef cl_context (*clCreateContextFromType_fn)(const void*, cl_bitfield, cl_context_callback, void*, cl_int*);
static clCreateContext_fn clCreateContext_ptr = nullptr;
static clCreateContextFromType_fn clCreateContextFromType_ptr = nullptr;
DECLARE_OPENCL_FUNC(clRetainContext, cl_int, cl_context)
DECLARE_OPENCL_FUNC(clReleaseContext, cl_int, cl_context)

// Command queue functions
DECLARE_OPENCL_FUNC(clCreateCommandQueue, cl_command_queue, cl_context, cl_device_id, cl_bitfield, cl_int*)
DECLARE_OPENCL_FUNC(clRetainCommandQueue, cl_int, cl_command_queue)
DECLARE_OPENCL_FUNC(clReleaseCommandQueue, cl_int, cl_command_queue)
DECLARE_OPENCL_FUNC(clFlush, cl_int, cl_command_queue)
DECLARE_OPENCL_FUNC(clFinish, cl_int, cl_command_queue)

// Memory functions
DECLARE_OPENCL_FUNC(clCreateBuffer, cl_mem, cl_context, cl_bitfield, cl_size_t, void*, cl_int*)
DECLARE_OPENCL_FUNC(clRetainMemObject, cl_int, cl_mem)
DECLARE_OPENCL_FUNC(clReleaseMemObject, cl_int, cl_mem)
DECLARE_OPENCL_FUNC(clEnqueueReadBuffer, cl_int, cl_command_queue, cl_mem, cl_int, cl_size_t, cl_size_t, void*, cl_uint, const cl_event*, cl_event*)
DECLARE_OPENCL_FUNC(clEnqueueWriteBuffer, cl_int, cl_command_queue, cl_mem, cl_int, cl_size_t, cl_size_t, const void*, cl_uint, const cl_event*, cl_event*)
DECLARE_OPENCL_FUNC(clEnqueueCopyBuffer, cl_int, cl_command_queue, cl_mem, cl_mem, cl_size_t, cl_size_t, cl_size_t, cl_uint, const cl_event*, cl_event*)
DECLARE_OPENCL_FUNC(clEnqueueFillBuffer, cl_int, cl_command_queue, cl_mem, const void*, cl_size_t, cl_size_t, cl_size_t, cl_uint, const cl_event*, cl_event*)
DECLARE_OPENCL_FUNC(clEnqueueMapBuffer, void*, cl_command_queue, cl_mem, cl_int, cl_bitfield, cl_size_t, cl_size_t, cl_uint, const cl_event*, cl_event*, cl_int*)
DECLARE_OPENCL_FUNC(clEnqueueUnmapMemObject, cl_int, cl_command_queue, cl_mem, void*, cl_uint, const cl_event*, cl_event*)
// clCreateImage has different signatures in OpenCL 1.2 vs 2.0+
// Using OpenCL 2.0 signature: clCreateImage(context, flags, image_format, image_desc, host_ptr, errcode_ret)
typedef cl_mem (*clCreateImage_fn)(cl_context, cl_bitfield, const void*, const void*, void*, cl_int*);
static clCreateImage_fn clCreateImage_ptr = nullptr;
DECLARE_OPENCL_FUNC(clCreateSubBuffer, cl_mem, cl_mem, cl_bitfield, cl_uint, void*, cl_int*)
DECLARE_OPENCL_FUNC(clEnqueueBarrierWithWaitList, cl_int, cl_command_queue, cl_uint, const cl_event*, cl_event*)
DECLARE_OPENCL_FUNC(clEnqueueMarkerWithWaitList, cl_int, cl_command_queue, cl_uint, const cl_event*, cl_event*)

// Program and kernel functions
DECLARE_OPENCL_FUNC(clCreateProgramWithSource, cl_program, cl_context, cl_uint, const char**, const cl_size_t*, cl_int*)
DECLARE_OPENCL_FUNC(clCreateProgramWithBinary, cl_program, cl_context, cl_uint, const cl_device_id*, const cl_size_t*, const unsigned char**, cl_int*, cl_int*)
DECLARE_OPENCL_FUNC(clRetainProgram, cl_int, cl_program)
DECLARE_OPENCL_FUNC(clReleaseProgram, cl_int, cl_program)
// clBuildProgram with callback
typedef void (*cl_program_callback)(cl_program, void*);
typedef cl_int (*clBuildProgram_fn)(cl_program, cl_uint, const cl_device_id*, const char*, cl_program_callback, void*);
static clBuildProgram_fn clBuildProgram_ptr = nullptr;
DECLARE_OPENCL_FUNC(clGetProgramBuildInfo, cl_int, cl_program, cl_device_id, cl_uint, cl_size_t, void*, cl_size_t*)
DECLARE_OPENCL_FUNC(clCreateKernel, cl_kernel, cl_program, const char*, cl_int*)
DECLARE_OPENCL_FUNC(clRetainKernel, cl_int, cl_kernel)
DECLARE_OPENCL_FUNC(clReleaseKernel, cl_int, cl_kernel)
DECLARE_OPENCL_FUNC(clSetKernelArg, cl_int, cl_kernel, cl_uint, cl_size_t, const void*)

// Execution functions
DECLARE_OPENCL_FUNC(clEnqueueNDRangeKernel, cl_int, cl_command_queue, cl_kernel, cl_uint, const cl_size_t*, const cl_size_t*, const cl_size_t*, cl_uint, const cl_event*, cl_event*)
DECLARE_OPENCL_FUNC(clEnqueueTask, cl_int, cl_command_queue, cl_kernel, cl_uint, const cl_event*, cl_event*)

// Event functions
DECLARE_OPENCL_FUNC(clWaitForEvents, cl_int, cl_uint, const cl_event*)
DECLARE_OPENCL_FUNC(clRetainEvent, cl_int, cl_event)
DECLARE_OPENCL_FUNC(clReleaseEvent, cl_int, cl_event)
DECLARE_OPENCL_FUNC(clGetEventInfo, cl_int, cl_event, cl_uint, cl_size_t, void*, cl_size_t*)

// Additional functions that llama.cpp might use
DECLARE_OPENCL_FUNC(clGetKernelWorkGroupInfo, cl_int, cl_kernel, cl_device_id, cl_uint, cl_size_t, void*, cl_size_t*)
DECLARE_OPENCL_FUNC(clGetMemObjectInfo, cl_int, cl_mem, cl_uint, cl_size_t, void*, cl_size_t*)

static void* opencl_lib_handle = nullptr;
static bool opencl_initialized = false;

// Helper macro to load a function
#define LOAD_OPENCL_FUNC(name) \
    do { \
        name##_ptr = (name##_fn)dlsym(opencl_lib_handle, #name); \
        if (!name##_ptr) { \
            ALOGW("Failed to load " #name ": %s", dlerror()); \
        } else { \
            ALOGI("Successfully loaded " #name " function pointer"); \
        } \
    } while(0)

// Initialize OpenCL library
static bool init_opencl() {
    // Check if already initialized AND critical functions are loaded
    if (opencl_initialized && clGetPlatformIDs_ptr) {
        return true;
    }
    // If initialized but critical function missing, reset and retry
    if (opencl_initialized && !clGetPlatformIDs_ptr) {
        ALOGW("init_opencl: Critical function missing, reinitializing...");
        opencl_initialized = false;
        if (opencl_lib_handle) {
            dlclose(opencl_lib_handle);
            opencl_lib_handle = nullptr;
        }
    }

    // Strategy: Use android_dlopen_ext to bypass Android namespace restrictions
    // 1. First try standard dlopen (may work on some devices)
    opencl_lib_handle = dlopen("libOpenCL.so", RTLD_NOW | RTLD_GLOBAL);
    if (opencl_lib_handle) {
        ALOGI("Successfully loaded libOpenCL.so with standard dlopen");
        return true;
    }
    ALOGW("Standard dlopen failed: %s. Trying android_dlopen_ext...", dlerror());

    // 2. Use android_dlopen_ext to load from system namespaces
    // Try to dynamically load android_get_exported_namespace function
    typedef android_namespace_t* (*android_get_exported_namespace_fn)(const char*);
    android_get_exported_namespace_fn get_namespace_fn = nullptr;
    
    // Try to get the function from libdl.so or libandroid.so
    void* dl_handle = dlopen("libdl.so", RTLD_LAZY);
    if (dl_handle) {
        get_namespace_fn = (android_get_exported_namespace_fn)dlsym(dl_handle, "android_get_exported_namespace");
        if (!get_namespace_fn) {
            dlclose(dl_handle);
            dl_handle = dlopen("libandroid.so", RTLD_LAZY);
            if (dl_handle) {
                get_namespace_fn = (android_get_exported_namespace_fn)dlsym(dl_handle, "android_get_exported_namespace");
            }
        }
    }
    
    if (get_namespace_fn) {
        android_dlextinfo extinfo;
        memset(&extinfo, 0, sizeof(extinfo));
        extinfo.flags = ANDROID_DLEXT_USE_NAMESPACE;

        // Try "sphal" namespace (System-as-root devices)
        android_namespace_t* sphal_namespace = get_namespace_fn("sphal");
        if (sphal_namespace) {
            extinfo.library_namespace = sphal_namespace;
            opencl_lib_handle = android_dlopen_ext("libOpenCL.so", RTLD_NOW | RTLD_GLOBAL, &extinfo);
            if (opencl_lib_handle) {
                ALOGI("Successfully loaded libOpenCL.so from 'sphal' namespace using android_dlopen_ext");
                if (dl_handle) dlclose(dl_handle);
                // Continue to function pointer loading (don't return here)
                goto load_functions;
            }
            ALOGW("Failed to load from 'sphal' namespace: %s", dlerror());
        }

        // Try "vendor" namespace
        android_namespace_t* vendor_namespace = get_namespace_fn("vendor");
        if (vendor_namespace) {
            extinfo.library_namespace = vendor_namespace;
            opencl_lib_handle = android_dlopen_ext("libOpenCL.so", RTLD_NOW | RTLD_GLOBAL, &extinfo);
            if (opencl_lib_handle) {
                ALOGI("Successfully loaded libOpenCL.so from 'vendor' namespace using android_dlopen_ext");
                if (dl_handle) dlclose(dl_handle);
                goto load_functions;
            }
            ALOGW("Failed to load from 'vendor' namespace: %s", dlerror());
        }

        // Try "default" namespace as last resort
        android_namespace_t* default_namespace = get_namespace_fn("default");
        if (default_namespace) {
            extinfo.library_namespace = default_namespace;
            opencl_lib_handle = android_dlopen_ext("libOpenCL.so", RTLD_NOW | RTLD_GLOBAL, &extinfo);
            if (opencl_lib_handle) {
                ALOGI("Successfully loaded libOpenCL.so from 'default' namespace using android_dlopen_ext");
                if (dl_handle) dlclose(dl_handle);
                goto load_functions;
            }
            ALOGW("Failed to load from 'default' namespace: %s", dlerror());
        }
        
        if (dl_handle) dlclose(dl_handle);
    } else {
        ALOGW("android_get_exported_namespace not available - trying android_dlopen_ext without namespace");
        // Fallback: Try android_dlopen_ext without namespace (may work on some devices)
        android_dlextinfo extinfo;
        memset(&extinfo, 0, sizeof(extinfo));
        opencl_lib_handle = android_dlopen_ext("libOpenCL.so", RTLD_NOW | RTLD_GLOBAL, &extinfo);
        if (opencl_lib_handle) {
            ALOGI("Successfully loaded libOpenCL.so using android_dlopen_ext (no namespace)");
            goto load_functions;
        }
    }
    
    if (!opencl_lib_handle) {
        ALOGE("Failed to load libOpenCL.so from any path");
        return false;
    }

load_functions:

    ALOGI("libOpenCL.so loaded successfully, handle=%p", opencl_lib_handle);
    
    // Load all OpenCL functions
    ALOGI("Loading OpenCL function pointers from libOpenCL.so...");
    LOAD_OPENCL_FUNC(clGetPlatformIDs);
    if (!clGetPlatformIDs_ptr) {
        ALOGE("CRITICAL: Failed to load clGetPlatformIDs - OpenCL will not work!");
        return false;
    }
    LOAD_OPENCL_FUNC(clGetPlatformInfo);
    LOAD_OPENCL_FUNC(clGetDeviceIDs);
    LOAD_OPENCL_FUNC(clGetDeviceInfo);
    clCreateContext_ptr = (clCreateContext_fn)dlsym(opencl_lib_handle, "clCreateContext");
    clCreateContextFromType_ptr = (clCreateContextFromType_fn)dlsym(opencl_lib_handle, "clCreateContextFromType");
    LOAD_OPENCL_FUNC(clRetainContext);
    LOAD_OPENCL_FUNC(clReleaseContext);
    LOAD_OPENCL_FUNC(clCreateCommandQueue);
    LOAD_OPENCL_FUNC(clRetainCommandQueue);
    LOAD_OPENCL_FUNC(clReleaseCommandQueue);
    LOAD_OPENCL_FUNC(clFlush);
    LOAD_OPENCL_FUNC(clFinish);
    LOAD_OPENCL_FUNC(clCreateBuffer);
    LOAD_OPENCL_FUNC(clRetainMemObject);
    LOAD_OPENCL_FUNC(clReleaseMemObject);
    LOAD_OPENCL_FUNC(clEnqueueReadBuffer);
    LOAD_OPENCL_FUNC(clEnqueueWriteBuffer);
    LOAD_OPENCL_FUNC(clEnqueueCopyBuffer);
    LOAD_OPENCL_FUNC(clEnqueueFillBuffer);
    LOAD_OPENCL_FUNC(clEnqueueMapBuffer);
    LOAD_OPENCL_FUNC(clEnqueueUnmapMemObject);
    clCreateImage_ptr = (clCreateImage_fn)dlsym(opencl_lib_handle, "clCreateImage");
    LOAD_OPENCL_FUNC(clCreateSubBuffer);
    LOAD_OPENCL_FUNC(clEnqueueBarrierWithWaitList);
    LOAD_OPENCL_FUNC(clEnqueueMarkerWithWaitList);
    LOAD_OPENCL_FUNC(clCreateProgramWithSource);
    LOAD_OPENCL_FUNC(clCreateProgramWithBinary);
    LOAD_OPENCL_FUNC(clRetainProgram);
    LOAD_OPENCL_FUNC(clReleaseProgram);
    clBuildProgram_ptr = (clBuildProgram_fn)dlsym(opencl_lib_handle, "clBuildProgram");
    LOAD_OPENCL_FUNC(clGetProgramBuildInfo);
    LOAD_OPENCL_FUNC(clCreateKernel);
    LOAD_OPENCL_FUNC(clRetainKernel);
    LOAD_OPENCL_FUNC(clReleaseKernel);
    LOAD_OPENCL_FUNC(clSetKernelArg);
    LOAD_OPENCL_FUNC(clEnqueueNDRangeKernel);
    LOAD_OPENCL_FUNC(clEnqueueTask);
    LOAD_OPENCL_FUNC(clWaitForEvents);
    LOAD_OPENCL_FUNC(clRetainEvent);
    LOAD_OPENCL_FUNC(clReleaseEvent);
    LOAD_OPENCL_FUNC(clGetEventInfo);
    LOAD_OPENCL_FUNC(clGetKernelWorkGroupInfo);
    LOAD_OPENCL_FUNC(clGetMemObjectInfo);

    // Only mark as initialized if all critical functions were loaded successfully
    if (!clGetPlatformIDs_ptr) {
        ALOGE("init_opencl: Critical function clGetPlatformIDs not loaded - initialization failed");
        return false;
    }
    
    opencl_initialized = true;
    ALOGI("OpenCL shim initialized successfully");
    return true;
}

// Constructor: Initialize when library is loaded
__attribute__((constructor))
static void shim_init() {
    init_opencl();
}

// Destructor: Cleanup when library is unloaded
__attribute__((destructor))
static void shim_cleanup() {
    if (opencl_lib_handle) {
        dlclose(opencl_lib_handle);
        opencl_lib_handle = nullptr;
        opencl_initialized = false;
    }
}

// Export all OpenCL functions as C symbols
extern "C" {

// Platform/Device functions with retry logic for Android GPU driver initialization
cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms) {
    if (!init_opencl() || !clGetPlatformIDs_ptr) {
        ALOGE("clGetPlatformIDs: OpenCL not initialized or function not available");
        return -1;
    }
    
    // Retry logic for Android GPU driver initialization
    // GPU drivers may need time to initialize, especially after app launch or system boot
    cl_int err;
    int max_retries = 3;
    int retry_delay_ms = 100;  // 100ms delay between retries
    
    for (int attempt = 1; attempt <= max_retries; attempt++) {
        err = clGetPlatformIDs_ptr(num_entries, platforms, num_platforms);
        
        if (err == CL_SUCCESS) {
            if (num_platforms && *num_platforms > 0) {
                if (attempt > 1) {
                    ALOGI("clGetPlatformIDs succeeded on attempt %d (found %u platforms)", 
                          attempt, *num_platforms);
                }
                return CL_SUCCESS;
            } else {
                ALOGW("clGetPlatformIDs returned CL_SUCCESS but found 0 platforms (attempt %d)", attempt);
                // Continue to retry if no platforms found
            }
        } else {
            ALOGW("clGetPlatformIDs failed with error %d on attempt %d", err, attempt);
        }
        
        // Wait before retry (except on last attempt)
        if (attempt < max_retries) {
            // Use usleep for microsecond precision (100ms = 100000 microseconds)
            usleep(retry_delay_ms * 1000);
            retry_delay_ms *= 2;  // Exponential backoff: 100ms, 200ms, 400ms
        }
    }
    
    ALOGE("clGetPlatformIDs failed after %d attempts with final error %d", max_retries, err);
    return err;
}

cl_int clGetPlatformInfo(cl_platform_id platform, cl_uint param_name, cl_size_t param_value_size, void* param_value, cl_size_t* param_value_size_ret) {
    if (!init_opencl() || !clGetPlatformInfo_ptr) return -1;
    return clGetPlatformInfo_ptr(platform, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int clGetDeviceIDs(cl_platform_id platform, cl_bitfield device_type, cl_uint num_entries, cl_device_id* devices, cl_uint* num_devices) {
    if (!init_opencl() || !clGetDeviceIDs_ptr) return -1;
    return clGetDeviceIDs_ptr(platform, device_type, num_entries, devices, num_devices);
}

cl_int clGetDeviceInfo(cl_device_id device, cl_uint param_name, cl_size_t param_value_size, void* param_value, cl_size_t* param_value_size_ret) {
    if (!init_opencl() || !clGetDeviceInfo_ptr) return -1;
    return clGetDeviceInfo_ptr(device, param_name, param_value_size, param_value, param_value_size_ret);
}

// Context functions
cl_context clCreateContext(const void* properties, cl_uint num_devices, const cl_device_id* devices, void (*pfn_notify)(const char*, const void*, size_t, void*), void* user_data, cl_int* errcode_ret) {
    if (!init_opencl() || !clCreateContext_ptr) {
        if (errcode_ret) *errcode_ret = -1;
        return nullptr;
    }
    return clCreateContext_ptr(properties, num_devices, devices, (cl_context_callback)pfn_notify, user_data, errcode_ret);
}

cl_context clCreateContextFromType(const void* properties, cl_bitfield device_type, void (*pfn_notify)(const char*, const void*, size_t, void*), void* user_data, cl_int* errcode_ret) {
    if (!init_opencl() || !clCreateContextFromType_ptr) {
        if (errcode_ret) *errcode_ret = -1;
        return nullptr;
    }
    return clCreateContextFromType_ptr(properties, device_type, (cl_context_callback)pfn_notify, user_data, errcode_ret);
}

cl_int clRetainContext(cl_context context) {
    if (!init_opencl() || !clRetainContext_ptr) return -1;
    return clRetainContext_ptr(context);
}

cl_int clReleaseContext(cl_context context) {
    if (!init_opencl() || !clReleaseContext_ptr) return -1;
    return clReleaseContext_ptr(context);
}

// Command queue functions
cl_command_queue clCreateCommandQueue(cl_context context, cl_device_id device, cl_bitfield properties, cl_int* errcode_ret) {
    if (!init_opencl() || !clCreateCommandQueue_ptr) {
        if (errcode_ret) *errcode_ret = -1;
        return nullptr;
    }
    return clCreateCommandQueue_ptr(context, device, properties, errcode_ret);
}

cl_int clRetainCommandQueue(cl_command_queue command_queue) {
    if (!init_opencl() || !clRetainCommandQueue_ptr) return -1;
    return clRetainCommandQueue_ptr(command_queue);
}

cl_int clReleaseCommandQueue(cl_command_queue command_queue) {
    if (!init_opencl() || !clReleaseCommandQueue_ptr) return -1;
    return clReleaseCommandQueue_ptr(command_queue);
}

cl_int clFlush(cl_command_queue command_queue) {
    if (!init_opencl() || !clFlush_ptr) return -1;
    return clFlush_ptr(command_queue);
}

cl_int clFinish(cl_command_queue command_queue) {
    if (!init_opencl() || !clFinish_ptr) return -1;
    return clFinish_ptr(command_queue);
}

// Memory functions
cl_mem clCreateBuffer(cl_context context, cl_bitfield flags, cl_size_t size, void* host_ptr, cl_int* errcode_ret) {
    if (!init_opencl() || !clCreateBuffer_ptr) {
        if (errcode_ret) *errcode_ret = -1;
        return nullptr;
    }
    return clCreateBuffer_ptr(context, flags, size, host_ptr, errcode_ret);
}

cl_int clRetainMemObject(cl_mem memobj) {
    if (!init_opencl() || !clRetainMemObject_ptr) return -1;
    return clRetainMemObject_ptr(memobj);
}

cl_int clReleaseMemObject(cl_mem memobj) {
    if (!init_opencl() || !clReleaseMemObject_ptr) return -1;
    return clReleaseMemObject_ptr(memobj);
}

cl_int clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_int blocking_read, cl_size_t offset, cl_size_t size, void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    if (!init_opencl() || !clEnqueueReadBuffer_ptr) return -1;
    return clEnqueueReadBuffer_ptr(command_queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
}

cl_int clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_int blocking_write, cl_size_t offset, cl_size_t size, const void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    if (!init_opencl() || !clEnqueueWriteBuffer_ptr) return -1;
    return clEnqueueWriteBuffer_ptr(command_queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
}

cl_int clEnqueueCopyBuffer(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer, cl_size_t src_offset, cl_size_t dst_offset, cl_size_t size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    if (!init_opencl() || !clEnqueueCopyBuffer_ptr) return -1;
    return clEnqueueCopyBuffer_ptr(command_queue, src_buffer, dst_buffer, src_offset, dst_offset, size, num_events_in_wait_list, event_wait_list, event);
}

cl_int clEnqueueFillBuffer(cl_command_queue command_queue, cl_mem buffer, const void* pattern, cl_size_t pattern_size, cl_size_t offset, cl_size_t size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    if (!init_opencl() || !clEnqueueFillBuffer_ptr) return -1;
    return clEnqueueFillBuffer_ptr(command_queue, buffer, pattern, pattern_size, offset, size, num_events_in_wait_list, event_wait_list, event);
}

void* clEnqueueMapBuffer(cl_command_queue command_queue, cl_mem buffer, cl_int blocking_map, cl_bitfield map_flags, cl_size_t offset, cl_size_t size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event, cl_int* errcode_ret) {
    if (!init_opencl() || !clEnqueueMapBuffer_ptr) {
        if (errcode_ret) *errcode_ret = -1;
        return nullptr;
    }
    return clEnqueueMapBuffer_ptr(command_queue, buffer, blocking_map, map_flags, offset, size, num_events_in_wait_list, event_wait_list, event, errcode_ret);
}

cl_int clEnqueueUnmapMemObject(cl_command_queue command_queue, cl_mem memobj, void* mapped_ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    if (!init_opencl() || !clEnqueueUnmapMemObject_ptr) return -1;
    return clEnqueueUnmapMemObject_ptr(command_queue, memobj, mapped_ptr, num_events_in_wait_list, event_wait_list, event);
}

cl_mem clCreateImage(cl_context context, cl_bitfield flags, const void* image_format, const void* image_desc, void* host_ptr, cl_int* errcode_ret) {
    if (!init_opencl() || !clCreateImage_ptr) {
        if (errcode_ret) *errcode_ret = -1;
        return nullptr;
    }
    return clCreateImage_ptr(context, flags, image_format, image_desc, host_ptr, errcode_ret);
}

cl_mem clCreateSubBuffer(cl_mem buffer, cl_bitfield flags, cl_uint buffer_create_type, void* buffer_create_info, cl_int* errcode_ret) {
    if (!init_opencl() || !clCreateSubBuffer_ptr) {
        if (errcode_ret) *errcode_ret = -1;
        return nullptr;
    }
    return clCreateSubBuffer_ptr(buffer, flags, buffer_create_type, buffer_create_info, errcode_ret);
}

cl_int clEnqueueBarrierWithWaitList(cl_command_queue command_queue, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    if (!init_opencl() || !clEnqueueBarrierWithWaitList_ptr) return -1;
    return clEnqueueBarrierWithWaitList_ptr(command_queue, num_events_in_wait_list, event_wait_list, event);
}

cl_int clEnqueueMarkerWithWaitList(cl_command_queue command_queue, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    if (!init_opencl() || !clEnqueueMarkerWithWaitList_ptr) return -1;
    return clEnqueueMarkerWithWaitList_ptr(command_queue, num_events_in_wait_list, event_wait_list, event);
}

// Program and kernel functions
cl_program clCreateProgramWithSource(cl_context context, cl_uint count, const char** strings, const cl_size_t* lengths, cl_int* errcode_ret) {
    if (!init_opencl() || !clCreateProgramWithSource_ptr) {
        if (errcode_ret) *errcode_ret = -1;
        return nullptr;
    }
    return clCreateProgramWithSource_ptr(context, count, strings, lengths, errcode_ret);
}

cl_program clCreateProgramWithBinary(cl_context context, cl_uint num_devices, const cl_device_id* device_list, const cl_size_t* lengths, const unsigned char** binaries, cl_int* binary_status, cl_int* errcode_ret) {
    if (!init_opencl() || !clCreateProgramWithBinary_ptr) {
        if (errcode_ret) *errcode_ret = -1;
        return nullptr;
    }
    return clCreateProgramWithBinary_ptr(context, num_devices, device_list, lengths, binaries, binary_status, errcode_ret);
}

cl_int clRetainProgram(cl_program program) {
    if (!init_opencl() || !clRetainProgram_ptr) return -1;
    return clRetainProgram_ptr(program);
}

cl_int clReleaseProgram(cl_program program) {
    if (!init_opencl() || !clReleaseProgram_ptr) return -1;
    return clReleaseProgram_ptr(program);
}

cl_int clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options, void (*pfn_notify)(cl_program, void*), void* user_data) {
    if (!init_opencl() || !clBuildProgram_ptr) return -1;
    return clBuildProgram_ptr(program, num_devices, device_list, options, (cl_program_callback)pfn_notify, user_data);
}

cl_int clGetProgramBuildInfo(cl_program program, cl_device_id device, cl_uint param_name, cl_size_t param_value_size, void* param_value, cl_size_t* param_value_size_ret) {
    if (!init_opencl() || !clGetProgramBuildInfo_ptr) return -1;
    return clGetProgramBuildInfo_ptr(program, device, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_kernel clCreateKernel(cl_program program, const char* kernel_name, cl_int* errcode_ret) {
    if (!init_opencl() || !clCreateKernel_ptr) {
        if (errcode_ret) *errcode_ret = -1;
        return nullptr;
    }
    return clCreateKernel_ptr(program, kernel_name, errcode_ret);
}

cl_int clRetainKernel(cl_kernel kernel) {
    if (!init_opencl() || !clRetainKernel_ptr) return -1;
    return clRetainKernel_ptr(kernel);
}

cl_int clReleaseKernel(cl_kernel kernel) {
    if (!init_opencl() || !clReleaseKernel_ptr) return -1;
    return clReleaseKernel_ptr(kernel);
}

cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index, cl_size_t arg_size, const void* arg_value) {
    if (!init_opencl() || !clSetKernelArg_ptr) return -1;
    return clSetKernelArg_ptr(kernel, arg_index, arg_size, arg_value);
}

// Execution functions
cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const cl_size_t* global_work_offset, const cl_size_t* global_work_size, const cl_size_t* local_work_size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    if (!init_opencl() || !clEnqueueNDRangeKernel_ptr) return -1;
    return clEnqueueNDRangeKernel_ptr(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event);
}

cl_int clEnqueueTask(cl_command_queue command_queue, cl_kernel kernel, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    if (!init_opencl() || !clEnqueueTask_ptr) return -1;
    return clEnqueueTask_ptr(command_queue, kernel, num_events_in_wait_list, event_wait_list, event);
}

// Event functions
cl_int clWaitForEvents(cl_uint num_events, const cl_event* event_list) {
    if (!init_opencl() || !clWaitForEvents_ptr) return -1;
    return clWaitForEvents_ptr(num_events, event_list);
}

cl_int clRetainEvent(cl_event event) {
    if (!init_opencl() || !clRetainEvent_ptr) return -1;
    return clRetainEvent_ptr(event);
}

cl_int clReleaseEvent(cl_event event) {
    if (!init_opencl() || !clReleaseEvent_ptr) return -1;
    return clReleaseEvent_ptr(event);
}

cl_int clGetEventInfo(cl_event event, cl_uint param_name, cl_size_t param_value_size, void* param_value, cl_size_t* param_value_size_ret) {
    if (!init_opencl() || !clGetEventInfo_ptr) return -1;
    return clGetEventInfo_ptr(event, param_name, param_value_size, param_value, param_value_size_ret);
}

// Additional functions
cl_int clGetKernelWorkGroupInfo(cl_kernel kernel, cl_device_id device, cl_uint param_name, cl_size_t param_value_size, void* param_value, cl_size_t* param_value_size_ret) {
    if (!init_opencl() || !clGetKernelWorkGroupInfo_ptr) return -1;
    return clGetKernelWorkGroupInfo_ptr(kernel, device, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int clGetMemObjectInfo(cl_mem memobj, cl_uint param_name, cl_size_t param_value_size, void* param_value, cl_size_t* param_value_size_ret) {
    if (!init_opencl() || !clGetMemObjectInfo_ptr) return -1;
    return clGetMemObjectInfo_ptr(memobj, param_name, param_value_size, param_value, param_value_size_ret);
}

} // extern "C"


