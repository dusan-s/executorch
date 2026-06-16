set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Toolchain
set(TC /home/ivam/praksa/k230_sdk/toolchain/riscv64-linux-musleabi_for_x86_64-pc-linux-gnu/bin/riscv64-unknown-linux-musl-)

set(CMAKE_C_COMPILER   ${TC}gcc)
set(CMAKE_CXX_COMPILER ${TC}g++)
set(CMAKE_AR           ${TC}ar     CACHE FILEPATH "")
set(CMAKE_RANLIB       ${TC}ranlib CACHE FILEPATH "")
set(CMAKE_STRIP        ${TC}strip  CACHE FILEPATH "")

# Compile flags
#set(CMAKE_C_FLAGS   "-mcmodel=medany -march=rv64imafdcv -mabi=lp64d" CACHE STRING "")
#set(CMAKE_CXX_FLAGS "-mcmodel=medany -march=rv64imafdcv -mabi=lp64d" CACHE STRING "")
#set(CMAKE_C_FLAGS   "-mcmodel=medany -march=rv64imafdcv -mabi=lp64d -DET_HAVE_PREAD=0" CACHE STRING "")
#set(CMAKE_CXX_FLAGS "-mcmodel=medany -march=rv64imafdcv -mabi=lp64d -DET_HAVE_PREAD=0" CACHE STRING "")
set(CMAKE_C_FLAGS   "-mcmodel=medany -march=rv64imafdcv -mabi=lp64d -DET_HAVE_PREAD=0 -O3" CACHE STRING "")
set(CMAKE_CXX_FLAGS "-mcmodel=medany -march=rv64imafdcv -mabi=lp64d -DET_HAVE_PREAD=0 -O3" CACHE STRING "")

# Linker flags
set(CMAKE_EXE_LINKER_FLAGS
    "-T /home/ivam/praksa/k230_sdk/src/big/mpp/userapps/sample/linker_scripts/riscv64/link.lds \
     -n --static \
     -L/home/ivam/praksa/k230_sdk/src/big/rt-smart/userapps/sdk/rt-thread/lib \
     -Wl,--whole-archive -lrtthread -Wl,--no-whole-archive \
     -L/home/ivam/praksa/k230_sdk/src/big/rt-smart/userapps/sdk/lib/risc-v/rv64 \
     -L/home/ivam/praksa/k230_sdk/src/big/rt-smart/userapps/sdk/rt-thread/lib/risc-v/rv64 \
     -Wl,--start-group -lrtthread -Wl,--end-group"
    CACHE STRING "")

# Don't search host paths
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# ExecuTorch options from zephyr.cmake
set(EXECUTORCH_BUILD_COREML             OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_ENABLE_EVENT_TRACER      ON CACHE BOOL "" FORCE)
#set(EXECUTORCH_BUILD_KERNELS_LLM        OFF CACHE BOOL "" FORCE)
#set(EXECUTORCH_BUILD_KERNELS_LLM_AOT    OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON CACHE BOOL "" FORCE)
#set(EXECUTORCH_BUILD_EXTENSION_LLM      OFF CACHE BOOL "" FORCE)
#set(EXECUTORCH_BUILD_EXTENSION_MODULE   OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_TRAINING OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_APPLE    OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_MPS                OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_NEURON             OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_OPENVINO           OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_PYBIND             OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_QNN                OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_KERNELS_OPTIMIZED  OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_KERNELS_QUANTIZED  OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_DEVTOOLS           ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_TESTS              OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_XNNPACK            OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_VULKAN             OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_PORTABLE_OPS       ON  CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_CADENCE            OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_PTHREADPOOL        OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_CPUINFO            OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_USE_CPP_CODE_COVERAGE    OFF CACHE BOOL "" FORCE)


set(EXECUTORCH_BUILD_KERNELS_OPTIMIZED        OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_KERNELS_LLM        OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_KERNELS_LLM_AOT    OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_LLM      OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP   ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_MODULE   ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON CACHE BOOL "" FORCE)
