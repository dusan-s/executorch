set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

#set(CMAKE_C_COMPILER riscv64-linux-gnu-gcc-15)
#set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++-15)
#set(CMAKE_AR riscv64-linux-gnu-ar)
#set(CMAKE_RANLIB riscv64-linux-gnu-ranlib)

set(TC /home/ivam/praksa/toolchainBananaPi/riscv64-glibc-ubuntu-24.04-gcc/riscv/bin/riscv64-unknown-linux-gnu-)

set(CMAKE_C_COMPILER   ${TC}gcc)
set(CMAKE_CXX_COMPILER ${TC}g++)
set(CMAKE_AR           ${TC}ar     CACHE FILEPATH "")
set(CMAKE_RANLIB       ${TC}ranlib CACHE FILEPATH "")
set(CMAKE_STRIP        ${TC}strip  CACHE FILEPATH "")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(MY_RISCV_FLAGS "-mcmodel=medlow -march=rv64gcv_zicbom_zicboz_zicntr_zicond_zicsr_zifencei_zihintpause_zihpm_zfh_zfhmin_zca_zcd_zba_zbb_zbc_zbs_zkt_zve32f_zve32x_zve64d_zve64f_zve64x_zvfh_zvfhmin -mvector-strict-align -mstrict-align -mabi=lp64d -O3 -DET_EVENT_TRACER_ENABLED")

#set(MY_RISCV_FLAGS "-mcmodel=medlow -march=rv64gcv -mvector-strict-align -mstrict-align -mabi=lp64d -O2")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MY_RISCV_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MY_RISCV_FLAGS}")

