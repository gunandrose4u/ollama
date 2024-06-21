
# set(TARGET ollama_ort_genai_server)
# option(LLAMA_SERVER_VERBOSE "Build verbose logging option for Server" ON)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR})
# add_executable(${TARGET} server.cpp utils.hpp json.hpp httplib.h)
# install(TARGETS ${TARGET} RUNTIME)
# target_compile_definitions(${TARGET} PRIVATE
#     SERVER_VERBOSE=$<BOOL:${LLAMA_SERVER_VERBOSE}>
# )
# target_link_libraries(${TARGET} PRIVATE common llava ${CMAKE_THREAD_LIBS_INIT})
# if (WIN32)
#     TARGET_LINK_LIBRARIES(${TARGET} PRIVATE ws2_32)
# endif()
# target_compile_features(${TARGET} PRIVATE cxx_std_11)


# #############################################
cmake_minimum_required(VERSION 3.18.1)

project(ortgenaiapp)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a")

option(USE_CUDA "Build with CUDA support" OFF)
option(USE_CXX "Invoke the C++ example" ON)

if(USE_CXX)
  add_compile_definitions(USE_CXX)
endif()

if(USE_CUDA)
  include(CheckLanguage)
  enable_language(CUDA)
endif()

set(ORT_GENAI_LIB_DIR ${CMAKE_SOURCE_DIR}/lib)

if(WIN32)
  set(ONNXRUNTIME_GENAI_LIB "onnxruntime-genai.dll")
  set(ONNXRUNTIME_GENAI_DEPENDENCY "*.dll")
elseif(APPLE)
  set(ONNXRUNTIME_GENAI_LIB "libonnxruntime-genai.dylib")
  set(ONNXRUNTIME_GENAI_DEPENDENCY "*.dylib")
else()
  set(ONNXRUNTIME_GENAI_LIB "libonnxruntime-genai.so")
  set(ONNXRUNTIME_GENAI_DEPENDENCY "*.so")
endif()

set(TARGET ollama_ort_genai_server)
add_executable(${TARGET} server.cpp utils.hpp json.hpp httplib.h)


target_link_directories(${TARGET} PRIVATE ${ORT_GENAI_LIB_DIR})
target_link_libraries(${TARGET} PRIVATE ${ONNXRUNTIME_GENAI_LIB})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(USE_CUDA)
  set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CUDA)
endif()

target_link_libraries(
    ${TARGET}
        PUBLIC
        onnxruntime-genai)

if(USE_CUDA)
  target_link_libraries(
    ${TARGET}
        PUBLIC
        cublasLt cublas cudnn curand cufft cudart)
endif()

file(GLOB ort_genai_libs "${CMAKE_SOURCE_DIR}/lib/${ONNXRUNTIME_GENAI_DEPENDENCY}")

foreach(DLL_FILE ${ort_genai_libs})
  add_custom_command(
    TARGET ${TARGET} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DLL_FILE} $<TARGET_FILE_DIR:phi3>
  )
endforeach()