set(SRC_FILE CiderExecutionKernel.cpp)
add_library(cider SHARED ${SRC_FILE})
target_link_libraries(cider mapd_thrift thrift_handler Catalog QueryState Parser QueryEngine DataMgr Geospatial ${Arrow_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} ${TBB_LIBS})

# To pass in Cmake parameter to omnisci side
add_definitions(-DRAPIDJSON_HAS_STDSTRING)

install(TARGETS cider DESTINATION "lib" COMPONENT "CIDER")
install(FILES CiderExecutionKernel.h DESTINATION "Cider" COMPONENT "include")
