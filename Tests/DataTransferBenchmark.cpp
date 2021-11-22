#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <benchmark/benchmark.h>
#include <cstring>
#include <iostream>
#include <vector>

#define NULL_32 std::numeric_limits<int32_t>::max()
#define NULL_64 std::numeric_limits<int64_t>::max()

struct ArrowSchema inputSchema {};
struct ArrowArray inputArray {};

void arrow_to_omnisci_memcpy(struct ArrowArray *inputArray, struct ArrowSchema *inputSchema, int8_t** col_buffer) {
    const int length = 10;
    // two column buffer (TODO: get type from input schema ??)
    int32_t* col_0_date = (int32_t*)std::malloc(sizeof(int32_t) * length);
    int64_t* col_1_long = (int64_t*)std::malloc(sizeof(int64_t) * length);
    
    const int32_t* col_0 = static_cast<const int32_t*>(inputArray->children[0]->buffers[1]);
    memcpy(col_0_date, col_0, length);
    const uint8_t* null_buffer_0 = static_cast<const uint8_t*>(inputArray->children[0]->buffers[0]);
    if (null_buffer_0 != nullptr) {
      for (int i = 0; i < length; i++) {
        // null represent as max value
        if (null_buffer_0[i] == 0) {
            col_0_date[i] = NULL_32;
        }
      }
    }
    
    const int64_t* col_1 = static_cast<const int64_t*>(inputArray->children[1]->buffers[1]);
    memcpy(col_1_long, col_1, length);
    const uint8_t* null_buffer_1 = static_cast<const uint8_t*>(inputArray->children[1]->buffers[0]);
    if (null_buffer_1 != nullptr) {
      for (int i = 0; i < length; i++) {
        if (null_buffer_1[i] == 0) {
            col_1_long[i] = NULL_64;
        }
      }
    }
    
    col_buffer[0] = reinterpret_cast<int8_t*>(col_0_date);
    col_buffer[1] = reinterpret_cast<int8_t*>(col_1_long);
}

void arrow_to_omnisci_zerocpy(struct ArrowArray *inputArray, struct ArrowSchema *inputSchema, int8_t** col_buffer) {
  const int length = 10;
  const int32_t* col_0 = static_cast<const int32_t*>(inputArray->children[0]->buffers[1]);
  const uint8_t* null_buffer_0 = static_cast<const uint8_t*>(inputArray->children[0]->buffers[0]);
  // if (null_buffer_0 != nullptr) {
  //   for (int i = 0; i < length; i++) {
  //     // null represent as max value
  //     if (null_buffer_0[i] == 0) {
  //         col_0[i] = NULL_32;
  //     }
  //   }
  // }
  
  
  const int64_t* col_1 = static_cast<const int64_t*>(inputArray->children[1]->buffers[1]);
  const uint8_t* null_buffer_1 = static_cast<const uint8_t*>(inputArray->children[1]->buffers[0]);
  // if (null_buffer_1 != nullptr) {
  //   for (int i = 0; i < length; i++) {
  //     if (null_buffer_1[i] == 0) {
  //         col_1[i] = NULL_64;
  //     }
  //   }
  // }
  
  // col_buffer[0] = reinterpret_cast<int8_t*>(col_0);
  // col_buffer[1] = reinterpret_cast<int8_t*>(col_1);
}

class UpdateFixture : public benchmark::Fixture {
 public:
 
  void SetUp(const ::benchmark::State& state) override {
    const int length = 10;
    // create arrow array 
    //const std::vector<std::optional<T>>& inputData 
    std::vector<int32_t> vec_0{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int64_t> vec_1{1, 22, 333, 4444, 55555, 666666, 7777777, 88888888, 999999999, -1};
    auto col_0 = std::make_shared<arrow::Int32Array>(length, arrow::Buffer::Wrap(vec_0));
    auto col_1 = std::make_shared<arrow::Int64Array>(length, arrow::Buffer::Wrap(vec_1));
    
    auto field_0 = arrow::field("date_col_0", arrow::int32());
    auto field_1 = arrow::field("long_col_1", arrow::int64());
    auto schema = arrow::schema({field_0, field_1});
    
    auto rb = arrow::RecordBatch::Make(schema, length, {col_0, col_1});

    arrow::ExportRecordBatch(*rb, &inputArray, &inputSchema);

    // another way to create arrow array ??
    // const void* data_buffers[2];
    // const int32_t* values = (int32_t*)std::malloc(sizeof(int32_t) * 10);
    // for (int i = 0; i < 10; i++) {
    //     values[i] = i;
    // }
    // data_buffers[0] = nullptr;
    // data_buffers[1] = values;

    // ArrowArray inputArray = initArray(10, 2, data_buffers);
    // ArrowSchema inputSchema = initSchema("i");

    // // create arrow array 
    // const void* data_buffers[2];
    // const int64_t* values = (int64_t*)std::malloc(sizeof(int64_t) * 10);
    // for (int i = 0; i < 10; i++) {
    //     values[i] = i;
    // }
    // data_buffers[0] = nullptr;
    // data_buffers[1] = values;

    // ArrowArray inputArray = initArray(10, 2, data_buffers);
    // ArrowSchema inputSchema = initSchema("l");

    int8_t** col_buffer = (int8_t**)std::malloc(sizeof(int8_t*) * 2);
    //int8_t** check_buffer = (int8_t**)std::malloc(sizeof(int8_t*) * 2);
    int32_t* col_date = (int32_t*)std::malloc(sizeof(int32_t) * 10);
    int64_t* col_long = (int64_t*)std::malloc(sizeof(int64_t) * 10);
    col_buffer[0] = reinterpret_cast<int8_t*>(col_date);
    col_buffer[1] = reinterpret_cast<int8_t*>(col_long);

    // warmed up & check correctness of conversion
    arrow_to_omnisci_memcpy(&inputArray, &inputSchema, col_buffer);
    // CHECK_EQ(std::equal(
    //   std::begin(col_buffer), std:end(col_buffer), 
    //   std:begin(check_buffer), std:end(check_buffer));

    //release col buffer
    std::free(col_buffer[0]);
    std::free(col_buffer[1]);
    std::free(col_buffer);

    // std::free(check_buffer[0]);
    // std::free(check_buffer[1]);
    // std::free(check_buffer);
  }

  void TearDown(const ::benchmark::State& state) override {
    // release arrow array ???
    //arrowArray.release(&arrowArray);
  }
};


// ArrowSchema initSchema(char* format){
//   return ArrowSchema{
//         .format = format,
//         .name = nullptr,
//         .metadata = nullptr,
//         .flags = 0,
//         .n_children = 0,
//         .children = nullptr,
//         .dictionary = nullptr,
//         .release = nullptr,
//         .private_data = nullptr,
//     };
// }

// ArrowArray initArray(int length, int n_buffers, const void** buffers){
//   return ArrowArray{
//         .length = length,
//         .null_count = 0,
//         .offset = 0,
//         .n_buffers = n_buffers,
//         .n_children = 0,
//         .buffers = buffers,
//         .children = nullptr,
//         .dictionary = nullptr,
//         .release = nullptr,
//         .private_data = nullptr,
//     };
// }

BENCHMARK_DEFINE_F(UpdateFixture, ArrowCToOmnisciMemcpyTest)(benchmark::State& state) {
  int8_t** col_buffer = (int8_t**)std::malloc(sizeof(int8_t*) * 2);
  for (auto _ : state) {
    arrow_to_omnisci_memcpy(&inputArray, &inputSchema, col_buffer);
  }
  std::free(col_buffer[0]);
  std::free(col_buffer[1]);
  std::free(col_buffer);
}

BENCHMARK_REGISTER_F(UpdateFixture, ArrowCToOmnisciMemcpyTest)
    ->Range(100, 1000000)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
    

BENCHMARK_DEFINE_F(UpdateFixture, ArrowCToOmnisciZerocpyTest)(benchmark::State& state) {
  int8_t** col_buffer = (int8_t**)std::malloc(sizeof(int8_t*) * 2);
  for (auto _ : state) {
    //arrow_to_omnisci_zerocpy(inputArray, inputSchema, col_buffer);
  }
  std::free(col_buffer[0]);
  std::free(col_buffer[1]);
  std::free(col_buffer);
}

BENCHMARK_REGISTER_F(UpdateFixture, ArrowCToOmnisciZerocpyTest)
    ->Range(100, 1000000)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

BENCHMARK_MAIN();