// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
//
// GGUF v2/v3 reader with float32 dequantization.
//
// Scope: read the header, every metadata KV pair (all value types, including
// nested arrays), the tensor infos, the alignment and the data section base;
// then dequantize a tensor to row-major float32.
//
// The file is mapped read-only (POSIX mmap, CreateFileMapping on Windows), so
// tensor payloads are never copied into RAM: dequantization reads straight out
// of the mapping.
//
// Layout reference: ggml docs/gguf.md, plus the block structs in ggml.h /
// ggml-quants.c. Block layouts here must stay byte-identical to llama.cpp or
// the numbers will be silently wrong.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace yskn::gguf {

// ---------------------------------------------------------------------------
// ggml tensor types (mirrors enum ggml_type in ggml.h).
// ---------------------------------------------------------------------------
enum GgmlType : uint32_t {
  TYPE_F32 = 0,
  TYPE_F16 = 1,
  TYPE_Q4_0 = 2,
  TYPE_Q4_1 = 3,
  TYPE_Q5_0 = 6,
  TYPE_Q5_1 = 7,
  TYPE_Q8_0 = 8,
  TYPE_Q8_1 = 9,  // legacy, not dequantized here
  TYPE_Q2_K = 10,
  TYPE_Q3_K = 11,
  TYPE_Q4_K = 12,
  TYPE_Q5_K = 13,
  TYPE_Q6_K = 14,
  TYPE_Q8_K = 15,
  TYPE_IQ4_NL = 20,
  TYPE_BF16 = 30,
};

// ---------------------------------------------------------------------------
// Metadata values.
// ---------------------------------------------------------------------------
enum class ValueType : uint32_t {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
};

// Human readable name of a metadata value type ("uint32", "array", ...).
const char* value_type_name(ValueType t);

// One metadata value. A tagged union: only the member matching 'type' is
// meaningful. ARRAY uses 'elem_type' + 'items' (items may themselves be
// arrays, which is how nested arrays are represented).
struct Value {
  ValueType type = ValueType::UINT32;
  ValueType elem_type = ValueType::UINT32;  // element type when type == ARRAY
  uint64_t u = 0;                           // unsigned ints, BOOL (0/1)
  int64_t i = 0;                            // signed ints
  double f = 0.0;                           // FLOAT32 / FLOAT64
  std::string str;                          // STRING
  std::vector<Value> items;                 // ARRAY

  bool is_array() const { return type == ValueType::ARRAY; }
  bool is_number() const;

  // Accessors. Each throws std::runtime_error when the stored type does not
  // match (no silent reinterpretation).
  uint64_t as_uint() const;
  int64_t as_int() const;
  double as_float() const;
  bool as_bool() const;
  const std::string& as_string() const;
  const std::vector<Value>& as_array() const;

  static Value make_uint(ValueType t, uint64_t v);
  static Value make_int(ValueType t, int64_t v);
  static Value make_float(ValueType t, double v);
  static Value make_bool(bool v);
  static Value make_string(std::string s);
  static Value make_array(ValueType elem_type, std::vector<Value> items);
};

// Insertion-ordered key/value store that preserves the file order (GGUF
// metadata is an ordered list; llama.cpp relies on that for nothing, but we
// keep it so callers can dump a file faithfully).
class Metadata {
 public:
  size_t size() const { return items_.size(); }
  bool empty() const { return items_.empty(); }
  bool contains(const std::string& key) const { return find(key) != nullptr; }

  const Value* find(const std::string& key) const;
  const Value& at(const std::string& key) const;

  // Typed lookups with defaults; throw when the key exists with a wrong type.
  uint64_t get_uint(const std::string& key, uint64_t fallback) const;
  int64_t get_int(const std::string& key, int64_t fallback) const;
  double get_float(const std::string& key, double fallback) const;
  bool get_bool(const std::string& key, bool fallback) const;
  std::string get_string(const std::string& key, const std::string& fallback) const;

  const std::vector<std::pair<std::string, Value>>& items() const { return items_; }

  // Parser use only.
  void insert(std::string key, Value v);

 private:
  std::vector<std::pair<std::string, Value>> items_;
};

// ---------------------------------------------------------------------------
// Tensor info.
// ---------------------------------------------------------------------------
struct TensorInfo {
  std::string name;
  std::vector<uint64_t> shape;  // GGUF order: fastest-varying dimension first
  uint32_t type = 0;            // GgmlType
  uint64_t offset = 0;          // byte offset relative to the data section base
  uint64_t n_elements = 0;      // product of shape (1 for a 0-dim scalar)

  // Byte size of the payload; throws when the element count is not a whole
  // number of blocks for this type.
  uint64_t nbytes() const;
};

// ---------------------------------------------------------------------------
// File.
// ---------------------------------------------------------------------------
class File {
 public:
  // Defined in gguf.cpp: Impl is incomplete here, so the destructor and the
  // default constructor cannot be inline.
  File();
  ~File();
  File(File&&) noexcept;
  File& operator=(File&&) noexcept;
  File(const File&) = delete;
  File& operator=(const File&) = delete;

  // Map and parse. Throws std::runtime_error with a message that names the
  // path and the byte offset of the problem.
  void open(const std::string& path);

  uint32_t version() const;
  uint64_t alignment() const;   // general.alignment, default 32
  uint64_t data_base() const;   // absolute file offset of tensor data
  uint64_t file_size() const;
  const std::string& path() const;

  const Metadata& meta() const;
  const std::vector<TensorInfo>& tensors() const;
  const TensorInfo* find(const std::string& name) const;

  // Raw payload pointer / size for one tensor (bounds checked).
  const uint8_t* tensor_data(const TensorInfo& t) const;
  uint64_t tensor_nbytes(const TensorInfo& t) const;

  // Dequantize to row-major float32. 'out' must hold t.n_elements floats.
  void dequant(const TensorInfo& t, float* out) const;
  std::vector<float> dequant(const TensorInfo& t) const;
  void dequant(const std::string& name, float* out) const;
  std::vector<float> dequant(const std::string& name) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// ---------------------------------------------------------------------------
// Type table helpers.
// ---------------------------------------------------------------------------
std::string type_name(uint32_t t);       // "Q4_K", "F32", "unknown(1234)", ...
uint64_t type_block_size(uint32_t t);    // bytes per block
uint64_t type_block_elems(uint32_t t);   // elements per block
bool type_supported(uint32_t t);         // dequant() accepts it
uint64_t type_nbytes(uint32_t t, uint64_t n_elements);  // throws if not whole blocks

}  // namespace yskn::gguf
