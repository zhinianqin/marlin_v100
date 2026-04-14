#include "marlin_dtypes.cuh"

namespace MARLIN_NAMESPACE_NAME {

// m16n8k16 tensor core mma instruction with fp16 inputs and fp32
// output/accumulation.
template <vllm::ScalarTypeId type_id, bool use_fp16_accum, int k_size = 16>
__device__ inline void mma(
    const typename MarlinScalarType<type_id>::FragA& a_frag,
    const typename MarlinScalarType<type_id>::FragB& frag_b,
    typename MarlinScalarType<type_id>::FragC& frag_c, int idx = 0) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  if constexpr (!std::is_same<scalar_t, half>::value || k_size != 16) {
    static_assert(!use_fp16_accum);
  }

  if constexpr (k_size == 16) {
    static_assert(std::is_same<scalar_t, half>::value,
                  "SM70 inline PTX mma currently supports fp16 inputs only.");
    static_assert(!use_fp16_accum,
                  "SM70 inline PTX mma currently supports fp32 accumulation only.");

    if constexpr (std::is_same<scalar_t, half>::value && !use_fp16_accum) {
      float* c = reinterpret_cast<float*>(&frag_c);
      /*asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(a[0]), "r"(a[1]), "r"(b[0]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
            "f"(c[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(a[2]), "r"(a[3]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
            "f"(c[3]));*/

      // 上面注释掉的代码是sm75版本正确的逻辑
      // 下面代码是错误的的，仅作占位约束使用

      // 1. Warp 内线程标识获取
      uint32_t lane_id;
      asm volatile("mov.u32 %0, %laneid;" : "=r"(lane_id));
      uint32_t src_lane = 4 * (lane_id & 7) + (lane_id >> 4);
      bool is_left_half = (lane_id & 8) == 0; 

      // 2. 准备 Dummy 寄存器，接住 Volta 强制输出的右半边废弃数据
      float dummy_c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

      // ==========================================================
      // 替换第一条 Turing m16n8k8 (K = 0~7)
      // 拆分为两条 Volta m8n8k4 (K = 0~3, K = 4~7)
      // ==========================================================

      // --- Volta Instr 1: K = 0~3 ---
      // A 的洗牌值（无条件执行同步）
      uint32_t a0_val = is_left_half ? a[0] : a[1];
      uint32_t a0_v0 = __shfl_sync(0xffffffff, a0_val, src_lane);
      uint32_t a0_v1 = __shfl_sync(0xffffffff, a0_val, src_lane + 2);

      // B 的洗牌值（无条件执行同步）
      uint32_t b0_shfl_0 = __shfl_sync(0xffffffff, b[0], src_lane);
      uint32_t b0_shfl_1 = __shfl_sync(0xffffffff, b[0], src_lane + 2);

      // 同步完成后，再对结果进行条件置零，防止死锁
      uint32_t b0_v0 = is_left_half ? b0_shfl_0 : 0;
      uint32_t b0_v1 = is_left_half ? b0_shfl_1 : 0;

      asm volatile(
          "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3]),
            "=f"(dummy_c[0]), "=f"(dummy_c[1]), "=f"(dummy_c[2]), "=f"(dummy_c[3])
          : "r"(a0_v0), "r"(a0_v1), "r"(b0_v0), "r"(b0_v1),
            "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
            "f"(dummy_c[0]), "f"(dummy_c[1]), "f"(dummy_c[2]), "f"(dummy_c[3])
      );

      // --- Volta Instr 2: K = 4~7 ---
      uint32_t a1_val = is_left_half ? a[1] : a[0]; 
      uint32_t a1_v0 = __shfl_sync(0xffffffff, a1_val, src_lane);
      uint32_t a1_v1 = __shfl_sync(0xffffffff, a1_val, src_lane + 2);

      // 同样，无条件执行 B 的洗牌（复用 b[0] 作为数据源）
      uint32_t b1_shfl_0 = __shfl_sync(0xffffffff, b[0], src_lane); 
      uint32_t b1_shfl_1 = __shfl_sync(0xffffffff, b[0], src_lane + 2);

      uint32_t b1_v0 = is_left_half ? b1_shfl_0 : 0; 
      uint32_t b1_v1 = is_left_half ? b1_shfl_1 : 0;

      asm volatile(
          "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3]),
            "=f"(dummy_c[0]), "=f"(dummy_c[1]), "=f"(dummy_c[2]), "=f"(dummy_c[3])
          : "r"(a1_v0), "r"(a1_v1), "r"(b1_v0), "r"(b1_v1),
            "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
            "f"(dummy_c[0]), "f"(dummy_c[1]), "f"(dummy_c[2]), "f"(dummy_c[3])
      );


      // ==========================================================
      // 替换第二条 Turing m16n8k8 (K = 8~15)
      // 拆解为两条 Volta m8n8k4 (K = 8~11, K = 12~15)
      // ==========================================================

      // --- Volta Instr 3: K = 8~11 ---
      uint32_t a2_val = is_left_half ? a[2] : a[3];
      uint32_t a2_v0 = __shfl_sync(0xffffffff, a2_val, src_lane);
      uint32_t a2_v1 = __shfl_sync(0xffffffff, a2_val, src_lane + 2);

      uint32_t b2_shfl_0 = __shfl_sync(0xffffffff, b[1], src_lane);
      uint32_t b2_shfl_1 = __shfl_sync(0xffffffff, b[1], src_lane + 2);

      uint32_t b2_v0 = is_left_half ? b2_shfl_0 : 0;
      uint32_t b2_v1 = is_left_half ? b2_shfl_1 : 0;

      asm volatile(
          "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3]),
            "=f"(dummy_c[0]), "=f"(dummy_c[1]), "=f"(dummy_c[2]), "=f"(dummy_c[3])
          : "r"(a2_v0), "r"(a2_v1), "r"(b2_v0), "r"(b2_v1),
            "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
            "f"(dummy_c[0]), "f"(dummy_c[1]), "f"(dummy_c[2]), "f"(dummy_c[3])
      );

      // --- Volta Instr 4: K = 12~15 ---
      uint32_t a3_val = is_left_half ? a[3] : a[2]; 
      uint32_t a3_v0 = __shfl_sync(0xffffffff, a3_val, src_lane);
      uint32_t a3_v1 = __shfl_sync(0xffffffff, a3_val, src_lane + 2);

      uint32_t b3_shfl_0 = __shfl_sync(0xffffffff, b[1], src_lane); 
      uint32_t b3_shfl_1 = __shfl_sync(0xffffffff, b[1], src_lane + 2);

      uint32_t b3_v0 = is_left_half ? b3_shfl_0 : 0; 
      uint32_t b3_v1 = is_left_half ? b3_shfl_1 : 0; 

      asm volatile(
          "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3]),
            "=f"(dummy_c[0]), "=f"(dummy_c[1]), "=f"(dummy_c[2]), "=f"(dummy_c[3])
          : "r"(a3_v0), "r"(a3_v1), "r"(b3_v0), "r"(b3_v1),
            "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
            "f"(dummy_c[0]), "f"(dummy_c[1]), "f"(dummy_c[2]), "f"(dummy_c[3])
      );

    } /*else if constexpr (std::is_same<scalar_t, half>::value &&
                         use_fp16_accum) {
      uint32_t* c = reinterpret_cast<uint32_t*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
          "{%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(c[0]), "r"(c[1]));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
          "{%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(a[2]), "r"(a[3]), "r"(b[1]), "r"(c[0]), "r"(c[1]));
    } else if constexpr (std::is_same<scalar_t, int8_t>::value) {
      int32_t* c = reinterpret_cast<int32_t*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
          : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
          : "r"(a[idx * 2]), "r"(a[idx * 2 + 1]), "r"(b[idx]), "r"(c[0]),
            "r"(c[1]), "r"(c[2]), "r"(c[3]));
    }*/
  } /*else if (k_size == 32) {
    if constexpr (std::is_same<scalar_t, int8_t>::value) {
      int32_t* c = reinterpret_cast<int32_t*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(a[0]), "r"(b[0]), "r"(c[0]), "r"(c[1]));
      asm volatile(
          "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=r"(c[2]), "=r"(c[3])
          : "r"(a[1]), "r"(b[0]), "r"(c[2]), "r"(c[3]));
      asm volatile(
          "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(a[2]), "r"(b[1]), "r"(c[0]), "r"(c[1]));
      asm volatile(
          "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=r"(c[2]), "=r"(c[3])
          : "r"(a[3]), "r"(b[1]), "r"(c[2]), "r"(c[3]));
    }
  }*/
}

template <vllm::ScalarTypeId type_id, bool use_fp16_accum, int k_size = 16>
__device__ inline void mma_trans(
    const typename MarlinScalarType<type_id>::FragA& a_frag,
    const typename MarlinScalarType<type_id>::FragB& frag_b,
    const typename MarlinScalarType<type_id>::FragB& frag_b2,
    typename MarlinScalarType<type_id>::FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  const uint32_t* b2 = reinterpret_cast<const uint32_t*>(&frag_b2);
  float* c = reinterpret_cast<float*>(&frag_c);
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  if constexpr (!std::is_same<scalar_t, half>::value || k_size != 16) {
    static_assert(!use_fp16_accum);
  }

  if constexpr (k_size == 16) {
    static_assert(std::is_same<scalar_t, half>::value,
                  "SM70 inline PTX mma currently supports fp16 inputs only.");
    static_assert(!use_fp16_accum,
                  "SM70 inline PTX mma currently supports fp32 accumulation only.");
    if constexpr (std::is_same<scalar_t, half>::value && !use_fp16_accum) {
      float* c = reinterpret_cast<float*>(&frag_c);
      /*asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(b[0]), "r"(b2[0]), "r"(a[0]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
            "f"(c[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(b[1]), "r"(b2[1]), "r"(a[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
            "f"(c[3]));*/

      // 上面注释掉的代码是sm75版本正确的逻辑
      // 下面代码是错误的的，仅作占位约束使用

      uint32_t lane = threadIdx.x % 32;

      // Volta m8n8k4 强制要求 C 为 8 个寄存器。
      // 左半部分 (c0, c1, c4, c5) 用于存储我们需要的 N=0..7 的结果
      // 右半部分 (c2, c3, c6, c7) 为无效废弃数据，用 dummy 变量接住
      float dummy0 = 0.0f, dummy1 = 0.0f, dummy2 = 0.0f, dummy3 = 0.0f;

      // 预计算 A 矩阵和 B 矩阵的 shuffle 源线程 (source lane)
      // 这样做可以避免在分支内部调用 __shfl_sync，保证安全性
      uint32_t src_A_step1, src_A_step2;
      uint32_t src_B_step1_reg0, src_B_step1_reg1;
      uint32_t src_B_step2_reg0, src_B_step2_reg1;

      // 映射 A 矩阵的线程分布 (对应原来代码里的 b 和 b2)
      if (lane < 8) {
          src_A_step1 = lane;       src_A_step2 = lane + 8;
      } else if (lane < 16) {
          src_A_step1 = lane + 8;   src_A_step2 = lane + 16;
      } else if (lane < 24) {
          src_A_step1 = lane - 16;  src_A_step2 = lane - 8;
      } else {
          src_A_step1 = lane - 8;   src_A_step2 = lane;
      }

      // 映射 B 矩阵的线程分布 (对应原来代码里的 a，并且映射出 2 个冗余寄存器)
      if (lane < 8) {
          src_B_step1_reg0 = lane;       src_B_step1_reg1 = lane + 8;
          src_B_step2_reg0 = lane + 16;  src_B_step2_reg1 = lane + 24;
      } else if (lane < 16) {
          src_B_step1_reg0 = lane - 8;   src_B_step1_reg1 = lane;
          src_B_step2_reg0 = lane + 8;   src_B_step2_reg1 = lane + 16;
      } else {
          // 线程 16~31 计算的是 C 矩阵的右半边，结果会被丢弃，给 0 即可
          src_B_step1_reg0 = 0;          src_B_step1_reg1 = 0;
          src_B_step2_reg0 = 0;          src_B_step2_reg1 = 0;
      }

      // ====================================================================
      // 第一条原指令: b[0], b2[0], a[0] -> 拆分为 Step 1 (K=0~3) 和 Step 2 (K=4~7)
      // ====================================================================

      // --- Step 1 (K=0..3) ---
      uint32_t a_s1_r0 = __shfl_sync(0xffffffff, b[0],  src_A_step1);
      uint32_t a_s1_r1 = __shfl_sync(0xffffffff, b2[0], src_A_step1);
      uint32_t b_s1_r0 = __shfl_sync(0xffffffff, a[0],  src_B_step1_reg0);
      uint32_t b_s1_r1 = __shfl_sync(0xffffffff, a[0],  src_B_step1_reg1);

      asm volatile(
          "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(dummy0), "=f"(dummy1),
            "=f"(c[2]), "=f"(c[3]), "=f"(dummy2), "=f"(dummy3)
          : "r"(a_s1_r0), "r"(a_s1_r1), "r"(b_s1_r0), "r"(b_s1_r1),
            "f"(c[0]), "f"(c[1]), "f"(dummy0), "f"(dummy1),
            "f"(c[2]), "f"(c[3]), "f"(dummy2), "f"(dummy3));

      // --- Step 2 (K=4..7) ---
      uint32_t a_s2_r0 = __shfl_sync(0xffffffff, b[0],  src_A_step2);
      uint32_t a_s2_r1 = __shfl_sync(0xffffffff, b2[0], src_A_step2);
      uint32_t b_s2_r0 = __shfl_sync(0xffffffff, a[0],  src_B_step2_reg0);
      uint32_t b_s2_r1 = __shfl_sync(0xffffffff, a[0],  src_B_step2_reg1);

      asm volatile(
          "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(dummy0), "=f"(dummy1),
            "=f"(c[2]), "=f"(c[3]), "=f"(dummy2), "=f"(dummy3)
          : "r"(a_s2_r0), "r"(a_s2_r1), "r"(b_s2_r0), "r"(b_s2_r1),
            "f"(c[0]), "f"(c[1]), "f"(dummy0), "f"(dummy1),
            "f"(c[2]), "f"(c[3]), "f"(dummy2), "f"(dummy3));


      // ====================================================================
      // 第二条原指令: b[1], b2[1], a[1] -> 拆分为 Step 3 (K=8~11) 和 Step 4 (K=12~15)
      // ====================================================================

      // --- Step 3 (K=8..11) ---
      uint32_t a_s3_r0 = __shfl_sync(0xffffffff, b[1],  src_A_step1);
      uint32_t a_s3_r1 = __shfl_sync(0xffffffff, b2[1], src_A_step1);
      uint32_t b_s3_r0 = __shfl_sync(0xffffffff, a[1],  src_B_step1_reg0);
      uint32_t b_s3_r1 = __shfl_sync(0xffffffff, a[1],  src_B_step1_reg1);

      asm volatile(
          "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(dummy0), "=f"(dummy1),
            "=f"(c[2]), "=f"(c[3]), "=f"(dummy2), "=f"(dummy3)
          : "r"(a_s3_r0), "r"(a_s3_r1), "r"(b_s3_r0), "r"(b_s3_r1),
            "f"(c[0]), "f"(c[1]), "f"(dummy0), "f"(dummy1),
            "f"(c[2]), "f"(c[3]), "f"(dummy2), "f"(dummy3));

      // --- Step 4 (K=12..15) ---
      uint32_t a_s4_r0 = __shfl_sync(0xffffffff, b[1],  src_A_step2);
      uint32_t a_s4_r1 = __shfl_sync(0xffffffff, b2[1], src_A_step2);
      uint32_t b_s4_r0 = __shfl_sync(0xffffffff, a[1],  src_B_step2_reg0);
      uint32_t b_s4_r1 = __shfl_sync(0xffffffff, a[1],  src_B_step2_reg1);

      asm volatile(
          "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(dummy0), "=f"(dummy1),
            "=f"(c[2]), "=f"(c[3]), "=f"(dummy2), "=f"(dummy3)
          : "r"(a_s4_r0), "r"(a_s4_r1), "r"(b_s4_r0), "r"(b_s4_r1),
            "f"(c[0]), "f"(c[1]), "f"(dummy0), "f"(dummy1),
            "f"(c[2]), "f"(c[3]), "f"(dummy2), "f"(dummy3));

    } /*else if constexpr (std::is_same<scalar_t, half>::value &&
                         use_fp16_accum) {
      uint32_t* c = reinterpret_cast<uint32_t*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
          "{%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(b[0]), "r"(b2[0]), "r"(a[0]), "r"(c[0]), "r"(c[1]));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
          "{%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(b[1]), "r"(b2[1]), "r"(a[1]), "r"(c[0]), "r"(c[1]));
    } else if constexpr (std::is_same<scalar_t, int8_t>::value) {
      int32_t* c = reinterpret_cast<int32_t*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
          : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
          : "r"(b[0]), "r"(b2[0]), "r"(a[0]), "r"(c[0]), "r"(c[1]), "r"(c[2]),
            "r"(c[3]));
    }*/
  } /*else {
    if constexpr (std::is_same<scalar_t, int8_t>::value) {
      int32_t* c = reinterpret_cast<int32_t*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(b[0]), "r"(a[0]), "r"(c[0]), "r"(c[1]));
      asm volatile(
          "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=r"(c[2]), "=r"(c[3])
          : "r"(b2[1]), "r"(a[0]), "r"(c[2]), "r"(c[3]));
      asm volatile(
          "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(b[0]), "r"(a[1]), "r"(c[0]), "r"(c[1]));
      asm volatile(
          "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=r"(c[2]), "=r"(c[3])
          : "r"(b2[1]), "r"(a[1]), "r"(c[2]), "r"(c[3]));
    }
  }*/
}

}  // namespace MARLIN_NAMESPACE_NAME
