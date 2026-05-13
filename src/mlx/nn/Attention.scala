package mlx.nn

import scala.python.*
import mlx.core.{Tensor, DataType}

// ---------------------------------------------------------------------------
// MultiHeadAttention — scaled dot-product attention across `num_heads`
// parallel heads.
//
// Constructor: `MultiHeadAttention(dims, numHeads)` (+ `bias` overload).
// MLX also accepts separate `query_input_dims` / `key_input_dims` /
// `value_input_dims` / `value_dims` / `value_output_dims` keyword
// arguments — those are rare in practice; users can reach them through
// `mha.toPy.MultiHeadAttention(...)` (the escape hatch from `Component`).
//
// Forward: takes 3+ tensors, which doesn't fit the inherited
// `Component.apply[DT](x: Tensor[DT]): Tensor[DT]` shape, so we expose it
// under a separately-named extension `attend(...)`.
// ---------------------------------------------------------------------------

@extern("mlx.nn", "MultiHeadAttention")
class MultiHeadAttention extends Component:
  /** Query projection. */
  @name("query_proj") val queryProj: PyDynamic = native
  /** Key projection. */
  @name("key_proj") val keyProj: PyDynamic = native
  /** Value projection. */
  @name("value_proj") val valueProj: PyDynamic = native
  /** Output projection. */
  @name("out_proj") val outProj: PyDynamic = native

object MultiHeadAttention:
  inline def apply(dims: Int, numHeads: Int): MultiHeadAttention =
    _nn.MultiHeadAttention(dims, numHeads).asInstanceOf[MultiHeadAttention]
  inline def apply(dims: Int, numHeads: Int, bias: Boolean): MultiHeadAttention =
    _nn.MultiHeadAttention(dims, numHeads, bias = bias).asInstanceOf[MultiHeadAttention]

extension (m: MultiHeadAttention)
  /** Self- or cross-attention without a mask. Dispatches via `__call__` —
   *  same rationale as `Component.apply` (see Component.scala). */
  inline def attend[DT <: DataType](
    queries: Tensor[DT], keys: Tensor[DT], values: Tensor[DT],
  ): Tensor[DT] =
    asPy(m).__call__(asPy(queries), asPy(keys), asPy(values)).asInstanceOf[Tensor[DT]]

  /** Same, plus an additive attention mask (broadcast against the score
   *  matrix). `mask` can carry any dtype — commonly `Bool` or a `Float32`
   *  of `-inf` / `0`. */
  inline def attend[DT <: DataType, MaskDT <: DataType](
    queries: Tensor[DT], keys: Tensor[DT], values: Tensor[DT], mask: Tensor[MaskDT],
  ): Tensor[DT] =
    asPy(m).__call__(asPy(queries), asPy(keys), asPy(values), asPy(mask)).asInstanceOf[Tensor[DT]]
