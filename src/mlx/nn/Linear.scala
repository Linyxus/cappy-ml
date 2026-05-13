package mlx.nn

import scala.python.*
import mlx.core.{Tensor, DataType}

// ---------------------------------------------------------------------------
// Linear and Embedding — direct facades for `mlx.nn.Linear` /
// `mlx.nn.Embedding`. Both inherit `Component` and pick up its full Module
// surface (`train`, `freeze`, `parameters`, `loadWeights`, …) for free.
//
// Linear's forward pass is the standard `apply[DT](x: Tensor[DT]):
// Tensor[DT]` inherited from `Component`. Embedding changes dtype on the
// way through, so it exposes `lookup` instead — see below.
// ---------------------------------------------------------------------------

@extern("mlx.nn", "Linear")
class Linear extends Component:
  /** Weight, shape `[output_dims, input_dims]`. Untyped — dtype is runtime. */
  val weight: PyDynamic = native
  /** Optional bias, shape `[output_dims]`. Absent if constructed `bias = false`. */
  val bias: PyDynamic = native

object Linear:
  inline def apply(inputDims: Int, outputDims: Int): Linear =
    _nn.Linear(inputDims, outputDims).asInstanceOf[Linear]
  inline def apply(inputDims: Int, outputDims: Int, bias: Boolean): Linear =
    _nn.Linear(inputDims, outputDims, bias = bias).asInstanceOf[Linear]

// ---------------------------------------------------------------------------
// Embedding maps integer indices to dense vectors. Input dtype is integer,
// output dtype is the weight's dtype — neither matches the Component-level
// `apply[DT](Tensor[DT]): Tensor[DT]`. So Embedding exposes a separately
// named `lookup[Out, In]` extension and overrides nothing else.
// ---------------------------------------------------------------------------

@extern("mlx.nn", "Embedding")
class Embedding extends Component:
  /** Weight, shape `[num_embeddings, dims]`. Untyped — dtype is runtime. */
  val weight: PyDynamic = native

object Embedding:
  inline def apply(numEmbeddings: Int, dims: Int): Embedding =
    _nn.Embedding(numEmbeddings, dims).asInstanceOf[Embedding]

extension (e: Embedding)
  /** Lookup. `In` is the integer index dtype, `Out` is the weight dtype.
   *  Dispatches through Python `__call__` for the same reason as
   *  `Component.apply` — see Component.scala. */
  inline def lookup[Out <: DataType, In <: DataType](x: Tensor[In]): Tensor[Out] =
    asPy(e).__call__(asPy(x)).asInstanceOf[Tensor[Out]]

  /** Tied-embedding projection: use the transposed weight as a `Linear`. */
  inline def asLinear[DT <: DataType](x: Tensor[DT]): Tensor[DT] =
    asPy(e).as_linear(asPy(x)).asInstanceOf[Tensor[DT]]
