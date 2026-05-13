package mlx.nn

import scala.python.*
import mlx.core.{Tensor, DataType}

// ---------------------------------------------------------------------------
// Activations — exposed two ways:
//   1. As `Component`-extending classes, so they slot into `Sequential` and
//      everywhere else a `Module` is expected. The forward pass is the
//      standard `apply[DT](x: Tensor[DT]): Tensor[DT]` inherited from
//      `Component` — no per-class extension needed.
//   2. As free functions on the `Activations` object, for inline use inside
//      a custom forward pass.
// MLX's own API surfaces both, and the docs cross-reference them.
// ---------------------------------------------------------------------------

// --- module-style activations ---------------------------------------------

@extern("mlx.nn", "ReLU")
class ReLU extends Component
object ReLU:
  inline def apply(): ReLU = _nn.ReLU().asInstanceOf[ReLU]

@extern("mlx.nn", "GELU")
class GELU extends Component
object GELU:
  inline def apply(): GELU = _nn.GELU().asInstanceOf[GELU]

@extern("mlx.nn", "SiLU")
class SiLU extends Component
object SiLU:
  inline def apply(): SiLU = _nn.SiLU().asInstanceOf[SiLU]

@extern("mlx.nn", "Sigmoid")
class Sigmoid extends Component
object Sigmoid:
  inline def apply(): Sigmoid = _nn.Sigmoid().asInstanceOf[Sigmoid]

@extern("mlx.nn", "Tanh")
class Tanh extends Component
object Tanh:
  inline def apply(): Tanh = _nn.Tanh().asInstanceOf[Tanh]

@extern("mlx.nn", "Softmax")
class Softmax extends Component
object Softmax:
  inline def apply(): Softmax = _nn.Softmax().asInstanceOf[Softmax]

// --- functional activations -----------------------------------------------
//
// Same signature pattern as `Tensor.softmax` (Tensor.scala:257-259), just
// dispatching through `_nn` instead of `_mx`. Every dispatch is a single
// `Dynamic` call with a no-op cast on the return.

object Activations:
  inline def relu[DT <: DataType](x: Tensor[DT]): Tensor[DT] =
    _nn.relu(asPy(x)).asInstanceOf[Tensor[DT]]
  inline def gelu[DT <: DataType](x: Tensor[DT]): Tensor[DT] =
    _nn.gelu(asPy(x)).asInstanceOf[Tensor[DT]]
  inline def silu[DT <: DataType](x: Tensor[DT]): Tensor[DT] =
    _nn.silu(asPy(x)).asInstanceOf[Tensor[DT]]
  inline def sigmoid[DT <: DataType](x: Tensor[DT]): Tensor[DT] =
    _nn.sigmoid(asPy(x)).asInstanceOf[Tensor[DT]]
  inline def tanh[DT <: DataType](x: Tensor[DT]): Tensor[DT] =
    _nn.tanh(asPy(x)).asInstanceOf[Tensor[DT]]
  inline def softmax[DT <: DataType](x: Tensor[DT]): Tensor[DT] =
    _nn.softmax(asPy(x)).asInstanceOf[Tensor[DT]]
  inline def softmax[DT <: DataType](x: Tensor[DT], axis: Int): Tensor[DT] =
    _nn.softmax(asPy(x), axis = axis).asInstanceOf[Tensor[DT]]
