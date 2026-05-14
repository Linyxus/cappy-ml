package mlx.nn

import scala.python.*
import mlx.utils.IsSizeType

// ---------------------------------------------------------------------------
// Convolution — `Conv1d` and `Conv2d`. Forward pass uses the standard
// `Component.apply` extension; no per-class override.
//
// `Conv2d`'s `kernelSize` / `stride` / `padding` accept either an `Int` or a
// `(Int, Int)` tuple. That's the same shape vocabulary the Tensor facade
// uses for `reshape` / `transpose` — we reuse the `IsSizeType` typeclass
// (`Tensor.scala:65-68`) to constrain a generic param.
//
// `Conv1d`'s window args are always single `Int`s, so we don't bother with
// the typeclass there — simpler signatures match the API.
// ---------------------------------------------------------------------------

// --- Conv1d ----------------------------------------------------------------

@extern("mlx.nn", "Conv1d")
class Conv1d extends Component:
  val weight: PyDynamic = native
  val bias: PyDynamic = native

object Conv1d:
  inline def apply(inChannels: Int, outChannels: Int, kernelSize: Int): Conv1d =
    _nn.Conv1d(inChannels, outChannels, kernelSize).asInstanceOf[Conv1d]
  inline def apply(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int): Conv1d =
    _nn.Conv1d(inChannels, outChannels, kernelSize, stride = stride).asInstanceOf[Conv1d]
  inline def apply(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, padding: Int): Conv1d =
    _nn.Conv1d(inChannels, outChannels, kernelSize,
      stride = stride, padding = padding).asInstanceOf[Conv1d]
  inline def apply(
    inChannels: Int, outChannels: Int, kernelSize: Int,
    stride: Int, padding: Int, dilation: Int, groups: Int, bias: Boolean,
  ): Conv1d =
    _nn.Conv1d(inChannels, outChannels, kernelSize,
      stride = stride, padding = padding, dilation = dilation,
      groups = groups, bias = bias).asInstanceOf[Conv1d]

// --- Conv2d ----------------------------------------------------------------

@extern("mlx.nn", "Conv2d")
class Conv2d extends Component:
  val weight: PyDynamic = native
  val bias: PyDynamic = native

object Conv2d:
  inline def apply[KS: IsSizeType](inChannels: Int, outChannels: Int, kernelSize: KS): Conv2d =
    _nn.Conv2d(inChannels, outChannels, kernelSize).asInstanceOf[Conv2d]
  inline def apply[KS: IsSizeType, S: IsSizeType](
    inChannels: Int, outChannels: Int, kernelSize: KS, stride: S,
  ): Conv2d =
    _nn.Conv2d(inChannels, outChannels, kernelSize, stride = stride).asInstanceOf[Conv2d]
  inline def apply[KS: IsSizeType, S: IsSizeType, P: IsSizeType](
    inChannels: Int, outChannels: Int, kernelSize: KS, stride: S, padding: P,
  ): Conv2d =
    _nn.Conv2d(inChannels, outChannels, kernelSize,
      stride = stride, padding = padding).asInstanceOf[Conv2d]
  inline def apply[KS: IsSizeType, S: IsSizeType, P: IsSizeType](
    inChannels: Int, outChannels: Int, kernelSize: KS,
    stride: S, padding: P, dilation: Int, groups: Int, bias: Boolean,
  ): Conv2d =
    _nn.Conv2d(inChannels, outChannels, kernelSize,
      stride = stride, padding = padding, dilation = dilation,
      groups = groups, bias = bias).asInstanceOf[Conv2d]
