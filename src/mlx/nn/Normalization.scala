package mlx.nn

import scala.python.*

// ---------------------------------------------------------------------------
// Normalisation layers — `LayerNorm`, `RMSNorm`, `BatchNorm`. All three
// are shape- and dtype-preserving, so their forward pass is the standard
// `Component.apply` extension; no per-class override needed.
// ---------------------------------------------------------------------------

// --- LayerNorm -------------------------------------------------------------

@extern("mlx.nn", "LayerNorm")
class LayerNorm extends Component:
  /** Learnable scale, shape `[dims]`. Present only when `affine = true`. */
  val weight: PyDynamic = native
  /** Learnable shift, shape `[dims]`. Present only when `bias = true`. */
  val bias: PyDynamic = native

object LayerNorm:
  inline def apply(dims: Int): LayerNorm =
    _nn.LayerNorm(dims).asInstanceOf[LayerNorm]
  inline def apply(dims: Int, eps: Double): LayerNorm =
    _nn.LayerNorm(dims, eps = eps).asInstanceOf[LayerNorm]
  inline def apply(dims: Int, eps: Double, affine: Boolean): LayerNorm =
    _nn.LayerNorm(dims, eps = eps, affine = affine).asInstanceOf[LayerNorm]
  inline def apply(dims: Int, eps: Double, affine: Boolean, bias: Boolean): LayerNorm =
    _nn.LayerNorm(dims, eps = eps, affine = affine, bias = bias).asInstanceOf[LayerNorm]

// --- RMSNorm ---------------------------------------------------------------

@extern("mlx.nn", "RMSNorm")
class RMSNorm extends Component:
  val weight: PyDynamic = native

object RMSNorm:
  inline def apply(dims: Int): RMSNorm =
    _nn.RMSNorm(dims).asInstanceOf[RMSNorm]
  inline def apply(dims: Int, eps: Double): RMSNorm =
    _nn.RMSNorm(dims, eps = eps).asInstanceOf[RMSNorm]

// --- BatchNorm -------------------------------------------------------------

@extern("mlx.nn", "BatchNorm")
class BatchNorm extends Component:
  val weight: PyDynamic = native
  val bias: PyDynamic = native
  /** Running mean over batches (when `trackRunningStats = true`). */
  @name("running_mean") val runningMean: PyDynamic = native
  /** Running variance over batches (when `trackRunningStats = true`). */
  @name("running_var") val runningVar: PyDynamic = native

object BatchNorm:
  inline def apply(numFeatures: Int): BatchNorm =
    _nn.BatchNorm(numFeatures).asInstanceOf[BatchNorm]
  inline def apply(numFeatures: Int, eps: Double): BatchNorm =
    _nn.BatchNorm(numFeatures, eps = eps).asInstanceOf[BatchNorm]
  inline def apply(numFeatures: Int, eps: Double, momentum: Double): BatchNorm =
    _nn.BatchNorm(numFeatures, eps = eps, momentum = momentum).asInstanceOf[BatchNorm]
  inline def apply(
    numFeatures: Int, eps: Double, momentum: Double,
    affine: Boolean, trackRunningStats: Boolean,
  ): BatchNorm =
    _nn.BatchNorm(
      numFeatures,
      eps = eps, momentum = momentum,
      affine = affine, track_running_stats = trackRunningStats,
    ).asInstanceOf[BatchNorm]
