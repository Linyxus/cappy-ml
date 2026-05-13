package mlx.nn

import scala.python.*

// ---------------------------------------------------------------------------
// Dropout — randomly zeroes a fraction `p` of the input during training.
// In eval mode (`Component.eval()`) it is a no-op. Forward pass is the
// standard `Component.apply` extension.
// ---------------------------------------------------------------------------

@extern("mlx.nn", "Dropout")
class Dropout extends Component

object Dropout:
  /** Default Python `p = 0.5`. */
  inline def apply(): Dropout = _nn.Dropout().asInstanceOf[Dropout]
  inline def apply(p: Double): Dropout = _nn.Dropout(p).asInstanceOf[Dropout]
