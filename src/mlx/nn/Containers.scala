package mlx.nn

import scala.python.*

// ---------------------------------------------------------------------------
// Sequential — applies a sequence of modules in order. MLX's Python ctor
// takes varargs (`Sequential(*modules)`); on the Scala side we mirror that
// with arity-overloaded `apply`s, exactly the same idiom used by
// `Tensor.concatenate` / `Tensor.stack` at `Tensor.scala:350-362`.
//
// Arguments are typed `Component` so any facade layer (`Linear`, `ReLU`,
// `Dropout`, …) composes without an explicit cast. The forward pass is
// the standard `Component.apply` extension.
// ---------------------------------------------------------------------------

@extern("mlx.nn", "Sequential")
class Sequential extends Component

object Sequential:
  inline def apply(a: Component, b: Component): Sequential =
    _nn.Sequential(a, b).asInstanceOf[Sequential]
  inline def apply(a: Component, b: Component, c: Component): Sequential =
    _nn.Sequential(a, b, c).asInstanceOf[Sequential]
  inline def apply(a: Component, b: Component, c: Component, d: Component): Sequential =
    _nn.Sequential(a, b, c, d).asInstanceOf[Sequential]
  inline def apply(a: Component, b: Component, c: Component, d: Component, e: Component): Sequential =
    _nn.Sequential(a, b, c, d, e).asInstanceOf[Sequential]
  inline def apply(a: Component, b: Component, c: Component, d: Component, e: Component, f: Component): Sequential =
    _nn.Sequential(a, b, c, d, e, f).asInstanceOf[Sequential]
  inline def apply(a: Component, b: Component, c: Component, d: Component, e: Component, f: Component, g: Component): Sequential =
    _nn.Sequential(a, b, c, d, e, f, g).asInstanceOf[Sequential]
  inline def apply(a: Component, b: Component, c: Component, d: Component, e: Component, f: Component, g: Component, h: Component): Sequential =
    _nn.Sequential(a, b, c, d, e, f, g, h).asInstanceOf[Sequential]
