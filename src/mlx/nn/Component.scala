package mlx.nn

import scala.python.*

@extern("mlx.nn", "Module")
class Component:
  val training: Boolean = native
