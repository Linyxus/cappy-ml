# An example of MLX in Python
import mlx.core as mx


def main():
    a = mx.array([10, 20, 30])
    print(a.shape)
    print(a.dtype)
    print(a.device)
    print(type(a))


if __name__ == "__main__":
    main()
