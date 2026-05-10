import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def pairwise_distances(points):
    n = points.shape[0]
    distances = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            distances[i, j] = np.linalg.norm(points[i] - points[j])

    return distances


def read_input_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    if not lines:
        raise ValueError("The input file is empty.")

    header = lines[0].split()
    input_type = header[0].upper()

    if input_type == "X":
        attribute_names = header[1:]
        object_names = []
        data = []

        for line in lines[1:]:
            parts = line.split()

            if len(parts) != len(attribute_names) + 1:
                raise ValueError(f"Invalid number of values in line: {line}")

            object_names.append(parts[0])
            data.append([float(value) for value in parts[1:]])

        x_matrix = np.array(data, dtype=float)
        target_distances = pairwise_distances(x_matrix)

        return object_names, target_distances

    if input_type == "D":
        column_names = header[1:]
        object_names = []
        distances = []

        for line in lines[1:]:
            parts = line.split()

            if len(parts) != len(column_names) + 1:
                raise ValueError(f"Invalid number of values in line: {line}")

            object_names.append(parts[0])
            distances.append([float(value) for value in parts[1:]])

        target_distances = np.array(distances, dtype=float)

        if target_distances.shape[0] != target_distances.shape[1]:
            raise ValueError("Matrix D must be square.")

        if len(object_names) != len(column_names):
            raise ValueError("The number of row and column object identifiers must be the same.")

        if object_names != column_names:
            raise ValueError("Row and column object identifiers are not the same.")

        return object_names, target_distances

    raise ValueError("The first value in the input file must be X or D.")


def frobenius_norm(matrix_a, matrix_b):
    return np.sqrt(np.sum((matrix_a - matrix_b) ** 2))


def objective_function(flat_coordinates, target_distances):
    coordinates = flat_coordinates.reshape(-1, 2)
    current_distances = pairwise_distances(coordinates)
    return frobenius_norm(target_distances, current_distances)


def find_2d_embedding(target_distances, restarts, maxiter, seed):
    rng = np.random.default_rng(seed)

    object_count = target_distances.shape[0]
    best_result = None

    for _ in range(restarts):
        initial_coordinates = rng.normal(size=(object_count, 2))
        initial_flat = initial_coordinates.flatten()

        result = minimize(
            objective_function,
            initial_flat,
            args=(target_distances,),
            method="BFGS",
            options={"maxiter": maxiter}
        )

        if best_result is None or result.fun < best_result.fun:
            best_result = result

    coordinates = best_result.x.reshape(-1, 2)

    coordinates = coordinates - coordinates.mean(axis=0)

    return coordinates, best_result.fun


def plot_embedding(object_names, coordinates, quality, output_file=None):
    plt.figure(figsize=(7, 7))
    plt.scatter(coordinates[:, 0], coordinates[:, 1])

    for name, point in zip(object_names, coordinates):
        plt.text(point[0], point[1], f" {name}", fontsize=10)

    plt.title(f"2D visualization, Frobenius norm = {quality:.6f}")
    plt.xlabel("dimension 1")
    plt.ylabel("dimension 2")
    plt.grid(True)
    plt.axis("equal")

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="2D visualization of objects based on an X or D matrix."
    )

    parser.add_argument(
        "-f",
        "--file",
        required=True,
        help="Path to the input file of type X or D."
    )

    parser.add_argument(
        "--restarts",
        type=int,
        default=20,
        help="Number of independent optimization restarts."
    )

    parser.add_argument(
        "--maxiter",
        type=int,
        default=1000,
        help="Maximum number of optimization iterations."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed."
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for saving the plot, for example result.png."
    )

    args = parser.parse_args()

    try:
        object_names, target_distances = read_input_file(args.file)
    except Exception as error:
        print(f"Input file error: {error}")
        sys.exit(1)

    coordinates, quality = find_2d_embedding(
        target_distances=target_distances,
        restarts=args.restarts,
        maxiter=args.maxiter,
        seed=args.seed
    )

    current_distances = pairwise_distances(coordinates)

    print("Objects:")
    print(object_names)

    print("\nTarget distance matrix D:")
    print(target_distances)

    print("\nFound 2D coordinates:")
    for name, point in zip(object_names, coordinates):
        print(f"{name}: {point[0]:.6f}, {point[1]:.6f}")

    print("\nCurrent distance matrix for 2D points:")
    print(current_distances)

    print(f"\nFrobenius norm: {quality:.6f}")

    plot_embedding(object_names, coordinates, quality, args.output)


if __name__ == "__main__":
    main()