import numpy as np
import matplotlib.pyplot as plt
import argparse
import ast
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


EPS = 1e-12
LARGE = 1e300

def safe_div(x, y, eps=EPS):
    if abs(y) < eps:
        if x > 0:
            return np.inf
        if x < 0:
            return -np.inf
        return np.nan
    return x / y


def safe_log(x, eps=EPS):
    if x < 0:
        return np.nan
    if x <= eps:
        return -np.inf
    return np.log(x)


def safe_log2(x, eps=EPS):
    if x < 0:
        return np.nan
    if x <= eps:
        return -np.inf
    return np.log2(x)


def safe_log10(x, eps=EPS):
    if x < 0:
        return np.nan
    if x <= eps:
        return -np.inf
    return np.log10(x)


def safe_sqrt(x, eps=EPS):
    if x < -eps:
        return np.nan
    return np.sqrt(max(x, 0.0))


def safe_pow(x, y):
    try:
        if x < 0 and abs(y - round(y)) > 1e-10:
            return np.nan
        return x ** y
    except Exception:
        return np.nan


def safe_exp(x):
    try:
        return np.exp(x)
    except Exception:
        return np.nan


def safe_abs(x):
    return abs(x)


def safe_min(*args):
    try:
        return min(args)
    except Exception:
        return np.nan


def safe_max(*args):
    try:
        return max(args)
    except Exception:
        return np.nan


def safe_sin(x):
    try:
        return np.sin(x)
    except Exception:
        return np.nan


def safe_cos(x):
    try:
        return np.cos(x)
    except Exception:
        return np.nan


def safe_tan(x):
    try:
        return np.tan(x)
    except Exception:
        return np.nan


def safe_clip(x, lo, hi):
    try:
        return min(max(x, lo), hi)
    except Exception:
        return np.nan


def safe_mean(*args):
    try:
        if len(args) == 0:
            return np.nan
        return sum(args) / len(args)
    except Exception:
        return np.nan


_ALLOWED_FUNCS = {
    "log": safe_log,
    "log2": safe_log2,
    "log10": safe_log10,
    "exp": safe_exp,
    "sqrt": safe_sqrt,
    "abs": safe_abs,
    "min": safe_min,
    "max": safe_max,
    "sin": safe_sin,
    "cos": safe_cos,
    "tan": safe_tan,
    "clip": safe_clip,
    "mean": safe_mean,
}

_ALLOWED_CONSTS = {
    "pi": np.pi,
    "e": np.e,
}

PRESETS = {
    "accuracy": "(a+c)/(a+b+c+d)",
    "precision": "a/(a+b)",
    "recall": "a/(a+d)",
    "specificity": "c/(b+c)",
    "f1": "2*(a/(a+b))*(a/(a+d))/((a/(a+b))+(a/(a+d)))",
    "distance_center": "sqrt((a-0.25)^2 + (b-0.25)^2 + (c-0.25)^2 + (d-0.25)^2)",
    "entropy": "-(a*log(a)+b*log(b)+c*log(c)+d*log(d))",
    "max_prob": "max(a,b,c,d)",
    "sum_squares": "a^2+b^2+c^2+d^2",
}

class SafeExpressionEvaluator:
    def __init__(self, expression: str):
        self.original_expression = expression
        self.expression = self._preprocess(expression)
        self.tree = ast.parse(self.expression, mode="eval")
        self._validate(self.tree)

    def _preprocess(self, expr: str) -> str:
        return expr.replace("^", "**")

    def _validate(self, node):
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,
            ast.Name,
            ast.Load,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.Mod,
            ast.USub,
            ast.UAdd,
            ast.Call,
            ast.Tuple,
            ast.List,
        )

        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Niedozwolony element składni: {type(node).__name__}")

        for child in ast.iter_child_nodes(node):
            self._validate(child)

        if isinstance(node, ast.Name):
            allowed_names = {"a", "b", "c", "d"} | set(_ALLOWED_FUNCS.keys()) | set(_ALLOWED_CONSTS.keys())
            if node.id not in allowed_names:
                raise ValueError(f"Niedozwolona nazwa: {node.id}")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Dozwolone są tylko proste wywołania funkcji")
            if node.func.id not in _ALLOWED_FUNCS:
                raise ValueError(f"Niedozwolona funkcja: {node.func.id}")

        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, bool)):
                raise ValueError(f"Niedozwolona stała: {node.value}")

    def eval_scalar(self, a, b, c, d):
        try:
            val = self._eval_node(self.tree.body, a, b, c, d)
            if isinstance(val, complex):
                return np.nan
            if np.isfinite(val) and abs(val) > LARGE:
                return np.nan
            return float(val)
        except Exception:
            return np.nan

    def _eval_node(self, node, a, b, c, d):
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            if node.id == "a":
                return a
            if node.id == "b":
                return b
            if node.id == "c":
                return c
            if node.id == "d":
                return d
            if node.id in _ALLOWED_CONSTS:
                return _ALLOWED_CONSTS[node.id]
            raise ValueError(f"Nieznana nazwa: {node.id}")

        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, a, b, c, d)
            right = self._eval_node(node.right, a, b, c, d)

            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                return safe_div(left, right)
            elif isinstance(node.op, ast.Pow):
                return safe_pow(left, right)
            elif isinstance(node.op, ast.Mod):
                return np.nan if abs(right) < EPS else (left % right)
            else:
                raise ValueError(f"Nieobsługiwany operator: {type(node.op).__name__}")

        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, a, b, c, d)

            if isinstance(node.op, ast.USub):
                return -operand
            elif isinstance(node.op, ast.UAdd):
                return +operand
            else:
                raise ValueError(f"Nieobsługiwany operator unarny: {type(node.op).__name__}")

        if isinstance(node, ast.Call):
            func_name = node.func.id
            func = _ALLOWED_FUNCS[func_name]
            args = [self._eval_node(arg, a, b, c, d) for arg in node.args]
            return func(*args)

        if isinstance(node, (ast.Tuple, ast.List)):
            return [self._eval_node(elt, a, b, c, d) for elt in node.elts]

        raise ValueError(f"Nieobsługiwany typ AST: {type(node).__name__}")


class SafeFilterEvaluator:
    def __init__(self, expression: str):
        self.original_expression = expression
        self.expression = self._preprocess(expression)
        self.tree = ast.parse(self.expression, mode="eval")
        self._validate(self.tree)

    def _preprocess(self, expr: str) -> str:
        expr = expr.replace("^", "**")
        expr = expr.replace("&", " and ")
        expr = expr.replace("|", " or ")
        return expr

    def _validate(self, node):
        allowed_nodes = (
            ast.Expression,
            ast.BoolOp,
            ast.And,
            ast.Or,
            ast.UnaryOp,
            ast.Not,
            ast.Compare,
            ast.BinOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.Mod,
            ast.USub,
            ast.UAdd,
            ast.Call,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.Tuple,
            ast.List,
            ast.Gt,
            ast.GtE,
            ast.Lt,
            ast.LtE,
            ast.Eq,
            ast.NotEq,
        )

        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Niedozwolony element w filtrze: {type(node).__name__}")

        for child in ast.iter_child_nodes(node):
            self._validate(child)

        if isinstance(node, ast.Name):
            allowed_names = {"a", "b", "c", "d"} | set(_ALLOWED_FUNCS.keys()) | set(_ALLOWED_CONSTS.keys())
            if node.id not in allowed_names:
                raise ValueError(f"Niedozwolona nazwa w filtrze: {node.id}")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Dozwolone są tylko proste wywołania funkcji w filtrze")
            if node.func.id not in _ALLOWED_FUNCS:
                raise ValueError(f"Niedozwolona funkcja w filtrze: {node.func.id}")

    def eval_bool(self, a, b, c, d):
        try:
            val = self._eval_node(self.tree.body, a, b, c, d)
            return bool(val)
        except Exception:
            return False

    def _eval_numeric_or_bool(self, node, a, b, c, d):
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            if node.id == "a":
                return a
            if node.id == "b":
                return b
            if node.id == "c":
                return c
            if node.id == "d":
                return d
            if node.id in _ALLOWED_CONSTS:
                return _ALLOWED_CONSTS[node.id]
            raise ValueError(f"Nieznana nazwa: {node.id}")

        if isinstance(node, ast.BinOp):
            left = self._eval_numeric_or_bool(node.left, a, b, c, d)
            right = self._eval_numeric_or_bool(node.right, a, b, c, d)

            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return safe_div(left, right)
            if isinstance(node.op, ast.Pow):
                return safe_pow(left, right)
            if isinstance(node.op, ast.Mod):
                return np.nan if abs(right) < EPS else (left % right)

            raise ValueError(f"Nieobsługiwany operator: {type(node.op).__name__}")

        if isinstance(node, ast.UnaryOp):
            operand = self._eval_numeric_or_bool(node.operand, a, b, c, d)

            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.Not):
                return not bool(operand)

            raise ValueError(f"Nieobsługiwany operator unarny: {type(node.op).__name__}")

        if isinstance(node, ast.Call):
            func_name = node.func.id
            func = _ALLOWED_FUNCS[func_name]
            args = [self._eval_numeric_or_bool(arg, a, b, c, d) for arg in node.args]
            return func(*args)

        if isinstance(node, (ast.Tuple, ast.List)):
            return [self._eval_numeric_or_bool(elt, a, b, c, d) for elt in node.elts]

        if isinstance(node, ast.Compare):
            left = self._eval_numeric_or_bool(node.left, a, b, c, d)
            if isinstance(left, (float, np.floating)) and np.isnan(left):
                return False

            current = left
            for op, comp in zip(node.ops, node.comparators):
                right = self._eval_numeric_or_bool(comp, a, b, c, d)
                if isinstance(right, (float, np.floating)) and np.isnan(right):
                    return False

                if isinstance(op, ast.Gt):
                    ok = current > right
                elif isinstance(op, ast.GtE):
                    ok = current >= right
                elif isinstance(op, ast.Lt):
                    ok = current < right
                elif isinstance(op, ast.LtE):
                    ok = current <= right
                elif isinstance(op, ast.Eq):
                    ok = current == right
                elif isinstance(op, ast.NotEq):
                    ok = current != right
                else:
                    raise ValueError(f"Nieobsługiwane porównanie: {type(op).__name__}")

                if not ok:
                    return False
                current = right

            return True

        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                for v in node.values:
                    if not bool(self._eval_node(v, a, b, c, d)):
                        return False
                return True

            if isinstance(node.op, ast.Or):
                for v in node.values:
                    if bool(self._eval_node(v, a, b, c, d)):
                        return True
                return False

            raise ValueError(f"Nieobsługiwany operator logiczny: {type(node.op).__name__}")

        raise ValueError(f"Nieobsługiwany typ AST: {type(node).__name__}")

    def _eval_node(self, node, a, b, c, d):
        if isinstance(node, (ast.Compare, ast.BoolOp)):
            return self._eval_numeric_or_bool(node, a, b, c, d)

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self._eval_numeric_or_bool(node, a, b, c, d)

        val = self._eval_numeric_or_bool(node, a, b, c, d)

        if isinstance(val, (bool, np.bool_)):
            return bool(val)

        if isinstance(val, (float, int, np.floating, np.integer)):
            if np.isnan(val):
                return False
            return bool(val)

        return bool(val)

def simplex4_to_cartesian(probs):
    probs = np.asarray(probs, dtype=float)

    if probs.ndim == 1:
        probs = probs.reshape(1, -1)

    v0 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2, 0.0])
    v3 = np.array([0.5, np.sqrt(3) / 6, np.sqrt(2 / 3)])

    xyz = (
        probs[:, 0:1] * v0 +
        probs[:, 1:2] * v1 +
        probs[:, 2:3] * v2 +
        probs[:, 3:4] * v3
    )
    return xyz


def tetrahedron_vertices():
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3) / 2, 0.0],
        [0.5, np.sqrt(3) / 6, np.sqrt(2 / 3)]
    ])


def draw_tetrahedron_edges(ax):
    vertices = tetrahedron_vertices()
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3)
    ]

    for i, j in edges:
        ax.plot(
            [vertices[i, 0], vertices[j, 0]],
            [vertices[i, 1], vertices[j, 1]],
            [vertices[i, 2], vertices[j, 2]],
            color="black",
            linewidth=1.1
        )

    return vertices

def generate_simplex_lattice(n=30):
    """
    Regularna siatka na sympleksie:
        a,b,c,d >= 0
        a+b+c+d = 1

    Parametr n oznacza liczbę poziomów na osi.
    Dla n=2 masz poziomy 0 i 1.
    Dla n=11 masz poziomy 0, 0.1, 0.2, ..., 1.0
    """
    if n < 2:
        raise ValueError("n musi być >= 2")

    pts = []
    denom = n - 1

    for ia in range(n):
        a = ia / denom
        remaining_abcd = denom - ia
        for ib in range(remaining_abcd + 1):
            b = ib / denom
            remaining_cd = remaining_abcd - ib
            for ic in range(remaining_cd + 1):
                c = ic / denom
                id_ = remaining_cd - ic
                d = id_ / denom
                pts.append([a, b, c, d])

    return np.array(pts, dtype=float)

def evaluate_expression_on_points(points, evaluator):
    return np.array(
        [evaluator.eval_scalar(a, b, c, d) for a, b, c, d in points],
        dtype=float
    )


def evaluate_filter_on_points(points, filter_evaluator):
    return np.array(
        [filter_evaluator.eval_bool(a, b, c, d) for a, b, c, d in points],
        dtype=bool
    )


def estimate_function_range(evaluator, all_points):
    values = evaluate_expression_on_points(all_points, evaluator)

    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        raise ValueError("Funkcja nie przyjmuje żadnych skończonych wartości na sympleksie.")

    finite_values = values[finite_mask]
    return np.min(finite_values), np.max(finite_values)


def scale_values_to_unit_interval(values, vmin, vmax):
    scaled = np.full_like(values, np.nan, dtype=float)
    finite_mask = np.isfinite(values)

    if not np.any(finite_mask):
        return scaled

    if np.isclose(vmin, vmax):
        scaled[finite_mask] = 0.5
        return scaled

    scaled[finite_mask] = (values[finite_mask] - vmin) / (vmax - vmin)
    scaled[finite_mask] = np.clip(scaled[finite_mask], 0.0, 1.0)
    return scaled


# ============================================================
# Rysowanie
# ============================================================

def plot_function_tetrahedron(
    points,
    raw_values,
    scaled_values,
    display_mask=None,
    class_names=None,
    title="Wizualizacja funkcji na sympleksie 4D",
    expr_label="f(a,b,c,d)",
    show_invalid=True,
    save_path=None
):
    xyz = simplex4_to_cartesian(points)

    if class_names is None:
        class_names = ["A", "B", "C", "D"]

    if display_mask is None:
        display_mask = np.ones(len(points), dtype=bool)

    finite_mask = np.isfinite(raw_values) & np.isfinite(scaled_values)
    bad_mask = ~finite_mask

    shown_valid_mask = display_mask & finite_mask
    shown_bad_mask = display_mask & bad_mask

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if np.any(shown_valid_mask):
        sc = ax.scatter(
            xyz[shown_valid_mask, 0],
            xyz[shown_valid_mask, 1],
            xyz[shown_valid_mask, 2],
            c=scaled_values[shown_valid_mask],
            s=16,
            alpha=0.9,
            cmap="viridis"
        )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.08)
        cbar.set_label("Wartość funkcji po skalowaniu do [0,1]")

    if show_invalid and np.any(shown_bad_mask):
        ax.scatter(
            xyz[shown_bad_mask, 0],
            xyz[shown_bad_mask, 1],
            xyz[shown_bad_mask, 2],
            s=16,
            alpha=0.95,
            c="black"
        )

    vertices = draw_tetrahedron_edges(ax)

    for i, name in enumerate(class_names):
        ax.text(
            vertices[i, 0],
            vertices[i, 1],
            vertices[i, 2] + 0.03,
            name,
            fontsize=11,
            ha="center"
        )

    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.view_init(elev=22, azim=35)

    shown_count = int(np.sum(display_mask))
    total_count = len(points)
    bad_count = int(np.sum(shown_bad_mask))
    fig.text(
        0.02, 0.02,
        f"{expr_label}\npokazane={shown_count}/{total_count}\nczarne={bad_count}",
        fontsize=10
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")

    plt.show()


# ============================================================
# Przykłady użycia
# ============================================================

def print_examples():
    examples = r"""
Przykłady użycia:

1) Accuracy:
   python .\4dVisualization.py --preset accuracy --labels TP FP TN FN

2) Precision:
   python .\4dVisualization.py --preset precision --labels TP FP TN FN

3) Recall:
   python .\4dVisualization.py --preset recall --labels TP FP TN FN

4) F1:
   python .\4dVisualization.py --preset f1 --labels TP FP TN FN

5) Entropia:
   python .\4dVisualization.py --preset entropy

6) Odległość od środka:
   python .\4dVisualization.py --preset distance_center

7) Własna funkcja:
   python .\4dVisualization.py -f "a+c"

8) Filtrowanie widoku:
   python .\4dVisualization.py -f "a+c" --labels TP FP TN FN --filter "d>0.2 and b<0.1"

9) Pokazanie środka tetraedru:
   python .\4dVisualization.py -f "a+c" --filter "min(a,b,c,d)>0.12"

10) Gęstsza siatka:
   python .\4dVisualization.py --preset accuracy --labels TP FP TN FN --n 60

11) Bardzo gęsta siatka:
   python .\4dVisualization.py --preset entropy --n 100

12) Zapis do pliku:
   python .\4dVisualization.py --preset entropy --save wynik.png

Uwaga o parametrze --n:
   Teraz --n oznacza liczbę poziomów siatki na osi.
   To nie jest liczba losowych punktów.

   Przykładowo:
   --n 11  -> poziomy 0.0, 0.1, ..., 1.0
   --n 21  -> poziomy 0.0, 0.05, ..., 1.0

Dostępne presety:
   accuracy, precision, recall, specificity, f1, entropy,
   distance_center, max_prob, sum_squares

Dostępne funkcje w -f i --filter:
   log, log2, log10, exp, sqrt, abs, min, max, sin, cos, tan, clip, mean
"""
    print(examples.strip())

def parse_args():
    epilog_text = """
Przykład:
  python .\\4dVisualization.py -f "a+c" --labels TP FP TN FN --filter "d>0.2 and b<0.1"

Albo preset:
  python .\\4dVisualization.py --preset accuracy --labels TP FP TN FN

Więcej przykładów:
  python .\\4dVisualization.py --examples
"""

    parser = argparse.ArgumentParser(
        description="Wizualizacja funkcji f(a,b,c,d) na sympleksie 4D z regularnej siatki.",
        epilog=epilog_text,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "-f", "--function",
        type=str,
        default=None,
        help='Wyrażenie funkcji, np. "a/(a+b)", "log(a+b)", "a^2+b^2+c^2+d^2"'
    )

    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=sorted(PRESETS.keys()),
        help="Gotowa funkcja"
    )

    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help='Filtr widoku, np. "d>0.3 and a>0.4" albo "a+b<0.6"'
    )

    parser.add_argument(
        "--n",
        type=int,
        default=30,
        help="Liczba poziomów siatki na osi"
    )

    parser.add_argument(
        "--labels",
        type=str,
        nargs=4,
        default=["A", "B", "C", "D"],
        help="Etykiety wierzchołków"
    )

    parser.add_argument(
        "--hide-invalid",
        action="store_true",
        help="Ukryj czarne punkty (inf/nan)"
    )

    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Zapis wykresu do pliku"
    )

    parser.add_argument(
        "--examples",
        action="store_true",
        help="Wypisz przykłady użycia i zakończ"
    )

    args = parser.parse_args()

    if not args.examples and not args.function and not args.preset:
        parser.error("podaj -f/--function albo --preset, chyba że używasz --examples")

    if args.function and args.preset:
        parser.error("użyj albo -f/--function, albo --preset, nie obu naraz")

    return args

def main():
    args = parse_args()

    if args.examples:
        print_examples()
        return

    function_expr = PRESETS[args.preset] if args.preset else args.function

    try:
        evaluator = SafeExpressionEvaluator(function_expr)
    except Exception as e:
        print(f"Błąd parsowania funkcji: {e}")
        return

    filter_evaluator = None
    if args.filter is not None:
        try:
            filter_evaluator = SafeFilterEvaluator(args.filter)
        except Exception as e:
            print(f"Błąd parsowania filtra: {e}")
            return

    try:
        points = generate_simplex_lattice(n=args.n)
    except Exception as e:
        print(f"Błąd generowania siatki: {e}")
        return

    try:
        vmin, vmax = estimate_function_range(
            evaluator=evaluator,
            all_points=points
        )
    except Exception as e:
        print(f"Błąd przy szacowaniu zakresu funkcji: {e}")
        return

    raw_values = evaluate_expression_on_points(points, evaluator)
    scaled_values = scale_values_to_unit_interval(raw_values, vmin, vmax)

    if filter_evaluator is None:
        display_mask = np.ones(len(points), dtype=bool)
    else:
        display_mask = evaluate_filter_on_points(points, filter_evaluator)

    title_expr = args.preset if args.preset else args.function
    title = f"Wizualizacja 3D: {title_expr}"
    title += f"\nsiatka regularna, n={args.n}"
    if args.filter:
        title += f"\nfiltr widoku: {args.filter}"

    plot_function_tetrahedron(
        points=points,
        raw_values=raw_values,
        scaled_values=scaled_values,
        display_mask=display_mask,
        class_names=args.labels,
        title=title,
        expr_label=function_expr,
        show_invalid=not args.hide_invalid,
        save_path=args.save
    )


if __name__ == "__main__":
    main()