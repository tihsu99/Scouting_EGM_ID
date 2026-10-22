"""Compile the configured cut-based ID into a raw-input correctionlib evaluator."""
import ast
import math
import re
import sys


class Expressions:
    """Translate the supported pandas expressions without executing Python code."""

    def __init__(self, inputs):
        self.inputs = list(inputs)
        self.symbols = {name: f"__raw_{index}__" for index, name in enumerate(self.inputs)}

    def compile(self, expression):
        quoted = {}

        def quote(match):
            key = f"__quoted_{len(quoted)}"
            quoted[key] = match.group(1)
            return key

        tree = ast.parse(re.sub(r"`([^`]+)`", quote, expression), mode="eval")

        def visit(node):
            if isinstance(node, ast.Name):
                name = quoted.get(node.id, node.id)
                if name not in self.symbols:
                    raise ValueError(f"Unknown ID input or feature: {name}")
                return self.symbols[name]
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if not math.isfinite(node.value):
                    raise ValueError("ID expressions must have finite constants")
                return repr(int(node.value) if isinstance(node.value, bool) else node.value)
            if isinstance(node, ast.BinOp):
                operator = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
                            ast.Pow: "^", ast.BitAnd: "&&", ast.BitOr: "||"}.get(type(node.op))
                if operator:
                    return f"({visit(node.left)}{operator}{visit(node.right)})"
            if isinstance(node, ast.UnaryOp):
                operator = {ast.USub: "-", ast.UAdd: "+", ast.Not: "!", ast.Invert: "!"}.get(type(node.op))
                if operator:
                    return f"({operator}{visit(node.operand)})"
            if isinstance(node, ast.BoolOp):
                operator = "&&" if isinstance(node.op, ast.And) else "||"
                return "(" + operator.join(visit(value) for value in node.values) + ")"
            if isinstance(node, ast.Compare):
                terms = []
                for left, op, right in zip([node.left, *node.comparators], node.ops, node.comparators):
                    operator = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
                                ast.Eq: "==", ast.NotEq: "!="}.get(type(op))
                    if operator is None:
                        break
                    terms.append(f"({visit(left)}{operator}{visit(right)})")
                else:
                    return "(" + "&&".join(terms) + ")"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
                arity = {"abs": 1, "cosh": 1, "sinh": 1, "sqrt": 1, "exp": 1,
                         "log": 1, "log10": 1, "min": 2, "max": 2, "pow": 2}
                if arity.get(node.func.id) == len(node.args):
                    return f"{node.func.id}({','.join(visit(arg) for arg in node.args)})"
            raise ValueError(f"Unsupported ID expression: {expression}")

        return visit(tree.body)

    def define(self, definitions):
        for name, definition in definitions.items():
            expression = self.compile(definition if isinstance(definition, str) else definition["expression"])
            if isinstance(definition, dict):
                for bound, function in (("lower", "max"), ("upper", "min")):
                    if definition.get(bound) is not None:
                        value = float(definition[bound])
                        if not math.isfinite(value):
                            raise ValueError(f"Nonfinite bound for {name}")
                        expression = f"{function}({value!r},{expression})"
            self.symbols[name] = expression

    def formula(self, expression):
        used = [name for index, name in enumerate(self.inputs) if f"__raw_{index}__" in expression]
        for index, name in enumerate(used):
            expression = expression.replace(f"__raw_{self.inputs.index(name)}__", f"x[{index}]")
        finite = "&&".join(f"(abs(x[{index}])<={sys.float_info.max!r})" for index in range(len(used)))
        expression = f"({finite})&&({expression})" if finite else expression
        # TFormula accepts scientific notation without an explicit positive exponent sign.
        expression = re.sub(r"(?<=\d)e\+", "e", expression)
        return used, dict(nodetype="formula", parser="TFormula", variables=used,
                          expression=expression)


def coefficient_expression(correction, eta):
    """Preserve the existing absolute-eta, scalar/binning calibration contract."""
    if (len(correction["inputs"]) != 1 or correction["inputs"][0]["name"] != "eta"
            or correction["inputs"][0]["type"] != "real"):
        raise ValueError(f"Calibration {correction['name']} must have one real eta input")
    data = correction["data"]
    if isinstance(data, (int, float)) and math.isfinite(data):
        return repr(float(data))
    if not isinstance(data, dict) or data.get("nodetype") != "binning" or data.get("input") != "eta":
        raise ValueError(f"Calibration {correction['name']} must use scalar or constant eta bins")
    edges, values = data["edges"], data["content"]
    if data.get("flow") != "clamp" or len(edges) != len(values) + 1:
        raise ValueError(f"Calibration {correction['name']} needs clamped eta bins")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in [*edges, *values]):
        raise ValueError(f"Calibration {correction['name']} needs finite constant bins")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError(f"Invalid eta edges for {correction['name']}")
    terms = []
    for index, value in enumerate(values):
        cuts = []
        if index:
            cuts.append(f"(abs({eta})>={edges[index]!r})")
        if index < len(values) - 1:
            cuts.append(f"(abs({eta})<{edges[index + 1]!r})")
        condition = "&&".join(cuts) or "1"
        terms.append(f"(({condition})*({float(value)!r}))")
    return "(" + "+".join(terms) + ")"


def build_correction_set(base, validation, calibration, wps_by_region):
    preparation = base["workflow"]["InspectDistribution"]
    compiler = Expressions(preparation["input_columns"])
    for correction in calibration["corrections"]:
        compiler.symbols[correction["name"]] = coefficient_expression(correction, compiler.symbols["eta"])
    compiler.define(preparation.get("define_variable", {}))
    compiler.define(validation.get("validation", {}).get("define_variable", {}))
    settings = validation["correctionlib"]
    requirement = compiler.compile(settings["requirement"])
    signatures = [{(source, wp.name) for source, wps in sources.items() for wp in wps}
                  for sources in wps_by_region.values()]
    if not signatures or any(keys != signatures[0] for keys in signatures):
        raise ValueError("Every region must provide the same source/WP names")
    branches = {}
    metadata = {}
    for region, sources in wps_by_region.items():
        acceptance = compiler.compile(base["region_definition"][region]["requirement"])
        for source, wps in sources.items():
            for wp in wps:
                key = f"{source}/{wp.name}"
                cuts = [requirement, acceptance]
                for variable, (low, high) in wp.window_cuts.items():
                    if not math.isfinite(low) or not math.isfinite(high) or low > high:
                        raise ValueError(f"Invalid cut for {region}/{key}/{variable}")
                    expression = compiler.compile(f"`{variable}`")
                    lower, upper = (">", "<") if wp.meta.get("strict", False) else (">=", "<=")
                    cuts.extend((f"({expression}{lower}{low!r})", f"({expression}{upper}{high!r})"))
                branches.setdefault(key, []).append("(" + "&&".join(cuts) + ")")
                metadata[key] = dict(source=source, wp=wp.name, label=wp.meta.get("label", wp.name))
    corrections = []
    for name, terms in branches.items():
        inputs, formula = compiler.formula("(" + "||".join(terms) + ")")
        corrections.append(dict(name=name, version=settings.get("version", 1),
                                description="Cut-based ID: 1=pass, 0=fail; nonfinite inputs fail. "
                                            + "Regions: " + ", ".join(wps_by_region),
                                inputs=[dict(name=name, type="real") for name in inputs],
                                output=dict(name="pass", type="real"), data=formula))
    return dict(schema_version=2, description="Full scouting ID from raw inputs; includes fiducial selection.",
                corrections=corrections), metadata
