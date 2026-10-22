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
                if isinstance(node.op, (ast.BitAnd, ast.BitOr)):
                    left, right = visit(node.left), visit(node.right)
                    if isinstance(node.op, ast.BitAnd):
                        return f"(({left}!=0)*({right}!=0))"
                    return f"((({left}!=0)+({right}!=0))>0)"
                operator = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
                            ast.Pow: "^"}.get(type(node.op))
                if operator:
                    return f"({visit(node.left)}{operator}{visit(node.right)})"
            if isinstance(node, ast.UnaryOp):
                if isinstance(node.op, (ast.Not, ast.Invert)):
                    return f"({visit(node.operand)}==0)"
                if isinstance(node.op, (ast.USub, ast.UAdd)):
                    return f"(-{visit(node.operand)})" if isinstance(node.op, ast.USub) else visit(node.operand)
            if isinstance(node, ast.BoolOp):
                terms = [f"({visit(value)}!=0)" for value in node.values]
                return ("(" + "*".join(terms) + ")" if isinstance(node.op, ast.And)
                        else "((" + "+".join(terms) + ")>0)")
            if isinstance(node, ast.Compare):
                terms = []
                for left, op, right in zip([node.left, *node.comparators], node.ops, node.comparators):
                    operator = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
                                ast.Eq: "==", ast.NotEq: "!="}.get(type(op))
                    if operator is None:
                        break
                    terms.append(f"({visit(left)}{operator}{visit(right)})")
                else:
                    return "(" + "*".join(terms) + ")"
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

    def formula(self, expression, check_finite=True):
        used = [name for index, name in enumerate(self.inputs) if f"__raw_{index}__" in expression]
        if len(used) > 4:
            raise ValueError(f"A legacy-compatible ID component supports at most four raw inputs: {used}")
        for index, name in enumerate(used):
            expression = expression.replace(f"__raw_{self.inputs.index(name)}__", "xyzt"[index])
        # Older TFormula parsers support comparisons and arithmetic, but no logical operators.
        finite = "*".join(f"(abs({symbol})<={sys.float_info.max!r})" for symbol in "xyzt"[:len(used)])
        expression = f"({finite})*(({expression})!=0)" if finite and check_finite else expression
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
        condition = "*".join(cuts) or "1"
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
    regions = {region: compiler.compile(base["region_definition"][region]["requirement"])
               for region in wps_by_region}
    branches = {}
    metadata = {}
    for region, sources in wps_by_region.items():
        for source, wps in sources.items():
            for wp in wps:
                key = f"{source}/{wp.name}"
                cuts = {}
                for variable, (low, high) in wp.window_cuts.items():
                    if not math.isfinite(low) or not math.isfinite(high) or low > high:
                        raise ValueError(f"Invalid cut for {region}/{key}/{variable}")
                    expression = compiler.compile(f"`{variable}`")
                    lower, upper = (">", "<") if wp.meta.get("strict", False) else (">=", "<=")
                    cuts[variable] = f"(({expression}{lower}{low!r})*({expression}{upper}{high!r}))"
                branches.setdefault(key, {})[region] = cuts
                metadata[key] = dict(source=source, wp=wp.name, label=wp.meta.get("label", wp.name))
    corrections = []

    def add_component(name, expression, region_guard=False):
        inputs, data = compiler.formula(expression, check_finite=not region_guard)
        if region_guard:
            # Reject overlapping regions rather than combining cuts from different regions.
            if not inputs:
                raise ValueError("ID region definitions must depend on raw inputs")
            data = dict(nodetype="transform", input=inputs[0], rule=data,
                        content=dict(nodetype="binning", input=inputs[0], edges=[-.5, 1.5],
                                     content=[1.], flow="error"))
        corrections.append(dict(name=name, version=settings.get("version", 1),
                                description="Internal ID component; evaluate the compound WP for the full ID.",
                                inputs=[dict(name=name, type="real") for name in inputs],
                                output=dict(name="pass", type="real"), data=data))
        return inputs

    common = ["__id__/exclusive_regions", "__id__/fiducial"]
    common_inputs = set(add_component(common[0], "+".join(f"({value})" for value in regions.values()),
                                      region_guard=True))
    common_inputs.update(add_component(common[1], requirement))
    compounds = []
    for name, regional_cuts in branches.items():
        inputs, stack = set(common_inputs), list(common)
        variables = list(dict.fromkeys(variable for cuts in regional_cuts.values() for variable in cuts))
        # Each cut sees the original raw inputs; multiplying binary results applies all cuts.
        for index, variable in enumerate(variables or [None]):
            expression = "+".join(f"({regions[region]}*({cuts.get(variable, '1')}))"
                                  for region, cuts in regional_cuts.items())
            component = f"__id__/{name}/cut_{index}"
            inputs.update(add_component(component, f"({expression})"))
            stack.append(component)
        compounds.append(dict(name=name,
                              description="Full ID: 1=pass, 0=fail; nonfinite inputs fail. "
                                          "Regions must not overlap: " + ", ".join(regions),
                              inputs=[dict(name=item, type="real") for item in compiler.inputs if item in inputs],
                              output=dict(name="pass", type="real"), inputs_update=[],
                              input_op="*", output_op="*", stack=stack))
    return dict(schema_version=2, description="Full scouting ID from raw inputs; includes fiducial selection.",
                corrections=corrections, compound_corrections=compounds), metadata
