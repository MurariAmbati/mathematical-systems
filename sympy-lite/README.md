# symbolic solver

a lightweight symbolic algebra engine for python. implements parsing, simplification, differentiation, and integration of mathematical expressions using an abstract syntax tree representation.

## overview

symbolic solver provides core computer algebra capabilities without external dependencies. designed for clarity, mathematical correctness, and extensibility.

## features

- expression parsing from string format
- algebraic simplification with pattern matching
- symbolic differentiation with chain rule support
- basic symbolic integration
- numeric evaluation with variable substitution
- text and latex output formatting

## installation

```bash
pip install -e .
```

for development:

```bash
pip install -e ".[dev]"
```

## quick start

```python
from symbolic import parse

# parse expression
expr = parse("x^2 + 2*x + 1")

# differentiate
derivative = expr.differentiate("x")
print(derivative.simplify())  # 2 * x + 2

# integrate
integral = expr.integrate("x")
print(integral.simplify())  # x ^ 3.0 / 3.0 + x ^ 2.0 + x

# evaluate
result = expr.evaluate(x=3.0)
print(result)  # 16.0

# latex output
latex = expr.to_latex()
print(latex)  # x^{2} + 2 \cdot x + 1
```

## expression syntax

### operators

- addition: `+`
- subtraction: `-`
- multiplication: `*`
- division: `/`
- exponentiation: `^`

### functions

- trigonometric: `sin(x)`, `cos(x)`, `tan(x)`
- exponential: `exp(x)`
- logarithmic: `log(x)`
- other: `sqrt(x)`, `abs(x)`

### constants

- `pi` - mathematical constant π
- `e` - euler's number

### examples

```python
# polynomial
parse("x^3 - 2*x^2 + x - 1")

# rational function
parse("(x + 1) / (x - 1)")

# trigonometric
parse("sin(x) * cos(x)")

# nested functions
parse("exp(sin(x^2))")
```

## api reference

### parsing

```python
from symbolic import parse

expr = parse("x^2 + 1")
```

### simplification

```python
expr = parse("x + 0")
simplified = expr.simplify()  # returns: x

expr = parse("2 + 3")
simplified = expr.simplify()  # returns: 5
```

simplification rules:
- arithmetic identities: `x + 0 = x`, `x * 1 = x`, `x * 0 = 0`
- constant folding: `2 + 3 = 5`
- power rules: `x^0 = 1`, `x^1 = x`, `(a^b)^c = a^(b*c)`

### differentiation

```python
expr = parse("x^2")
derivative = expr.differentiate("x")
print(derivative.simplify())  # 2 * x
```

supported rules:
- basic: constants to zero, variables to one
- sum and difference rules
- product rule: `d(u*v)/dx = u'*v + u*v'`
- quotient rule: `d(u/v)/dx = (u'*v - u*v')/v^2`
- chain rule: `d(f(g(x)))/dx = f'(g(x)) * g'(x)`
- power rule: `d(x^n)/dx = n*x^(n-1)`
- trigonometric: `d(sin(x))/dx = cos(x)`, `d(cos(x))/dx = -sin(x)`
- exponential: `d(exp(x))/dx = exp(x)`
- logarithmic: `d(log(x))/dx = 1/x`

### integration

```python
expr = parse("x^2")
integral = expr.integrate("x")
print(integral.simplify())  # x ^ 3.0 / 3.0
```

supported integrals:
- polynomials: `∫x^n dx = x^(n+1)/(n+1)`
- exponential: `∫exp(x) dx = exp(x)`
- trigonometric: `∫sin(x) dx = -cos(x)`, `∫cos(x) dx = sin(x)`
- linear substitution: `∫sin(ax) dx = -cos(ax)/a`

### evaluation

```python
expr = parse("x^2 + y^2")
result = expr.evaluate(x=3.0, y=4.0)
print(result)  # 25.0
```

### output formatting

```python
expr = parse("x^2 / (x + 1)")

# string format
print(expr.to_string())
# x ^ 2.0 / (x + 1)

# latex format
print(expr.to_latex())
# \frac{x^{2}}{x + 1}
```

## architecture

### expression tree

all expressions are represented as immutable ast nodes:

- `Constant` - numeric values
- `Variable` - symbolic variables
- `BinaryOp` - binary operations (+, -, *, /, ^)
- `UnaryOp` - unary operations (-, +)
- `Function` - mathematical functions (sin, cos, exp, log)

### node interface

each node implements:
- `simplify() -> Expression`
- `differentiate(var: str) -> Expression`
- `integrate(var: str) -> Expression`
- `evaluate(**values: float) -> float`
- `to_string() -> str`
- `to_latex() -> str`

## examples

### polynomial differentiation

```python
from symbolic import parse

# define polynomial
expr = parse("x^3 - 3*x^2 + 2*x - 1")

# compute derivative
derivative = expr.differentiate("x").simplify()
print(derivative)  # 3 * x ^ 2.0 - 6 * x + 2

# evaluate at point
value = derivative.evaluate(x=2.0)
print(value)  # 2.0
```

### numerical verification

```python
# symbolic derivative
expr = parse("x^2")
symbolic = expr.differentiate("x").simplify()

# numerical derivative (finite difference)
h = 0.0001
x_val = 3.0
f_plus = expr.evaluate(x=x_val + h)
f_minus = expr.evaluate(x=x_val - h)
numerical = (f_plus - f_minus) / (2 * h)

# compare
print(f"symbolic: {symbolic.evaluate(x=x_val)}")
print(f"numerical: {numerical}")
```

### integration verification

```python
# integrate
expr = parse("x^2")
integral = expr.integrate("x").simplify()

# verify by differentiation
check = integral.differentiate("x").simplify()
print(f"original: {expr}")
print(f"integral: {integral}")
print(f"derivative of integral: {check}")
```

## testing

run test suite:

```bash
pytest tests/ -v
```

with coverage:

```bash
pytest tests/ --cov=symbolic --cov-report=html
```

## error handling

the library uses structured exceptions:

- `ParseError` - invalid expression syntax
- `EvaluationError` - evaluation failures (division by zero, undefined variables)
- `IntegrationError` - unsupported integrals
- `SimplificationError` - simplification failures

```python
from symbolic import parse, ParseError, EvaluationError

try:
    expr = parse("x +")
except ParseError as e:
    print(f"parse error: {e}")

try:
    expr = parse("x + y")
    result = expr.evaluate(x=1.0)  # y not provided
except EvaluationError as e:
    print(f"evaluation error: {e}")
```

## performance

designed for correctness over speed. typical performance:

- parsing: ~1000 expressions/second
- simplification: ~5000 operations/second
- differentiation: ~3000 operations/second

## limitations

current limitations:

- integration requires symbolic forms (no numerical methods)
- no support for complex numbers
- limited trigonometric simplification
- no matrix operations
- no equation solving

## contributing

contributions welcome. ensure:

- pep8 compliance
- type annotations for public methods
- unit tests for new features
- documentation for new apis

## license

mit license. see LICENSE file for details.

## acknowledgments

inspired by sympy and mathematica. built for educational purposes and lightweight symbolic computation. for fun most importantly. 
