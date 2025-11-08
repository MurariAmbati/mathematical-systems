# usage guide

quick reference for symbolic solver operations.

## basic workflow

```python
from symbolic import parse

# 1. parse expression
expr = parse("x^2 + 2*x + 1")

# 2. perform operations
simplified = expr.simplify()
derivative = expr.differentiate("x")
integral = expr.integrate("x")

# 3. evaluate numerically
result = expr.evaluate(x=3.0)

# 4. format output
text = expr.to_string()
latex = expr.to_latex()
```

## common patterns

### polynomial manipulation

```python
# expand and simplify
expr = parse("(x + 1)^2")  # not automatic
expr = parse("x^2 + 2*x + 1")  # explicit form
simplified = expr.simplify()
```

### calculus operations

```python
# compute derivative
f = parse("x^3 - 3*x^2 + 2*x")
f_prime = f.differentiate("x").simplify()

# compute integral
g = parse("x^2")
g_integral = g.integrate("x").simplify()

# verify calculus operations
check = g_integral.differentiate("x").simplify()
# check should equal g
```

### numeric evaluation

```python
# single variable
expr = parse("x^2 + 1")
value = expr.evaluate(x=5.0)  # 26.0

# multiple variables
expr = parse("x*y + x^2")
value = expr.evaluate(x=2.0, y=3.0)  # 10.0

# with special constants
expr = parse("pi * x^2")
value = expr.evaluate(x=1.0)  # π
```

### working with functions

```python
# trigonometric
expr = parse("sin(x)^2 + cos(x)^2")
# note: does not automatically simplify to 1

# exponential
expr = parse("exp(x) * exp(y)")
# does not automatically combine exponents

# nested functions
expr = parse("sin(cos(exp(x)))")
derivative = expr.differentiate("x").simplify()
```

## operator precedence

from highest to lowest:

1. functions: `sin(...)`, `exp(...)`
2. exponentiation: `^` (right associative)
3. unary: `-x`, `+x`
4. multiplication/division: `*`, `/`
5. addition/subtraction: `+`, `-`

use parentheses to override:

```python
parse("2 + 3 * 4")      # = 2 + (3 * 4) = 14
parse("(2 + 3) * 4")    # = 5 * 4 = 20

parse("2^3^4")          # = 2^(3^4) = 2^81
parse("(2^3)^4")        # = 8^4 = 4096
```

## error handling

```python
from symbolic import parse, ParseError, EvaluationError, IntegrationError

# parse errors
try:
    expr = parse("x +")
except ParseError as e:
    print(f"syntax error: {e}")

# evaluation errors
try:
    expr = parse("x / 0")
    result = expr.evaluate(x=1.0)
except EvaluationError as e:
    print(f"division by zero: {e}")

# integration errors
try:
    expr = parse("x * sin(x)")
    integral = expr.integrate("x")
except IntegrationError as e:
    print(f"cannot integrate: {e}")
```

## performance tips

- simplify after differentiation: `expr.differentiate("x").simplify()`
- simplify before evaluation for complex expressions
- avoid deep nesting when possible
- use constants instead of repeated calculations

## limitations

expressions that are not supported:

```python
# implicit multiplication
parse("2x")  # error: use "2*x"

# multiple character variables
parse("xy")  # parses as variable "xy", not x*y

# equation solving
parse("x^2 = 4")  # not an expression

# symbolic exponents with base containing variable
expr = parse("x^x")
derivative = expr.differentiate("x")  # uses general formula

# integration by parts
parse("x * sin(x)").integrate("x")  # error: not supported

# definite integrals
# no built-in support for limits
```

## tips and tricks

### numerical differentiation comparison

```python
def compare_derivatives(expr_str, var, point):
    """compare symbolic vs numerical derivative."""
    expr = parse(expr_str)
    symbolic = expr.differentiate(var).simplify()
    
    h = 1e-7
    f_plus = expr.evaluate(**{var: point + h})
    f_minus = expr.evaluate(**{var: point - h})
    numerical = (f_plus - f_minus) / (2 * h)
    
    symbolic_val = symbolic.evaluate(**{var: point})
    
    print(f"symbolic: {symbolic_val}")
    print(f"numerical: {numerical}")
    print(f"error: {abs(symbolic_val - numerical)}")

compare_derivatives("x^3", "x", 2.0)
```

### latex rendering

```python
# generate latex for documentation
expr = parse("x^2 + sqrt(x) / (x + 1)")
latex = expr.to_latex()
print(f"$$\n{latex}\n$$")
```

### batch operations

```python
# differentiate multiple times
expr = parse("x^4")
for i in range(4):
    expr = expr.differentiate("x").simplify()
    print(f"derivative {i+1}: {expr}")
```

## debugging

enable expression inspection:

```python
expr = parse("x^2 + 1")

# view internal representation
print(repr(expr))  # shows node types

# check for variable
print(expr.contains_var("x"))  # True
print(expr.contains_var("y"))  # False

# step-by-step simplification
expr = parse("(x + 0) * 1")
print(f"original: {expr}")
print(f"simplified: {expr.simplify()}")
```

## examples directory

see the examples directory for jupyter notebooks:

- `differentiation_demo.ipynb` - differentiation techniques
- `integration_demo.ipynb` - integration examples
- `simplification_demo.ipynb` - simplification rules

## further reading

- README.md - comprehensive api documentation
- demo.py - complete feature demonstration
- tests/ - example usage patterns
