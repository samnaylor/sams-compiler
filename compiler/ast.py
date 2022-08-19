from dataclasses import dataclass, field
from typing import Literal


class Node:
    " Base Node for all node types "

    # TODO: Location information inside of nodes


class Expr(Node):
    " Nodes that represent an expression (value construct) "


class Stmt(Node):
    " Nodes that represent a statement (non-value construct) "


@dataclass(slots=True, frozen=True)
class TypeIdentifier(Node):
    typename: str
    is_array: bool = field(default=False)
    array_sz: int = field(default=0)


@dataclass(slots=True, frozen=True)
class IntLiteral(Expr):
    int_value: int


@dataclass(slots=True, frozen=True)
class ArrayLiteral(Expr):  # NOTE: Array intialiser
    arr_values: list[Expr]


@dataclass(slots=True, frozen=True)
class Variable(Expr):
    var_name: str
    # ? Flag indicating whether to load the pointer ... or is that in the code generator


@dataclass(slots=True, frozen=True)
class Call(Expr):
    callee: Expr
    callargs: list[Expr]


@dataclass(slots=True, frozen=True)
class Index(Expr):
    target: Expr
    index: Expr  # ? Limitations on the value, integer or variable only


@dataclass(slots=True, frozen=True)
class BinaryOp(Expr):
    op: Literal["+", "-", "*"]
    lhs: Expr
    rhs: Expr


@dataclass(slots=True, frozen=True)
class ComparisonOp(Expr):  # ? Python-Style chained comparisons
    op: Literal[">", "<"]
    lhs: Expr
    rhs: Expr


@dataclass(slots=True, frozen=True)
class Selection(Expr):
    condition: Expr
    if_true: Expr
    if_alt: Expr


@dataclass(slots=True, frozen=True)
class Assignment(Expr):
    target: Expr
    value: Expr


@dataclass(slots=True, frozen=True)
class Declarator(Node):
    identifier: str
    type_id: TypeIdentifier
    initialiser: Expr


@dataclass(slots=True, frozen=True)
class Expr_(Stmt):
    expression: Expr


@dataclass(slots=True, frozen=True)
class Return(Stmt):
    return_value: Expr


@dataclass(slots=True, frozen=True)
class While(Stmt):
    condition: Expr
    body: Stmt


@dataclass(slots=True, frozen=True)
class Var(Stmt):
    declarators: list[Declarator]


@dataclass(slots=True, frozen=True)
class Block(Stmt):
    body: list[Stmt]


@dataclass(slots=True, frozen=True)
class FunctionParameter(Node):
    parameter_name: str
    parameter_type: TypeIdentifier


@dataclass(slots=True, frozen=True)
class FunctionSignature(Node):
    function_name: str
    function_params: list[FunctionParameter]
    function_retty: TypeIdentifier


@dataclass(slots=True, frozen=True)
class FunctionDefinition(Node):
    function_signature: FunctionSignature
    function_body: Stmt


@dataclass(slots=True, frozen=True)
class Program(Node):
    function_defs: list[FunctionDefinition]
