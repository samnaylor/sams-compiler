import sys

from enum import Enum
from string import digits, ascii_letters
from typing import cast, Literal, Generator
from os.path import relpath
from dataclasses import field, dataclass

from llvmlite import ir, binding as llvm


# region Constants

RED = "\u001b[31;1m"
CYAN = "\u001b[36;1m"
GREEN = "\u001b[32;1m"
MAGENTA = "\u001b[35;1m"
RESET = "\u001b[0m"

# endregion


# region Lexer

@dataclass(slots=True, frozen=True, repr=False)
class Location:
    line: int
    column: int

    def __repr__(self) -> str:
        return f"({self.line}, {self.column})"


def generic_error(errtype: str, filename: str, location: Location, message: str) -> None:
    print(f"{RED}{errtype}{RESET} {CYAN}{filename}:{location.line}:{location.column}{RESET}")
    print(f"{RED}{message}{RESET}")
    sys.exit(1)


class TokenKind(Enum):
    IntLiteral = 0
    Identifier = 1

    Fun = 10
    Var = 11
    While = 12
    Return = 13

    Gt = 20
    Lt = 21
    Plus = 22
    Minus = 23
    Star = 24
    Arrow = 25
    Colon = 26
    Comma = 27
    Equal = 28
    Question = 29

    Lpar = 30
    Rpar = 31
    Lsqu = 32
    Rsqu = 33

    Eof = 40
    Indent = 41
    Dedent = 42

    def __repr__(self) -> str:
        return self.name


@dataclass(slots=True, frozen=True)
class Token:
    kind: TokenKind
    value: str | None
    location: Location


def tokeniser_error(filename: str, location: Location, message: str) -> None:
    generic_error("Tokeniser Error", filename, location, message)


def tokeniser(source: str, filename: str) -> Generator[Token, None, None]:
    lvl, level, index, line, column = 0, 0, 0, 1, 1
    levels: list[int] = []

    keywords = {
        "fun": TokenKind.Fun,
        "var": TokenKind.Var,
        "while": TokenKind.While,
        "return": TokenKind.Return
    }

    symbols = {
        "+": TokenKind.Plus,
        "-": TokenKind.Minus,
        "*": TokenKind.Star,
        "=": TokenKind.Equal,
        ">": TokenKind.Gt,
        "<": TokenKind.Lt,
        "?": TokenKind.Question,
        ":": TokenKind.Colon,
        ",": TokenKind.Comma,
        "(": TokenKind.Lpar,
        ")": TokenKind.Rpar,
        "[": TokenKind.Lsqu,
        "]": TokenKind.Rsqu,

        "->": TokenKind.Arrow,
    }

    while index < len(source):
        if source[index] == "\n":
            while index < len(source) and source[index] == "\n":
                index += 1
                line += 1

            lvl = 0
            column = 1

            index -= 1

            while (index := index + 1) < len(source) and source[index] == " ":
                lvl += 1
                column += 1

            if lvl > level:
                yield Token(TokenKind.Indent, None, Location(line, column))
                levels.append(level := lvl)

            while lvl < level:
                yield Token(TokenKind.Dedent, None, Location(line, column))
                levels.pop(-1)

                try:
                    level = levels[-1]
                except IndexError:
                    level = 0

                if level < lvl:
                    raise IndentationError(f"{filename}:{line}:{column}")

            continue

        if source[index] in ascii_letters:
            identifier = source[index]
            column += 1
            while ((index := index + 1) < len(source)) and (source[index] in (ascii_letters + "_" + digits)):
                identifier += source[index]
                column += 1

            yield Token(keywords.get(identifier, TokenKind.Identifier), identifier, Location(line, column - len(identifier)))

        elif source[index] in digits:
            number = source[index]
            column += 1
            while ((index := index + 1) < len(source)) and (source[index] in digits + "_"):
                number += source[index]
                column += 1

            yield Token(TokenKind.IntLiteral, number, Location(line, column - len(number)))

        elif source[index] in symbols.keys():
            symbol = source[index]
            column += 1
            while ((index := index + 1) < len(source)) and symbols.get((symbol + source[index])) is not None:
                symbol += source[index]
                column += 1

            yield Token(symbols[symbol], symbol, Location(line, column - len(symbol)))

        elif source[index] == " ":
            index += 1
            column += 1

        else:
            tokeniser_error(filename, Location(line, column), f"Unexpected character `{source[index]}`")

    yield Token(TokenKind.Eof, None, Location(line, column))

# endregion


# region AST

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

# endregion


# region Parser

def parser_error(filename: str, location: Location, message: str) -> None:
    generic_error("Parser Error", filename, location, message)


class Parser:
    def __init__(self, source: str, filename: str) -> None:
        self.filename = filename
        self.tokens = tokeniser(source, filename)
        self.top = next(self.tokens)

    def advance(self) -> None:
        self.top = next(self.tokens)

    def match(self, token_kind: TokenKind, *, value: str | None = None) -> bool:
        # TODO: Support matching of multiple token kinds at once
        if value is None:
            return self.top.kind == token_kind

        return (self.top.kind == token_kind) and (self.top.value == value)

    def expect(self, token_kind: TokenKind) -> None:
        if not self.match(token_kind):
            parser_error(self.filename, self.top.location, f"Unexpected `{self.top.value}`. Expecting token of kind {token_kind}")

        self.advance()

    def parse(self) -> Node:
        function_defs: list[FunctionDefinition] = []

        while not self.match(TokenKind.Eof):
            function_defs.append(self.parse_function_definition())

        return Program(function_defs)

    def parse_function_definition(self) -> FunctionDefinition:
        signature = self.parse_function_signature()
        self.expect(TokenKind.Colon)
        body = self.parse_statement()
        return FunctionDefinition(signature, body)

    def parse_function_signature(self) -> FunctionSignature:
        self.expect(TokenKind.Fun)
        name = self.top.value
        assert name is not None
        self.advance()
        self.expect(TokenKind.Lpar)
        parameters: list[FunctionParameter] = []

        if not self.match(TokenKind.Rpar):
            parameters = self.parse_function_parameters()

        self.expect(TokenKind.Rpar)
        self.expect(TokenKind.Arrow)

        retty = self.parse_type_identifier()

        return FunctionSignature(name,   parameters, retty)

    def parse_function_parameters(self) -> list[FunctionParameter]:
        parameters: list[FunctionParameter] = []

        while not self.match(TokenKind.Rpar):
            parameters.append(self.parse_function_parameter())
            if not self.match(TokenKind.Rpar):
                self.expect(TokenKind.Comma)

        return parameters

    def parse_function_parameter(self) -> FunctionParameter:
        name = self.top.value  # TODO: Do we need to match here... for extra safety
        assert name is not None
        self.advance()
        type = self.parse_type_identifier()
        return FunctionParameter(name, type)

    def parse_type_identifier(self) -> TypeIdentifier:
        if self.match(TokenKind.Identifier, value="i32"):
            self.advance()
            return TypeIdentifier("i32")
        elif self.match(TokenKind.Lsqu):
            self.advance()
            type = self.parse_type_identifier()
            assert not type.is_array, "2D arrays not yet supported!"
            self.expect(TokenKind.Comma)
            size = self.top.value
            assert size is not None and size.isdigit()
            self.advance()
            self.expect(TokenKind.Rsqu)
            return TypeIdentifier(type.typename, True, int(size))
        else:
            parser_error(self.filename, self.top.location, f"Unexpected `{self.top.value}` in parse_type_identifier")

    def parse_statement(self) -> Stmt:
        # TODO: The logic here doesn't quite fit. a block isn't just indented code. it must follow something. See Python lark or snail_lang parser
        if self.match(TokenKind.Indent):
            return self.parse_block_statement()
        elif self.match(TokenKind.Var):
            return self.parse_var_statement()
        elif self.match(TokenKind.While):
            return self.parse_while_statement()
        elif self.match(TokenKind.Return):
            return self.parse_return_statement()
        else:
            return self.parse_expr_statement()

    def parse_block_statement(self) -> Block:
        children: list[Stmt] = []
        self.expect(TokenKind.Indent)

        while not self.match(TokenKind.Dedent):
            children.append(self.parse_statement())

        self.expect(TokenKind.Dedent)
        return Block(children)

    def parse_var_statement(self) -> Var:
        self.expect(TokenKind.Var)
        declarators: list[Declarator] = []

        if not self.match(TokenKind.Identifier):
            parser_error(self.filename, self.top.location, "Var statement expects at least 1 declarator!")

        declarators.append(self.parse_declarator())

        while self.match(TokenKind.Comma):
            self.advance()
            declarators.append(self.parse_declarator())

        return Var(declarators)

    def parse_while_statement(self) -> While:
        self.expect(TokenKind.While)
        cond = self.parse_expression()
        self.expect(TokenKind.Colon)
        body = self.parse_statement()

        return While(cond, body)

    def parse_return_statement(self) -> Return:
        self.expect(TokenKind.Return)
        value = self.parse_expression()
        return Return(value)

    def parse_expr_statement(self) -> Expr_:
        expr = self.parse_expression()
        return Expr_(expr)

    def parse_declarator(self) -> Declarator:
        name = self.top.value
        assert name is not None
        self.advance()
        type = self.parse_type_identifier()
        self.expect(TokenKind.Equal)
        init = self.parse_initialiser()
        return Declarator(name, type, init)

    def parse_initialiser(self) -> Expr:
        if self.match(TokenKind.Lsqu):  # * Array initialiser
            self.advance()
            values: list[Expr] = []
            while not self.match(TokenKind.Rsqu):
                values.append(self.parse_expression())
                if not self.match(TokenKind.Rsqu):
                    self.expect(TokenKind.Comma)

            self.expect(TokenKind.Rsqu)

            return ArrayLiteral(values)
        else:
            return self.parse_expression()

    def parse_expression(self) -> Expr:
        return self.parse_assignment()

    def parse_assignment(self) -> Expr:
        loc = self.top.location
        lhs = self.parse_conditional()

        if not self.match(TokenKind.Equal):
            return lhs

        if not isinstance(lhs, (Variable, Index)):
            # TODO: This location will be wrong... needs to be the lhs location
            parser_error(self.filename, loc, f"lvalue must be assignable, {lhs.__class__.__name__} is not!")

        self.expect(TokenKind.Equal)
        value = self.parse_assignment()
        return Assignment(lhs, value)

    def parse_conditional(self) -> Expr:
        cond = self.parse_relational()

        if not self.match(TokenKind.Question):
            return cond

        self.expect(TokenKind.Question)
        then = self.parse_expression()
        self.expect(TokenKind.Colon)
        alt = self.parse_expression()

        return Selection(cond, then, alt)

    def parse_relational(self) -> Expr:
        lhs = self.parse_additive()

        if not (self.match(TokenKind.Gt) or self.match(TokenKind.Lt)):
            return lhs

        op = cast(Literal["<", ">"], self.top.value)
        self.advance()
        rhs = self.parse_additive()

        return ComparisonOp(op, lhs, rhs)

    def parse_additive(self) -> Expr:
        lhs = self.parse_multiplicative()

        while (self.match(TokenKind.Plus) or self.match(TokenKind.Minus)):
            op = cast(Literal["+", "-"], self.top.value)
            self.advance()
            rhs = self.parse_multiplicative()
            lhs = BinaryOp(op, lhs, rhs)

        return lhs

    def parse_multiplicative(self) -> Expr:
        lhs = self.parse_postfix()

        while (self.match(TokenKind.Star)):
            op = cast(Literal["*"], self.top.value)
            self.advance()
            rhs = self.parse_postfix()
            lhs = BinaryOp(op, lhs, rhs)

        return lhs

    def parse_postfix(self) -> Expr:
        lhs = self.parse_primary()

        if self.match(TokenKind.Lsqu):
            self.advance()
            index = self.parse_expression()
            self.expect(TokenKind.Rsqu)
            return Index(lhs, index)
        elif self.match(TokenKind.Lpar):
            self.advance()
            args: list[Expr] = []
            if not self.match(TokenKind.Rpar):
                args = self.parse_callargs()
            self.expect(TokenKind.Rpar)
            return Call(lhs, args)
        else:
            return lhs

    def parse_primary(self) -> Expr:
        node: Expr
        if self.match(TokenKind.Identifier):
            assert self.top.value is not None
            node = Variable(self.top.value)
            self.advance()
        elif self.match(TokenKind.IntLiteral):
            assert self.top.value is not None
            node = IntLiteral(int(self.top.value))
            self.advance()
        elif self.match(TokenKind.Lpar):
            self.advance()
            node = self.parse_expression()
            self.expect(TokenKind.Rpar)
        else:
            parser_error(self.filename, self.top.location, f"Unexpected {self.top.value}")

        return node

    # def parse_constant(self) -> Expr:
    #     raise NotImplementedError

    def parse_callargs(self) -> list[Expr]:
        args: list[Expr] = []

        while not self.match(TokenKind.Rpar):
            args.append(self.parse_expression())
            if not self.match(TokenKind.Rpar):
                self.expect(TokenKind.Comma)

        return args

# endregion


# region LLVM Generator

class LLVMGenerator:
    def __init__(self, filename: str):
        self.module = ir.Module()
        self.module.name = filename

        self.builder: ir.IRBuilder | None = None

        self.locals: dict[str, ir.Value] = {}
        self.functions: dict[str, ir.Function] = {}

    def _optimise(self) -> llvm.ModuleRef:
        module = llvm.parse_assembly(str(self.module))
        module.name = module.name
        module.triple = llvm.Target.from_default_triple().triple

        pmb = llvm.create_pass_manager_builder()
        pm = llvm.create_module_pass_manager()

        pmb.populate(pm)
        pmb.opt_level = 3

        pm.add_instruction_combining_pass()
        pm.add_dead_arg_elimination_pass()
        pm.add_dead_code_elimination_pass()
        pm.add_function_inlining_pass(1_000)
        pm.add_cfg_simplification_pass()
        pm.add_constant_merge_pass()

        pm.run(module)

        module.verify()

        return module

    def __str__(self) -> str:
        return str(self._optimise())

    def generate_default(self, node: Node, *, flag: int = 0) -> None:
        raise NotImplementedError(node.__class__.__name__)

    def generate(self, node: Node, *, flag: int = 0) -> ir.Value | None:
        return getattr(self, f"generate_{node.__class__.__name__}", self.generate_default)(node, flag=flag)

    def generate_Program(self, node: Program, *, flag: int = 0) -> None:
        for fdef in node.function_defs:
            self.generate(fdef)

    def generate_FunctionDefinition(self, node: FunctionDefinition, *, flag: int = 0) -> None:
        self.locals = {}

        func = cast(ir.Function, self.generate(node.function_signature))
        self.builder = ir.IRBuilder(func.append_basic_block("entry"))

        for arg in func.args:
            alloca = self.builder.alloca(arg.type, name=arg.name)
            self.builder.store(arg, alloca)
            self.locals[arg.name] = alloca

        self.generate(node.function_body)
        self.builder = None

    def generate_FunctionSignature(self, node: FunctionSignature, *, flag: int = 0) -> ir.Value:
        retty = self.generate(node.function_retty)
        types = [self.generate(param.parameter_type) for param in node.function_params]

        signature = ir.FunctionType(retty, types)
        function = ir.Function(self.module, signature, node.function_name)

        for (arg, param) in zip(function.args, node.function_params):
            arg.name = param.parameter_name

        self.functions[node.function_name] = function

        return function

    def generate_TypeIdentifier(self, node: TypeIdentifier, *, flag: int = 0) -> ir.Type:
        base: ir.Type

        if node.typename == "i32":
            base = ir.IntType(32)

        if node.is_array:
            base = ir.ArrayType(base, node.array_sz)

        return base

    def generate_Block(self, node: Block, *, flag: int = 0) -> None:
        for stmt in node.body:
            self.generate(stmt)

    def generate_Return(self, node: Return, *, flag: int = 0) -> None:
        assert self.builder is not None

        self.builder.ret(self.generate(node.return_value, flag=1))

    def generate_While(self, node: While, *, flag: int = 0) -> None:
        assert self.builder is not None

        cond_block = self.builder.append_basic_block("while.cond")
        body_block = self.builder.append_basic_block("while.body")
        post_block = self.builder.append_basic_block("while.post")

        self.builder.branch(cond_block)
        self.builder.position_at_start(cond_block)
        cond = self.generate(node.condition, flag=1)
        self.builder.cbranch(cond, body_block, post_block)
        self.builder.position_at_start(body_block)
        self.generate(node.body)
        self.builder.branch(cond_block)
        self.builder.position_at_start(post_block)

        # TODO: this errors if we return from within the loop

    def generate_Var(self, node: Var, *, flag: int = 0) -> None:
        for decl in node.declarators:
            self.generate(decl)

    def generate_Declarator(self, node: Declarator, *, flag: int = 0) -> None:
        assert self.builder is not None

        typ = self.generate(node.type_id)
        alloca = self.builder.alloca(typ, name=node.identifier)
        value = self.generate(node.initialiser, flag=1)
        self.builder.store(value, alloca)
        self.locals[node.identifier] = alloca

    def generate_Expr_(self, node: Expr_, *, flag: int = 0) -> None:
        self.generate(node.expression)

    def generate_BinaryOp(self, node: BinaryOp, *, flag: int = 0) -> None:
        assert self.builder is not None

        lhs = self.generate(node.lhs, flag=1)
        rhs = self.generate(node.rhs, flag=1)

        match node.op:
            case "+": return self.builder.add(lhs, rhs)
            case "-": return self.builder.sub(lhs, rhs)
            case "*": return self.builder.mul(lhs, rhs)
            case _: raise ArithmeticError(node.op)

    def generate_Variable(self, node: Variable, *, flag: int = 0) -> ir.Value:
        assert self.builder is not None

        if (func := self.functions.get(node.var_name)) is not None:
            return func

        alloca = self.locals[node.var_name]

        if flag:
            return self.builder.load(alloca, name=node.var_name)

        return alloca

    def generate_Assignment(self, node: Assignment, *, flag: int = 0) -> ir.Value:
        assert self.builder is not None

        ptr = self.generate(node.target, flag=0)
        val = self.generate(node.value, flag=1)

        self.builder.store(val, ptr)

        return self.builder.load(ptr)

    def generate_Selection(self, node: Selection, *, flag: int = 0) -> ir.Value:
        assert self.builder is not None

        cond_block = self.builder.append_basic_block("if.cond")
        true_block = self.builder.append_basic_block("if.true")
        else_block = self.builder.append_basic_block("if.else")
        post_block = self.builder.append_basic_block("if.post")

        self.builder.branch(cond_block)
        self.builder.position_at_start(cond_block)
        con = self.generate(node.condition, flag=1)
        self.builder.cbranch(con, true_block, else_block)
        self.builder.position_at_start(true_block)
        lhs = self.generate(node.if_true, flag=1)
        self.builder.branch(post_block)
        self.builder.position_at_start(else_block)
        rhs = self.generate(node.if_alt, flag=1)
        self.builder.branch(post_block)
        self.builder.position_at_start(post_block)
        phi = self.builder.phi(lhs.type)
        phi.add_incoming(lhs, true_block)
        phi.add_incoming(rhs, else_block)

        return phi

    def generate_ComparisonOp(self, node: ComparisonOp, *, flag: int = 0) -> ir.Value:
        assert self.builder is not None

        lhs = self.generate(node.lhs, flag=1)
        rhs = self.generate(node.rhs, flag=1)

        return self.builder.icmp_signed(node.op, lhs, rhs)

    def generate_IntLiteral(self, node: IntLiteral, *, flag: int = 0) -> ir.Value:
        return ir.IntType(32)(node.int_value)

    def generate_Call(self, node: Call, *, flag: int = 0) -> ir.Value:
        assert self.builder is not None

        func = self.generate(node.callee, flag=1)
        args = [self.generate(arg, flag=1) for arg in node.callargs]

        return self.builder.call(func, args)

    def generate_ArrayLiteral(self, node: ArrayLiteral, *, flag: int = 0) -> ir.Value:
        return ir.Constant.literal_array([self.generate(value, flag=1) for value in node.arr_values])

    def generate_Index(self, node: Index, *, flag: int = 0) -> ir.Value:
        assert self.builder is not None

        target = self.generate(node.target, flag=0)
        index = self.generate(node.index, flag=1)

        value = self.builder.gep(target, (ir.IntType(32)(0), index))

        if flag == 1:
            return self.builder.load(value)

        return value

# endregion


def main() -> int:
    filename = relpath("examples/factorial.sam")

    with open(filename, "r") as f:
        source = f.read()

    tree = Parser(source, filename).parse()

    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_all_asmprinters()

    codegen = LLVMGenerator(filename)
    codegen.generate(tree)

    print(codegen)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
