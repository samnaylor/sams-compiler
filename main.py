import sys

from os import remove
from abc import ABC, abstractmethod
from enum import auto, Enum
from string import digits, ascii_letters
from typing import Any, Generator, Literal, cast
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from subprocess import CalledProcessError, check_output
from dataclasses import dataclass, field

from llvmlite import ir
from llvmlite import binding as llvm


# region Utility

RED = "\u001b[31;1m"
CYAN = "\u001b[36;1m"
GREEN = "\u001b[32;1m"
MAGENTA = "\u001b[35;1m"
RESET = "\u001b[0m"


ErrType = Literal["LexerError", "ParserError", "CodeGenerationError", "CompilerError"]


def error(error_type: ErrType, message: str) -> None:
    print(f"{RED}ERROR:{RESET} {message}")
    sys.exit(1)

# endregion


# region Backend

i1 = ir.IntType(1)
i8 = ir.IntType(8)
i16 = ir.IntType(16)
i32 = ir.IntType(32)
i64 = ir.IntType(64)
void = ir.VoidType()
float_ = ir.FloatType()
double = ir.DoubleType()


class LLVMGeneratorContext:
    SYMBOLS_TO_TYPE = {
        "bool": i1,
        "i8": i8,
        "i16": i16,
        "i32": i32,
        "i64": i64,
        "void": void,
        "float_": float_,
        "double": double
    }

    TYPE_SYMBOLS = {
        i1: "b",
        i8: "c",
        i16: "s",
        i32: "i",
        i64: "l",
        void: "v",
        float_: "f",
        double: "d"
    }

    def __init__(self, name: str = "main") -> None:
        self._module = ir.Module(name=name)
        self._context = self._module.context

        self._module.triple = llvm.Target.from_default_triple().triple

        self._strcount: int = 0

        self._locals: dict[str, ir.Value] = {}
        self._externs: set[str] = set[str]()
        self._structs: dict[str, list[str]] = {}
        self._builders: list[ir.IRBuilder] = []
        self._functions: dict[str, ir.Function] = {}

        self.block_stack_start: list[ir.Block] = []
        self.block_stack_end: list[ir.Block] = []

    @property
    def builder(self) -> ir.IRBuilder:
        return self._builders[-1]

    @property
    def strcount(self) -> int:
        self._strcount += 1
        return self._strcount

    def new_builder(self, function: ir.Function) -> None:
        self._builders.append(ir.IRBuilder(function.append_basic_block()))

    def pop_builder(self) -> None:
        self._builders.pop()

    def get_local(self, name: str) -> ir.Value:
        return self._locals[name]

    def set_local(self, name: str, alloca: ir.Value) -> None:
        self._locals[name] = alloca

    def clear_locals(self) -> None:
        self._locals.clear()

    def get_function(self, mangled: str) -> ir.Function:
        return self._functions[mangled]

    def mangle_name(self, name: str, *args: ir.Type) -> str:
        argtyps: str = ""

        for arg in args:
            if arg.is_pointer:
                count = 0
                pointee = cast(ir.Type, cast(ir.PointerType, arg).pointee)
                while pointee.is_pointer:
                    pointee = cast(ir.Type, cast(ir.PointerType, pointee).pointee)
                    count += 1

                argtyps += f"{'P'*count}{self.TYPE_SYMBOLS[pointee]}"
            else:
                argtyps += self.TYPE_SYMBOLS[arg]

        return f"__{name}_{argtyps}"

    def __str__(self) -> str:
        return str(self._module)


class Node(ABC):
    @abstractmethod
    def generate(self, context: LLVMGeneratorContext) -> Any:
        raise NotImplementedError


class Expression(Node, ABC):
    @abstractmethod
    def generate(self, context: LLVMGeneratorContext, *, as_pointer: bool = False) -> ir.Value:
        raise NotImplementedError


class Statement(Node, ABC):
    @abstractmethod
    def generate(self, context: LLVMGeneratorContext) -> None:
        raise NotImplementedError


@dataclass(slots=True)
class TypeID(Node):
    typename: Literal["bool", "i8", "i16", "i32", "i64", "void", "float", "double"] | str
    pointer: int = field(default=0, kw_only=True)

    def generate(self, context: LLVMGeneratorContext) -> ir.Type:
        typ = context.SYMBOLS_TO_TYPE[self.typename]

        for _ in range(self.pointer):
            typ = ir.PointerType(typ)

        return typ


@dataclass(slots=True)
class FuncDef(Node):
    function_name: str
    function_rety: TypeID
    function_args: list[tuple[str, TypeID]]
    function_body: list[Statement]

    mangle: bool = field(default=True, kw_only=True)

    def generate(self, context: LLVMGeneratorContext) -> ir.Function:
        rety = self.function_rety.generate(context)
        names: list[str] = []
        fargs: list[ir.Type] = []

        for (argname, argtype) in self.function_args:
            names.append(argname)
            fargs.append(argtype.generate(context))

        ftyp = ir.FunctionType(rety, fargs, var_arg=False)

        name = context.mangle_name(self.function_name, *ftyp.args) if self.mangle else self.function_name
        context._functions[name] = func = ir.Function(context._module, ftyp, name)

        context.new_builder(func)
        context.clear_locals()

        for (farg, name) in zip(func.args, names):
            farg.name = name
            alloca = context.builder.alloca(farg.type, name=name)
            context.builder.store(farg, alloca)
            context.set_local(name, alloca)

        for statement in self.function_body:
            statement.generate(context)

        if not context.builder.block.is_terminated:
            context.builder.ret_void()

        context.pop_builder()

        return func


@dataclass(slots=True)
class ExternDef(Node):
    extern_name: str
    extern_args: list[TypeID]
    extern_rety: TypeID
    extern_vararg: bool = field(default=False, kw_only=True)

    def generate(self, context: LLVMGeneratorContext) -> Any:
        rety = self.extern_rety.generate(context)
        args = [arg.generate(context) for arg in self.extern_args]
        ftyp = ir.FunctionType(rety, args, var_arg=self.extern_vararg)
        func = ir.Function(context._module, ftyp, self.extern_name)

        context._externs.add(self.extern_name)
        context._functions[self.extern_name] = func


@dataclass(slots=True)
class StructDef(Node):
    struct_name: str
    struct_fields: list[tuple[str, TypeID]]
    create_constructor: bool = field(default=True, kw_only=True)

    def generate(self, context: LLVMGeneratorContext) -> Any:
        struct = context._context.get_identified_type(self.struct_name)

        names: list[str] = []
        types: list[ir.Type] = []

        for (name, ftyp) in self.struct_fields:
            names.append(name)
            types.append(ftyp.generate(context))

        struct.set_body(*types)
        context._structs[self.struct_name] = names

        if self.create_constructor:
            # Automatically generated constructor function

            constructor = ir.Function(context._module, ir.FunctionType(struct, types, var_arg=False), self.struct_name)
            builder = ir.IRBuilder(constructor.append_basic_block())

            new_struct = builder.alloca(struct)

            for idx, (arg, name) in enumerate(zip(constructor.args, names)):
                arg.name = name
                ptr = builder.gep(new_struct, (i32(0), i32(idx)), inbounds=True)
                builder.store(arg, ptr)

            builder.ret(builder.load(new_struct))
            context._functions[context.mangle_name(self.struct_name, *types)] = constructor

        context.SYMBOLS_TO_TYPE[self.struct_name] = struct
        context.TYPE_SYMBOLS[struct] = self.struct_name


@dataclass(slots=True)
class Declarator(Node):
    decl_name: str
    decl_type: TypeID
    decl_init: Expression

    def generate(self, context: LLVMGeneratorContext) -> Any:
        typ = self.decl_type.generate(context)
        val = self.decl_init.generate(context)

        alloca = context.builder.alloca(typ, name=self.decl_name)
        context.builder.store(val, alloca)
        context.set_local(self.decl_name, alloca)


@dataclass(slots=True)
class VarDecl(Statement):
    declarators: list[Declarator]

    def generate(self, context: LLVMGeneratorContext) -> None:
        for decl in self.declarators:
            decl.generate(context)


@dataclass(slots=True)
class IfElse(Statement):
    if_cond: Expression
    if_then: list[Statement]
    if_else: list[Statement] | None

    def generate(self, context: LLVMGeneratorContext) -> None:
        cond = self.if_cond.generate(context)

        if self.if_else is not None:
            with context.builder.if_else(cond) as (tb, fb):
                with tb:
                    for stmt in self.if_then:
                        stmt.generate(context)
                with fb:
                    for stmt in self.if_else:
                        stmt.generate(context)
        else:
            with context.builder.if_then(cond):
                for stmt in self.if_then:
                    stmt.generate(context)


@dataclass(slots=True)
class While(Statement):
    while_cond: Expression
    while_body: list[Statement]

    def generate(self, context: LLVMGeneratorContext) -> None:
        condblock = context.builder.append_basic_block("while.cond")
        bodyblock = context.builder.append_basic_block("while.body")
        exitblock = context.builder.append_basic_block("while.exit")

        context.block_stack_start.append(condblock)
        context.block_stack_end.append(exitblock)

        context.builder.branch(condblock)
        context.builder.position_at_start(condblock)
        cond = self.while_cond.generate(context)
        context.builder.cbranch(cond, bodyblock, exitblock)
        context.builder.position_at_start(bodyblock)

        for statement in self.while_body:
            statement.generate(context)

        context.builder.branch(condblock)
        context.builder.position_at_start(exitblock)

        context.block_stack_start.pop()
        context.block_stack_end.pop()


@dataclass(slots=True)
class Break(Statement):
    def generate(self, context: LLVMGeneratorContext) -> None:
        context.builder.branch(context.block_stack_end[-1])


@dataclass(slots=True)
class Continue(Statement):
    def generate(self, context: LLVMGeneratorContext) -> None:
        context.builder.branch(context.block_stack_start[-1])


@dataclass(slots=True)
class Return(Statement):
    return_value: Expression | None

    def generate(self, context: LLVMGeneratorContext) -> None:
        if self.return_value is not None:
            context.builder.ret(self.return_value.generate(context))
        else:
            context.builder.ret_void()


@dataclass(slots=True)
class Expr(Statement):
    expression: Expression

    def generate(self, context: LLVMGeneratorContext) -> None:
        self.expression.generate(context)


@dataclass(slots=True)
class TypeCast(Expression):
    to_type: TypeID
    value: Expression

    def generate(self, context: LLVMGeneratorContext, *, as_pointer: bool = False) -> ir.Value:
        to_typ = self.to_type.generate(context)
        value = self.value.generate(context)

        if to_typ.is_pointer and value.type.is_pointer:
            return context.builder.bitcast(value, to_typ)

        match (str(to_typ), str(value.type)):
            case "i8", "i32":
                return context.builder.trunc(value, to_typ)

            case "double", "i32":
                return context.builder.sitofp(value, to_typ)

            case "i32", "double":
                return context.builder.fptosi(value, to_typ)


@dataclass(slots=True)
class Assign(Expression):
    lvalue: Expression
    rvalue: Expression

    def generate(self, context: LLVMGeneratorContext, *, as_pointer: bool = False) -> ir.Value:
        lvalue = self.lvalue.generate(context, as_pointer=True)
        rvalue = self.rvalue.generate(context, as_pointer=False)

        context.builder.store(rvalue, lvalue)
        return context.builder.load(lvalue)


@dataclass(slots=True)
class Conditional(Expression):
    cond: Expression
    lhs: Expression
    rhs: Expression

    def generate(self, context: LLVMGeneratorContext, *, as_pointer: bool = False) -> ir.Value:
        condblock = context.builder.append_basic_block("conditional.cond")
        trueblock = context.builder.append_basic_block("conditional.true")
        elseblock = context.builder.append_basic_block("conditional.else")
        postblock = context.builder.append_basic_block("conditional.post")

        context.builder.branch(condblock)
        context.builder.position_at_start(condblock)
        cond = self.cond.generate(context)
        context.builder.cbranch(cond, trueblock, elseblock)
        context.builder.position_at_start(trueblock)
        lhs = self.lhs.generate(context)
        context.builder.branch(postblock)
        context.builder.position_at_start(elseblock)
        rhs = self.rhs.generate(context)
        context.builder.branch(postblock)
        context.builder.position_at_start(postblock)
        phi = context.builder.phi(lhs.type, name="conditional.phi.result")
        phi.add_incoming(lhs, trueblock)
        phi.add_incoming(rhs, elseblock)

        return phi


@dataclass(slots=True)
class FuncCall(Expression):
    function_name: str
    function_args: list[Expression]

    def generate(self, context: LLVMGeneratorContext, *, as_pointer: bool = False) -> ir.Value:
        args = [arg.generate(context, as_pointer=False) for arg in self.function_args]

        if self.function_name in context._externs:
            func = context.get_function(self.function_name)
        else:
            name = context.mangle_name(self.function_name, *[arg.type for arg in args])
            func = context.get_function(name)

        return context.builder.call(func, args, name=f"call.{self.function_name}")


@dataclass(slots=True)
class BinaryOp(Expression):
    op: Literal["+", "*", "-", "/", "%"]
    lhs: Expression
    rhs: Expression

    def generate(self, context: LLVMGeneratorContext, *, as_pointer: bool = False) -> ir.Value:
        lhs = self.lhs.generate(context, as_pointer=False)
        rhs = self.rhs.generate(context, as_pointer=False)

        match self.op:
            case "+":
                return context.builder.add(lhs, rhs)

            case "-":
                return context.builder.sub(lhs, rhs)

            case "*":
                return context.builder.mul(lhs, rhs)

            case _:
                raise NotImplementedError(f"BinaryOp({self.op})")


@dataclass(slots=True)
class UnaryOp(Expression):
    op: Literal["-", "&", "*", "not"]
    rhs: Expression

    def generate(self, context: LLVMGeneratorContext, *, as_pointer: bool = False) -> ir.Value:
        match self.op:
            case "-":
                return context.builder.neg(self.rhs.generate(context))

            case "&":
                return self.rhs.generate(context, as_pointer=True)

            case "*":  # NOTE: This seems hacky... but it fixes the problem of derefencing not working on pointer arguments
                rhs = self.rhs.generate(context)
                if rhs.type.is_pointer and not as_pointer:
                    return context.builder.load(rhs)
                return rhs

            case _:
                error("CodeGenerationError", f"UnaryOp({self.op}) not implemented")


@dataclass(slots=True)
class ComparisonOp(Expression):
    op: Literal["==", "!=", ">", "<", ">=", "<="]
    lhs: Expression
    rhs: Expression

    def generate(self, context: LLVMGeneratorContext, *, as_pointer: bool = False) -> ir.Value:
        lhs = self.lhs.generate(context)
        rhs = self.rhs.generate(context)

        return context.builder.icmp_signed(self.op, lhs, rhs)


@dataclass(slots=True)
class LogicalOp(Expression):
    op: Literal["and", "or", "xor"]
    lhs: Expression
    rhs: Expression

    def generate(self, context: LLVMGeneratorContext, *, as_pointer: bool = False) -> ir.Value:
        error("CodeGenerationError", f"LogicalOp({self.op}) not implemented")


@dataclass(slots=True)
class BitwiseOp(Expression):
    op: Literal["<<", ">>", "&", "|", "^"]
    lhs: Expression
    rhs: Expression

    def generate(self, context: LLVMGeneratorContext, *, as_pointer: bool = False) -> ir.Value:
        lhs = self.lhs.generate(context)
        rhs = self.rhs.generate(context)

        match self.op:
            case ">>":
                return context.builder.ashr(lhs, rhs)

            case "<<":
                return context.builder.shl(lhs, rhs)

            case "|":
                return context.builder.or_(lhs, rhs)

            case _:
                error("CodeGenerationError", f"BitwiseOp({self.op}) not implemented")


@dataclass(slots=True)
class MemberAccess(Expression):
    lvalue: Expression
    member: str
    accessor: Literal["->", "."]

    def generate(self, context: LLVMGeneratorContext, *, as_pointer: bool = False) -> ir.Value:
        match self.accessor:
            case ".":
                lvalue = self.lvalue.generate(context, as_pointer=True)

            case "->":
                lvalue = self.lvalue.generate(context, as_pointer=False)

        typ = lvalue.type

        while typ.is_pointer:
            typ = typ.pointee

        struct_name = typ.name
        struct = context._context.get_identified_type(struct_name)
        field = i32(context._structs[struct_name].index(self.member))
        ptr = context.builder.gep(lvalue, (i32(0), field), inbounds=True, name=f"{struct_name}.{self.member}")

        if as_pointer:
            return ptr

        return context.builder.load(ptr, name=f"{struct.name}.{self.member}")


@dataclass(slots=True)
class Index(Expression):
    target: Expression
    index: Expression

    def generate(self, context: LLVMGeneratorContext, *, as_pointer: bool = False) -> ir.Value:
        target = self.target.generate(context, as_pointer=False)
        index = self.index.generate(context)
        ptr = context.builder.gep(target, (index,), inbounds=True)

        if as_pointer:
            return ptr

        return context.builder.load(ptr)


@dataclass(slots=True)
class Variable(Expression):
    var_name: str

    def generate(self, context: LLVMGeneratorContext, *, as_pointer: bool = False) -> ir.Value:
        ptr = context.get_local(self.var_name)

        if not as_pointer:
            ptr = context.builder.load(ptr, name=self.var_name)

        return ptr


@dataclass(slots=True)
class IntLiteral(Expression):
    int_value: int

    def generate(self, context: LLVMGeneratorContext, *, as_pointer: bool = False) -> ir.Constant:
        return ir.Constant(i32, self.int_value)


@dataclass(slots=True)
class CharLiteral(Expression):
    char_value: int

    def __post_init__(self) -> None:
        self.char_value &= 0xFF

    def generate(self, context: LLVMGeneratorContext, *, as_pointer: bool = False) -> ir.Value:
        return ir.Constant(i8, self.char_value)


@dataclass(slots=True)
class BoolLiteral(Expression):
    bool_value: bool

    def generate(self, context: LLVMGeneratorContext, *, as_pointer: bool = False) -> ir.Value:
        return ir.Constant(i1, self.bool_value)


@dataclass(slots=True)
class StringLiteral(Expression):
    string_value: str

    def generate(self, context: LLVMGeneratorContext, *, as_pointer: bool = False) -> ir.Value:
        typ = ir.ArrayType(i8, len(self.string_value) + 1)
        global_ = ir.GlobalVariable(context._module, typ, f".str{context.strcount}")
        global_.initializer = ir.Constant(typ, bytearray((self.string_value + "\0").encode()))
        global_.global_constant = True

        return context.builder.gep(global_, (i32(0), i32(0)), inbounds=True)

# endregion


# region Frontend


@dataclass(slots=True, repr=False)
class Location:
    line: int
    column: int

    def __repr__(self) -> str:
        return f"{self.line:03d}, {self.column:03d}"


class TokenKind(Enum):
    IntLiteral = 0x00
    StrLiteral = auto()
    ChrLiteral = auto()
    Identifier = auto()

    KwdIf = 0x10                # if
    KwdOr = auto()              # or
    KwdAnd = auto()             # and
    KwdFun = auto()             # fun
    KwdNot = auto()             # not
    KwdXor = auto()             # xor
    KwdVar = auto()             # var
    KwdElse = auto()            # else
    KwdBreak = auto()           # break
    KwdWhile = auto()           # while
    KwdExtern = auto()          # extern
    KwdMethod = auto()          # method
    KwdReturn = auto()          # return
    KwdStruct = auto()          # struct
    KwdContinue = auto()        # continue
    KwdConstructor = auto()     # constructor

    Plus = 0x30                 # +
    Star = auto()               # *
    Minus = auto()              # -
    Slash = auto()              # /
    Percent = auto()            # %
    Shl = auto()                # <<
    Shr = auto()                # >>
    And = auto()                # &
    Or = auto()                 # |
    Xor = auto()                # ^
    Eq = auto()                 # ==
    Ne = auto()                 # !=
    Gt = auto()                 # >
    Lt = auto()                 # <
    Le = auto()                 # >=
    Ge = auto()                 # <=

    Dot = 0x50                  # .
    Lpar = auto()               # (
    Lsqu = auto()               # [
    Rpar = auto()               # )
    Rsqu = auto()               # ]
    Arrow = auto()              # ->
    Colon = auto()              # :
    Comma = auto()              # ,
    Equal = auto()              # =
    Ellipsis = auto()           # ...
    Question = auto()           # ?

    Indent = 0xFC
    Dedent = auto()
    Eof = auto()
    DoesNotExist = auto()       # .. NOTE: added because of the way of finding compound symbols is silly

    def __repr__(self) -> str:
        return self.name


@dataclass(slots=True, repr=False)
class Token:
    kind: TokenKind
    value: str
    location: Location

    def __str__(self) -> str:
        stub = f"Token(location=({self.location}), kind={self.kind.name}"
        if self.value:
            stub += f", value='{self.value}'"

        return f"{stub})"


KEYWORDS = {
    "if": TokenKind.KwdIf,
    "or": TokenKind.KwdOr,
    "and": TokenKind.KwdAnd,
    "fun": TokenKind.KwdFun,
    "not": TokenKind.KwdNot,
    "xor": TokenKind.KwdXor,
    "var": TokenKind.KwdVar,
    "else": TokenKind.KwdElse,
    "break": TokenKind.KwdBreak,
    "while": TokenKind.KwdWhile,
    "extern": TokenKind.KwdExtern,
    "method": TokenKind.KwdMethod,
    "return": TokenKind.KwdReturn,
    "struct": TokenKind.KwdStruct,
    "continue": TokenKind.KwdContinue,
    "constructor": TokenKind.KwdConstructor,
}

SYMBOLS = {
    "+": TokenKind.Plus,
    "*": TokenKind.Star,
    "-": TokenKind.Minus,
    "/": TokenKind.Slash,
    "%": TokenKind.Percent,
    "<<": TokenKind.Shl,
    ">>": TokenKind.Shr,
    "&": TokenKind.And,
    "|": TokenKind.Or,
    "^": TokenKind.Xor,
    "==": TokenKind.Eq,
    "!=": TokenKind.Ne,
    ">": TokenKind.Gt,
    "<": TokenKind.Lt,
    ">=": TokenKind.Le,
    "<=": TokenKind.Ge,
    ".": TokenKind.Dot,
    "(": TokenKind.Lpar,
    "[": TokenKind.Lsqu,
    ")": TokenKind.Rpar,
    "]": TokenKind.Rsqu,
    "->": TokenKind.Arrow,
    ":": TokenKind.Colon,
    ",": TokenKind.Comma,
    "=": TokenKind.Equal,
    "...": TokenKind.Ellipsis,
    "?": TokenKind.Question,

    "..": TokenKind.DoesNotExist
}


def tokenise(filename: str) -> Generator[Token, None, None]:
    with open(filename, "r") as f:
        source = f.read()

    lvl, level, index, line, column = 0, 0, 0, 1, 1
    levels: list[int] = []

    while index < len(source):
        if source[index] == "\n":
            while (index < len(source)) and (source[index] == "\n"):
                index += 1
                line += 1

            lvl = 0
            column = 1
            index -= 1

            while ((index := index + 1) < len(source)) and (source[index] == " "):
                lvl += 1
                column += 1

            if lvl > level:
                levels.append(level := lvl)
                yield Token(TokenKind.Indent, "", Location(line, column))

            while lvl < level:
                levels.pop(-1)
                yield Token(TokenKind.Dedent, "", Location(line, column))

                level = levels[-1] if len(levels) else 0

                if level < lvl:
                    raise IndentationError(f"{filename}:{line}:{column}")

        if index >= len(source):
            break

        if source[index] == "#":
            while (index < len(source)) and (source[index] != "\n"):
                index += 1

        elif source[index] in ascii_letters:
            identifier = source[index]
            column += 1

            while ((index := index + 1) < len(source)) and (source[index] in (ascii_letters + "_" + digits)):
                identifier += source[index]
                column += 1

            yield Token(KEYWORDS.get(identifier, TokenKind.Identifier), identifier, Location(line, column - len(identifier)))

        elif source[index] in digits:
            number = source[index]
            column += 1

            while ((index := index + 1) < len(source)) and (source[index] in digits + "_" + "."):
                number += source[index]
                column += 1

            if (count := number.count(".")) > 1:
                error("LexerError", f"{filename}:{line}:{column - len(number)} - Floating point number can only have 1 decimal, place found {count}")

            # TODO: Support floats
            assert count == 0, f"{filename}:{line}:{column - len(number)} - Float literals not yet supported"

            yield Token(
                TokenKind.IntLiteral,
                number.replace("_", ""),
                Location(line, column - len(number))
            )

        elif source[index] in SYMBOLS.keys():
            symbol = source[index]
            column += 1

            while ((index := index + 1) < len(source)) and SYMBOLS.get((symbol + source[index])) is not None:
                symbol += source[index]
                column += 1

            yield Token(SYMBOLS[symbol], symbol, Location(line, column - len(symbol)))

        elif source[index] == '"':
            # TODO: character escaping
            string = source[index]
            column += 1

            while ((index := index + 1) < len(source)) and source[index] != '"':
                string += source[index]
                column += 1

            if (index >= len(source)) or (source[index] != '"'):
                error("LexerError", f"{filename}:{line}:{column - len(number)} - String Literal is not closed")

            string += source[index]
            column += 1
            index += 1

            yield Token(TokenKind.StrLiteral, string, Location(line, column - len(symbol)))

        elif source[index] == "'":
            # TODO: character escaping
            index += 1
            column += 1

            character = source[index]
            index += 1
            column += 1

            if source[index] != "'":
                error("LexerError", f"{filename}:{line}:{column - 2} - Char Literals can only contain 1 character")

            index += 1
            column += 1

            yield Token(TokenKind.ChrLiteral, character, Location(line, column - 3))

        elif source[index] == " ":
            index += 1
            column += 1

        else:
            error("LexerError", f"{filename}:{line}:{column} - Unexpected character {source[index]}")

    yield Token(TokenKind.Eof, "", Location(line, column))


class Parser:
    """
    program  ::= toplevel*
    toplevel ::= extern_definition
               | struct_definition
               | function_definition

    extern_definition   ::= "extern" identifier "(" type_id_list ")" "->" type_id
    struct_definition   ::= "struct" identifier ":" INDENT field+ [constructor] method* DEDENT
    function_definition ::= "fun" identifier "(" [parameter_list] ")" "->" type_id ":" suite
    constructor         ::= "constructor" identifier "(" [parameter_list] ")" "->" type_id ":" suite
    method              ::= "method" identifier "(" [parameter_list] ")" "->" type_id ":" suite

    type_id        ::= "i8"
                     | "i32"
                     | "void"
                     | identifier
                     | type_id ("*")*
    type_id_list   ::= type_id ("," (type_id | "..."))*
    field          ::= identifier type_id
    suite          ::= statement
                     | INDENT statement+ DEDENT
    parameter_list ::= field ("," field)*

    statement ::= declaration
                | while
                | if-else
                | return
                | break
                | continue
                | expr

    declaration ::= "var" declarator ("," declarator)*
    while       ::= "while" expression ":" suite
    if-else     ::= "if" expresion ":" suite ["else" ":" suite]
    return      ::= "return" [expression]
    break       ::= "break"
    continue    ::= "continue
    expr        ::= expression

    declarator ::= identifier type_identifier "=" expression

    expression     ::= assignment
    assignment     ::= conditional
                     | postfix "=" assignment
    conditional    ::= logicalor
                     | logicalor "?" expression ":" conditional
    logicalor      ::= logicalxor
                     | logicalor ("or" logicalxor)*
    logicalxor     ::= logicaland
                     | logicalxor ("xor" logicaland)*
    logicaland     ::= logicalnot
                     | logicaland ("and" logicalnot)*
    logicalnot     ::= relational
                     | "not" logicalnot
    relational     ::= bitwiseor
                     | relational ">" bitwiseor
                     | relational "<" bitwiseor
                     | relational "==" bitwiseor
                     | relational "!=" bitwiseor
                     | relational ">=" bitwiseor
                     | relational "<=" bitwiseor
    bitwiseor      ::= bitwisexor
                     | bitwiseor ("|" bitwisexor)*
    bitwisexor     ::= bitwiseand
                     | bitwisexor ("^" bitwiseand)*
    bitwiseand     ::= shift
                     | bitwiseand ("&" bitwisenot)*
    shift          ::= additive
                     | shift (">>" additive)*
                     | shift ("<<" additive)*
    additive       ::= multiplicative
                     | additive ("+" multiplicative)*
                     | additive ("-" multiplicative)*
    multiplicative ::= cast
                     | multiplicative ("*" cast)*
    cast           ::= unary
                     | "<" type_id ">" cast
    unary          ::= postfix
                     | "-" unary
                     | "&" unary
                     | "*" unary
    postfix        ::= primary
                     | postfix "[" expression "]"
                     | postfix "(" [callargs] ")"
                     | postfix "." identifier
                     | postfix "->" identifier
    primary        ::= identifier
                     | constant
                     | "(" expression ")"
    constant       ::= INTEGER | CHAR | STRING

    callargs ::= expression ("," expression)*
    """

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self._tokens = tokenise(filename)
        self._current = next(self._tokens)

        self._struct_name = ""
        self._loop_stack: list[str] = []

    @property
    def top(self) -> Token:
        return self._current

    def match(self, *kinds: TokenKind) -> bool:
        return self.top.kind in kinds

    def expect(self, kind: TokenKind) -> None:
        if not self.match(kind):
            error("ParserError", f"{self.filename}:{self.top.location.line}:{self.top.location.column} - Expected {kind}, got {self.top}")

        self.advance()

    def advance_and_extract(self) -> str:
        value = self.top.value
        self.advance()
        return value

    def expect_and_extract(self, kind: TokenKind) -> str:
        if not self.match(kind):
            error("ParserError", f"{self.filename}:{self.top.location.line}:{self.top.location.column} - Expected {kind}, got {self.top}")

        value = self.top.value
        self.advance()
        return value

    def advance(self) -> Token:
        self._current = next(self._tokens)

    def parse(self) -> list[Node]:
        program: list[Node] = []

        while not self.match(TokenKind.Eof):
            program.extend(self.parse_toplevel())

        return program

    def parse_toplevel(self) -> list[Node]:
        match self.top.kind:
            case TokenKind.KwdExtern:
                return [self.parse_extern()]

            case TokenKind.KwdStruct:
                return self.parse_struct()

            case TokenKind.KwdFun:
                return [self.parse_function()]

    def parse_extern(self) -> ExternDef:
        self.advance()
        identifier = self.expect_and_extract(TokenKind.Identifier)
        self.expect(TokenKind.Lpar)
        type_ids, is_vararg = self.parse_type_id_list()
        self.expect(TokenKind.Rpar)
        self.expect(TokenKind.Arrow)
        rettyid = self.parse_type_id()

        return ExternDef(identifier, type_ids, rettyid, extern_vararg=is_vararg)

    def parse_struct(self) -> list[Node]:
        values: list[Node] = []

        self.advance()
        name = self.expect_and_extract(TokenKind.Identifier)
        self.expect(TokenKind.Colon)
        self.expect(TokenKind.Indent)

        fields: list[tuple[str, TypeID]] = []

        while not self.match(TokenKind.KwdMethod, TokenKind.KwdConstructor, TokenKind.Dedent):
            fields.append(self.parse_field())

        values.append(StructDef(name, fields, create_constructor=(not self.match(TokenKind.KwdConstructor))))

        self._struct_name = name
        seen_constructor = False

        while not self.match(TokenKind.Dedent):
            match self.top.kind:
                case TokenKind.KwdConstructor:  # TODO: multiple constructors allowed ?
                    if seen_constructor:
                        error(
                            "ParserError",
                            f"{self.filename}:{self.top.location.line}:{self.top.location.column} - Struct {name} multiple constructors not allowed"
                        )

                    seen_constructor = True
                    values.append(self.parse_constructor())

                case TokenKind.KwdMethod:
                    values.append(self.parse_method())

        self._struct_name = ""
        self.advance()
        return values

    def parse_function(self) -> FuncDef:
        self.advance()
        name = self.expect_and_extract(TokenKind.Identifier)
        self.expect(TokenKind.Lpar)
        args: list[tuple[str, TypeID]] = self.parse_parameter_list()
        self.expect(TokenKind.Rpar)
        self.expect(TokenKind.Arrow)
        rettyid = self.parse_type_id()
        self.expect(TokenKind.Colon)
        body = self.parse_suite()

        return FuncDef(name, rettyid, args, body, mangle=(name != "main"))

    def parse_constructor(self) -> FuncDef:
        self.advance()
        name = self.expect_and_extract(TokenKind.Identifier)

        if name != self._struct_name:
            error("ParserError", f"{self.filename}:{self.top.location.line}:{self.top.location.column} - Constructor name must match struct name")

        self.expect(TokenKind.Lpar)
        args: list[tuple[str, TypeID]] = self.parse_parameter_list()
        self.expect(TokenKind.Rpar)
        self.expect(TokenKind.Arrow)
        rettyid = self.parse_type_id()
        self.expect(TokenKind.Colon)
        body = self.parse_suite()

        return FuncDef(name, rettyid, args, body, mangle=True)

    def parse_method(self) -> FuncDef:
        self.advance()
        name = self.expect_and_extract(TokenKind.Identifier)
        self.expect(TokenKind.Lpar)
        args: list[tuple[str, TypeID]] = self.parse_parameter_list(implicit_self=True)
        self.expect(TokenKind.Rpar)
        self.expect(TokenKind.Arrow)
        rettyid = self.parse_type_id()
        self.expect(TokenKind.Colon)
        body = self.parse_suite()

        return FuncDef(name, rettyid, args, body, mangle=True)

    def parse_type_id(self) -> TypeID:
        type_id = self.expect_and_extract(TokenKind.Identifier)
        pointer = 0

        while self.match(TokenKind.Star):
            pointer += 1
            self.advance()

        return TypeID(type_id, pointer=pointer)

    def parse_type_id_list(self) -> tuple[list[TypeID], bool]:
        type_ids: list[TypeID] = []
        is_vararg = False

        while not self.match(TokenKind.Rpar):
            type_ids.append(self.parse_type_id())
            if not self.match(TokenKind.Rpar):
                self.expect(TokenKind.Comma)
                if self.match(TokenKind.Ellipsis):
                    is_vararg = True
                    self.advance()
                    if not self.match(TokenKind.Rpar):
                        error("ParserError", f"{self.filename}:{self.top.location.line}:{self.top.location.column} - Expected ) after vararg")

        return (type_ids, is_vararg)

    def parse_field(self) -> tuple[str, TypeID]:
        return (self.expect_and_extract(TokenKind.Identifier), self.parse_type_id())

    def parse_suite(self) -> list[Statement]:
        statements: list[Statement] = []

        match self.top.kind:
            case TokenKind.Indent:
                self.advance()

                while not self.match(TokenKind.Dedent):
                    statements.append(self.parse_statement())

                self.advance()

            case _:
                statements.append(self.parse_statement())

        return statements

    def parse_parameter_list(self, *, implicit_self: bool = False) -> list[tuple[str, TypeID]]:
        params: list[tuple[str, TypeID]] = []

        if implicit_self:
            params.append(("self", TypeID(self._struct_name, pointer=True)))

        while not self.match(TokenKind.Rpar):
            params.append(self.parse_field())
            if not self.match(TokenKind.Rpar):
                self.expect(TokenKind.Comma)

        return params

    def parse_statement(self) -> Statement:
        match self.top.kind:
            case TokenKind.KwdVar:
                return self.parse_declaration()

            case TokenKind.KwdWhile:
                return self.parse_while()

            case TokenKind.KwdIf:
                return self.parse_if_else()

            case TokenKind.KwdReturn:
                return self.parse_return()

            case TokenKind.KwdBreak:
                return self.parse_break()

            case TokenKind.KwdContinue:
                return self.parse_continue()

            case _:
                return self.parse_expr()

    def parse_declaration(self) -> VarDecl:
        self.advance()
        decls: list[Declarator] = [self.parse_declarator()]

        while self.match(TokenKind.Comma):
            self.advance()
            decls.append(self.parse_declarator())

        return VarDecl(decls)

    def parse_while(self) -> While:
        self.advance()
        condition = self.parse_expression()
        self.expect(TokenKind.Colon)

        self._loop_stack.append("while")
        body = self.parse_suite()
        self._loop_stack.pop()

        return While(condition, body)

    def parse_if_else(self) -> IfElse:
        self.advance()
        condition = self.parse_expression()
        self.expect(TokenKind.Colon)
        if_then = self.parse_suite()

        if not self.match(TokenKind.KwdElse):
            return IfElse(condition, if_then, None)

        self.expect(TokenKind.KwdElse)
        self.expect(TokenKind.Colon)
        if_else = self.parse_suite()

        return IfElse(condition, if_then, if_else)

    def parse_return(self) -> Return:
        self.advance()  # TODO: empty return
        return Return(self.parse_expression())

    def parse_break(self) -> Break:
        if not len(self._loop_stack):
            error("ParserError", f"{self.filename}:{self.top.location.line}:{self.top.location.column} - Cannot break outside of loop")

        self.advance()
        return Break()

    def parse_continue(self) -> Continue:
        if not len(self._loop_stack):
            error("ParserError", f"{self.filename}:{self.top.location.line}:{self.top.location.column} - Cannot continue outside of loop")

        self.advance()
        return Continue()

    def parse_expr(self) -> Expr:
        return Expr(self.parse_expression())

    def parse_declarator(self) -> Declarator:
        name = self.expect_and_extract(TokenKind.Identifier)
        type_id = self.parse_type_id()
        self.expect(TokenKind.Equal)
        init = self.parse_expression()

        return Declarator(name, type_id, init)

    def parse_expression(self) -> Expression:
        return self.parse_assignment()

    def parse_assignment(self) -> Expression:
        lhs = self.parse_conditional()

        if not self.match(TokenKind.Equal):
            return lhs

        self.expect(TokenKind.Equal)
        value = self.parse_assignment()
        return Assign(lhs, value)

    def parse_conditional(self) -> Expression:
        condition = self.parse_logical_or()

        if self.match(TokenKind.Question):
            self.advance()
            if_then = self.parse_expression()
            self.expect(TokenKind.Colon)
            if_else = self.parse_expression()
            condition = Conditional(condition, if_then, if_else)

        return condition

    def parse_logical_or(self) -> Expression:
        lhs = self.parse_logical_xor()

        while self.match(TokenKind.KwdOr):
            op = cast(Literal["or"], self.advance_and_extract())
            rhs = self.parse_logical_xor()
            lhs = LogicalOp(op, lhs, rhs)

        return lhs

    def parse_logical_xor(self) -> Expression:
        lhs = self.parse_logical_and()

        while self.match(TokenKind.KwdOr):
            op = cast(Literal["xor"], self.advance_and_extract())
            rhs = self.parse_logical_and()
            lhs = LogicalOp(op, lhs, rhs)

        return lhs

    def parse_logical_and(self) -> Expression:
        lhs = self.parse_logical_not()

        while self.match(TokenKind.KwdOr):
            op = cast(Literal["and"], self.advance_and_extract())
            rhs = self.parse_logical_not()
            lhs = LogicalOp(op, lhs, rhs)

        return lhs

    def parse_logical_not(self) -> Expression:
        if self.match(TokenKind.KwdNot):
            self.advance()
            rhs = self.parse_logical_not()
            return UnaryOp("not", rhs)

        return self.parse_relational()

    def parse_relational(self) -> Expression:
        lhs = self.parse_bitwise_or()

        if self.match(TokenKind.Eq, TokenKind.Ne, TokenKind.Gt, TokenKind.Lt, TokenKind.Ge, TokenKind.Le):
            op = cast(Literal["==", "!=", ">=", "<=", ">", "<"], self.advance_and_extract())
            rhs = self.parse_bitwise_or()
            lhs = ComparisonOp(op, lhs, rhs)

        return lhs

    def parse_bitwise_or(self) -> Expression:
        lhs = self.parse_bitwise_xor()

        while self.match(TokenKind.Or):
            op = cast(Literal["|"], self.advance_and_extract())
            rhs = self.parse_bitwise_xor()
            lhs = BitwiseOp(op, lhs, rhs)

        return lhs

    def parse_bitwise_xor(self) -> Expression:
        lhs = self.parse_bitwise_and()

        while self.match(TokenKind.Xor):
            op = cast(Literal["^"], self.advance_and_extract())
            rhs = self.parse_bitwise_and()
            lhs = BitwiseOp(op, lhs, rhs)

        return lhs

    def parse_bitwise_and(self) -> Expression:
        lhs = self.parse_shift()

        while self.match(TokenKind.And):
            op = cast(Literal["&"], self.advance_and_extract())
            rhs = self.parse_shift()
            lhs = BitwiseOp(op, lhs, rhs)

        return lhs

    def parse_shift(self) -> Expression:
        lhs = self.parse_additive()

        while self.match(TokenKind.Shl, TokenKind.Shr):
            op = cast(Literal["<<", ">>"], self.advance_and_extract())
            rhs = self.parse_additive()
            lhs = BitwiseOp(op, lhs, rhs)

        return lhs

    def parse_additive(self) -> Expression:
        lhs = self.parse_multiplicative()

        while self.match(TokenKind.Plus, TokenKind.Minus):
            op = cast(Literal["+", "-"], self.advance_and_extract())
            rhs = self.parse_multiplicative()
            lhs = BinaryOp(op, lhs, rhs)

        return lhs

    def parse_multiplicative(self) -> Expression:
        lhs = self.parse_cast()

        while self.match(TokenKind.Star, TokenKind.Slash, TokenKind.Percent):
            op = cast(Literal["*", "/", "%"], self.advance_and_extract())
            rhs = self.parse_cast()
            lhs = BinaryOp(op, lhs, rhs)

        return lhs

    def parse_cast(self) -> Expression:
        if self.match(TokenKind.Lt):
            self.advance()
            type_id = self.parse_type_id()
            self.expect(TokenKind.Gt)
            value = self.parse_cast()
            return TypeCast(type_id, value)
        else:
            return self.parse_unary()

    def parse_unary(self) -> Expression:
        match self.top.kind:
            case TokenKind.Minus:
                return UnaryOp(cast(Literal["-", "&", "*"], self.advance_and_extract()), self.parse_unary())

            case TokenKind.And:
                return UnaryOp(cast(Literal["-", "&", "*"], self.advance_and_extract()), self.parse_unary())

            case TokenKind.Star:
                return UnaryOp(cast(Literal["-", "&", "*"], self.advance_and_extract()), self.parse_unary())

            case _:
                return self.parse_postfix()

    def parse_postfix(self) -> Expression:
        lhs = self.parse_primary()

        match self.top.kind:
            case TokenKind.Lsqu:
                self.advance()
                index = self.parse_expression()
                self.expect(TokenKind.Rsqu)
                return Index(lhs, index)

            case TokenKind.Lpar:
                assert isinstance(lhs, Variable)
                self.advance()
                callargs = self.parse_callargs()
                self.expect(TokenKind.Rpar)
                return FuncCall(lhs.var_name, callargs)

            case TokenKind.Dot:
                self.advance()
                member = self.expect_and_extract(TokenKind.Identifier)
                return MemberAccess(lhs, member, ".")

            case TokenKind.Arrow:
                self.advance()
                member = self.expect_and_extract(TokenKind.Identifier)
                return MemberAccess(lhs, member, "->")

            case _:
                return lhs

    def parse_primary(self) -> Expression:
        match self.top.kind:
            case TokenKind.Identifier:
                value = self.advance_and_extract()
                return Variable(value)

            case TokenKind.Lpar:
                self.advance()
                node = self.parse_expression()
                self.expect(TokenKind.Rpar)
                return node

            case _:
                return self.parse_constant()

    def parse_constant(self) -> Expression:
        match self.top.kind:
            case TokenKind.IntLiteral:
                value = self.advance_and_extract()
                return IntLiteral(int(value))

            case TokenKind.ChrLiteral:
                value = self.advance_and_extract()
                return CharLiteral(ord(value))

            case TokenKind.StrLiteral:
                value = self.advance_and_extract()
                return StringLiteral(value[1:-1])

            case _:
                error("ParserError", f"{self.filename}:{self.top.location.line}:{self.top.location.column} - Unexpected {self.top} in parse_constant")

    def parse_callargs(self) -> list[Expression]:
        args: list[Expression] = []

        while not self.match(TokenKind.Rpar):
            args.append(self.parse_expression())
            if not self.match(TokenKind.Rpar):
                self.expect(TokenKind.Comma)

        return args

# endregion


def main() -> int:
    args = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    args.add_argument("filename", type=str, help="File to compile")
    args.add_argument("--emit-llvm", action="store_true", default=False, help="Keep emitted LLVM file")

    parsed = args.parse_args()
    filename = Path(parsed.filename)

    if not filename.exists():
        error("CompilerError", f"{filename} could not be found!")

    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    tree = Parser(str(filename.absolute())).parse()
    cgen = LLVMGeneratorContext(filename.stem)

    try:
        for node in tree:
            node.generate(cgen)
    except Exception as e:
        error("CodeGenerationError", str(e))
    else:
        with open(f"{filename.stem}.ll", "w") as f:
            f.write(str(cgen))

        try:
            check_output(["clang", "-O3", "-o", filename.stem, f"{filename.stem}.ll"])
        except CalledProcessError as e:
            error("CompilerError", f"{e.output}")
        finally:
            if not parsed.emit_llvm:
                remove(f"{filename.stem}.ll")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
