from typing import cast, Literal

from .lexer import Location, TokenKind, generic_error, tokenise
from .ast import (
    Node,
    FunctionDefinition,
    FunctionSignature,
    FunctionParameter,
    TypeIdentifier,
    Stmt,
    Call,
    Var,
    Block,
    Declarator,
    While,
    Expr_,
    Expr,
    Variable,
    Index,
    Assignment,
    Selection,
    ArrayLiteral,
    IntLiteral,
    ComparisonOp,
    BinaryOp,
    Return,
    Program,
    IfElse,
    Break,
    Continue,
    Import,
    Extern,
    UnaryOp,
    FloatLiteral,
    StringLiteral,
    Cast,
    StructDefinition,
    Attr
)


def parser_error(filename: str, location: Location, message: str) -> None:
    generic_error("Parser Error", filename, location, message)


class Parser:
    def __init__(self, source: str, filename: str) -> None:
        self.filename = filename
        self.tokens = tokenise(source, filename)
        self.top = next(self.tokens)

        self.types: set[str] = {"i32", "i8", "float", "string"}

    def advance(self) -> None:
        self.top = next(self.tokens)

    def match(self, *token_kinds: TokenKind, value: str | None = None) -> bool:
        # TODO: Support matching of multiple token kinds at once
        if value is None:
            return self.top.kind in token_kinds

        return (self.top.kind in token_kinds) and (self.top.value == value)

    def expect(self, token_kind: TokenKind) -> None:
        if not self.match(token_kind):
            parser_error(self.filename, self.top.location, f"Unexpected `{self.top.value}`. Expecting token of kind {token_kind}")

        self.advance()

    def parse(self) -> Node:
        imports: list[Import] = []
        structs: list[StructDefinition] = []
        externs: list[Extern] = []
        function_defs: list[FunctionDefinition] = []

        location = self.top.location

        while not self.match(TokenKind.Eof):
            if self.match(TokenKind.Fun):
                function_defs.append(self.parse_function_definition())
            elif self.match(TokenKind.Import):
                imports.append(self.parse_import())
            elif self.match(TokenKind.Extern):
                externs.append(self.parse_extern())
            elif self.match(TokenKind.Type):
                structs.append(self.parse_struct_definition())

        return Program(location, imports, structs, externs, function_defs)

    def parse_import(self) -> Import:
        location = self.top.location
        self.expect(TokenKind.Import)
        mod_name = self.top.value
        assert mod_name is not None
        self.advance()
        return Import(location, mod_name)

    def parse_struct_definition(self) -> StructDefinition:
        location = self.top.location
        self.expect(TokenKind.Type)
        name = self.top.value
        assert name is not None
        self.types.add(name)
        self.advance()
        self.expect(TokenKind.Colon)
        self.expect(TokenKind.Indent)
        body: list[FunctionParameter] = []

        while not self.match(TokenKind.Dedent):
            body.append(self.parse_function_parameter())

        self.expect(TokenKind.Dedent)

        if len(body) == 0:
            parser_error(self.filename, location, f"Struct definition ({name}) has no members!")

        return StructDefinition(location, name, body)

    def parse_extern(self) -> Extern:
        location = self.top.location
        self.expect(TokenKind.Extern)
        signature = self.parse_function_signature()
        return Extern(location, signature)

    def parse_function_definition(self) -> FunctionDefinition:
        location = self.top.location
        self.expect(TokenKind.Fun)
        signature = self.parse_function_signature()
        self.expect(TokenKind.Colon)
        body = self.parse_statement()
        return FunctionDefinition(location, signature, body)

    def parse_function_signature(self) -> FunctionSignature:
        location = self.top.location
        name = self.top.value
        assert name is not None
        self.advance()
        self.expect(TokenKind.Lpar)
        parameters: list[FunctionParameter] = []

        if not self.match(TokenKind.Rpar):
            parameters = self.parse_function_parameters()

        self.expect(TokenKind.Rpar)

        is_variadic = False

        if self.match(TokenKind.Variadic):
            is_variadic = True
            self.advance()

        self.expect(TokenKind.Arrow)

        retty = self.parse_type_identifier()

        return FunctionSignature(location, name, parameters, retty, is_variadic)

    def parse_function_parameters(self) -> list[FunctionParameter]:
        parameters: list[FunctionParameter] = []

        while not self.match(TokenKind.Rpar):
            parameters.append(self.parse_function_parameter())
            if not self.match(TokenKind.Rpar):
                self.expect(TokenKind.Comma)

        return parameters

    def parse_function_parameter(self) -> FunctionParameter:
        location = self.top.location
        name = self.top.value  # TODO: Do we need to match here... for extra safety
        assert name is not None
        self.advance()
        type = self.parse_type_identifier()
        return FunctionParameter(location, name, type)

    def parse_type_identifier(self) -> TypeIdentifier:
        location = self.top.location
        if self.match(TokenKind.Identifier) and (typ := self.top.value) in self.types:
            self.advance()
            return TypeIdentifier(location, typ)
        elif self.match(TokenKind.Lsqu):
            self.advance()
            type = self.parse_type_identifier()
            assert not type.is_array, "2D arrays not yet supported!"
            self.expect(TokenKind.Comma)
            size = self.top.value
            assert size is not None and size.isdigit()
            self.advance()
            self.expect(TokenKind.Rsqu)
            return TypeIdentifier(location, type.typename, True, int(size))
        else:
            parser_error(self.filename, location, f"Unexpected `{self.top.value}` in parse_type_identifier")

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
        elif self.match(TokenKind.If):
            return self.parse_if_statement()
        elif self.match(TokenKind.Continue):
            return self.parse_continue_statement()
        elif self.match(TokenKind.Break):
            return self.parse_break_statement()
        else:
            return self.parse_expr_statement()

    def parse_block_statement(self) -> Block:
        location = self.top.location
        children: list[Stmt] = []
        self.expect(TokenKind.Indent)

        while not self.match(TokenKind.Dedent):
            children.append(self.parse_statement())

        self.expect(TokenKind.Dedent)
        return Block(location, children)

    def parse_var_statement(self) -> Var:
        location = self.top.location
        self.expect(TokenKind.Var)
        declarators: list[Declarator] = []

        if not self.match(TokenKind.Identifier):
            parser_error(self.filename, location, "Var statement expects at least 1 declarator!")

        declarators.append(self.parse_declarator())

        while self.match(TokenKind.Comma):
            self.advance()
            declarators.append(self.parse_declarator())

        return Var(location, declarators)

    def parse_while_statement(self) -> While:
        location = self.top.location
        self.expect(TokenKind.While)
        cond = self.parse_expression()
        self.expect(TokenKind.Colon)
        body = self.parse_statement()

        return While(location, cond, body)

    def parse_return_statement(self) -> Return:
        location = self.top.location
        self.expect(TokenKind.Return)
        value = self.parse_expression()
        return Return(location, value)

    def parse_if_statement(self) -> IfElse:
        location = self.top.location
        self.expect(TokenKind.If)
        if_cond = self.parse_expression()
        self.expect(TokenKind.Colon)
        if_then = self.parse_statement()

        if not self.match(TokenKind.Else):
            return IfElse(location, if_cond, if_then, None)

        self.expect(TokenKind.Else)
        self.expect(TokenKind.Colon)
        if_else = self.parse_statement()
        return IfElse(location, if_cond, if_then, if_else)

    def parse_expr_statement(self) -> Expr_:
        location = self.top.location
        expr = self.parse_expression()
        return Expr_(location, expr)

    def parse_declarator(self) -> Declarator:
        location = self.top.location
        name = self.top.value
        assert name is not None
        self.advance()
        type = self.parse_type_identifier()
        self.expect(TokenKind.Equal)
        init = self.parse_initialiser()
        return Declarator(location, name, type, init)

    def parse_continue_statement(self) -> Continue:
        location = self.top.location
        self.expect(TokenKind.Continue)
        return Continue(location)

    def parse_break_statement(self) -> Break:
        location = self.top.location
        self.expect(TokenKind.Break)
        return Break(location)

    def parse_initialiser(self) -> Expr:
        location = self.top.location
        if self.match(TokenKind.Lsqu):  # * Array initialiser
            self.advance()
            values: list[Expr] = []
            while not self.match(TokenKind.Rsqu):
                values.append(self.parse_expression())
                if not self.match(TokenKind.Rsqu):
                    self.expect(TokenKind.Comma)

            self.expect(TokenKind.Rsqu)

            return ArrayLiteral(location, values)
        else:
            return self.parse_expression()

    def parse_expression(self) -> Expr:
        return self.parse_assignment()

    def parse_assignment(self) -> Expr:
        lhs = self.parse_conditional()

        if not self.match(TokenKind.Equal):
            return lhs

        if not isinstance(lhs, (Variable, Index)):
            # TODO: This location will be wrong... needs to be the lhs location
            parser_error(self.filename, lhs.location, f"lvalue must be assignable, {lhs.__class__.__name__} is not!")

        self.expect(TokenKind.Equal)
        value = self.parse_assignment()
        return Assignment(lhs.location, lhs, value)

    def parse_conditional(self) -> Expr:
        cond = self.parse_logical_or()

        if not self.match(TokenKind.Question):
            return cond

        self.expect(TokenKind.Question)
        then = self.parse_expression()
        self.expect(TokenKind.Colon)
        alt = self.parse_expression()

        return Selection(cond.location, cond, then, alt)

    def parse_logical_or(self) -> Expr:
        lhs = self.parse_logical_xor()

        while self.match(TokenKind.Or):
            op = cast(Literal["or"], self.top.value)
            self.advance()
            rhs = self.parse_logical_xor()
            lhs = BinaryOp(lhs.location, op, lhs, rhs)

        return lhs

    def parse_logical_xor(self) -> Expr:
        lhs = self.parse_logical_and()

        while self.match(TokenKind.Xor):
            op = cast(Literal["xor"], self.top.value)
            self.advance()
            rhs = self.parse_logical_and()
            lhs = BinaryOp(lhs.location, op, lhs, rhs)

        return lhs

    def parse_logical_and(self) -> Expr:
        lhs = self.parse_logical_not()

        while self.match(TokenKind.And):
            op = cast(Literal["and"], self.top.value)
            self.advance()
            rhs = self.parse_logical_not()
            lhs = BinaryOp(lhs.location, op, lhs, rhs)

        return lhs

    def parse_logical_not(self) -> Expr:
        location = self.top.location
        if self.match(TokenKind.Not):
            return UnaryOp(location, "not", self.parse_logical_not())

        return self.parse_relational()

    def parse_relational(self) -> Expr:
        lhs = self.parse_bitwise_or()

        if not self.match(TokenKind.Eq, TokenKind.Ne, TokenKind.Ge, TokenKind.Le, TokenKind.Gt, TokenKind.Lt):
            return lhs

        op = cast(Literal["==", "!=", ">=", "<=", "<", ">"], self.top.value)
        self.advance()
        rhs = self.parse_bitwise_or()

        return ComparisonOp(lhs.location, op, lhs, rhs)

    def parse_bitwise_or(self) -> Expr:
        lhs = self.parse_bitwise_xor()

        while self.match(TokenKind.Pipe):
            op = cast(Literal["|"], self.top.value)
            self.advance()
            rhs = self.parse_bitwise_xor()
            lhs = BinaryOp(lhs.location, op, lhs, rhs)

        return lhs

    def parse_bitwise_xor(self) -> Expr:
        lhs = self.parse_bitwise_and()

        while self.match(TokenKind.Caret):
            op = cast(Literal["^"], self.top.value)
            self.advance()
            rhs = self.parse_bitwise_and()
            lhs = BinaryOp(lhs.location, op, lhs, rhs)

        return lhs

    def parse_bitwise_and(self) -> Expr:
        lhs = self.parse_shifts()

        while self.match(TokenKind.Ampersand):
            op = cast(Literal["&"], self.top.value)
            self.advance()
            rhs = self.parse_shifts()
            lhs = BinaryOp(lhs.location, op, lhs, rhs)

        return lhs

    def parse_shifts(self) -> Expr:
        lhs = self.parse_additive()

        while self.match(TokenKind.Shl, TokenKind.Shr):
            op = cast(Literal["<<", ">>"], self.top.value)
            self.advance()
            rhs = self.parse_additive()
            lhs = BinaryOp(lhs.location, op, lhs, rhs)

        return lhs

    def parse_additive(self) -> Expr:
        lhs = self.parse_multiplicative()

        while self.match(TokenKind.Plus, TokenKind.Minus):
            op = cast(Literal["+", "-"], self.top.value)
            self.advance()
            rhs = self.parse_multiplicative()
            lhs = BinaryOp(lhs.location, op, lhs, rhs)

        return lhs

    def parse_multiplicative(self) -> Expr:
        lhs = self.parse_cast()

        while self.match(TokenKind.Star, TokenKind.Slash, TokenKind.Percent):
            op = cast(Literal["*", "/", "%"], self.top.value)
            self.advance()
            rhs = self.parse_cast()
            lhs = BinaryOp(lhs.location, op, lhs, rhs)

        return lhs

    def parse_cast(self) -> Expr:
        location = self.top.location
        if self.match(TokenKind.Lt):
            self.advance()
            totyp = self.parse_type_identifier()
            self.expect(TokenKind.Gt)
            value = self.parse_cast()
            return Cast(location, totyp, value)

        return self.parse_unary()

    def parse_unary(self) -> Expr:
        location = self.top.location
        if self.match(TokenKind.Minus):
            self.advance()
            return UnaryOp(location, "-", self.parse_unary())

        return self.parse_postfix()

    def parse_postfix(self) -> Expr:
        lhs = self.parse_primary()

        while self.match(TokenKind.Lsqu, TokenKind.Lpar, TokenKind.Dot):

            if self.match(TokenKind.Lsqu):
                self.advance()
                index = self.parse_expression()
                self.expect(TokenKind.Rsqu)
                lhs = Index(lhs.location, lhs, index)
            elif self.match(TokenKind.Lpar):
                self.advance()
                args: list[Expr] = []
                if not self.match(TokenKind.Rpar):
                    args = self.parse_callargs()
                self.expect(TokenKind.Rpar)
                lhs = Call(lhs.location, lhs, args)
            elif self.match(TokenKind.Dot):
                self.advance()
                attr = self.top.value
                assert attr is not None
                self.advance()
                lhs = Attr(lhs.location, lhs, attr)

        return lhs

    def parse_primary(self) -> Expr:
        node: Expr
        location = self.top.location
        if self.match(TokenKind.Identifier):
            assert self.top.value is not None
            node = Variable(location, self.top.value)
            self.advance()
        elif self.match(TokenKind.IntLiteral):
            assert self.top.value is not None
            node = IntLiteral(location, int(self.top.value))
            self.advance()
        elif self.match(TokenKind.StringLiteral):
            assert self.top.value is not None
            node = StringLiteral(location, self.top.value[1:-1] + "\0")
            self.advance()
        elif self.match(TokenKind.FloatLiteral):
            assert self.top.value is not None
            node = FloatLiteral(location, float(self.top.value))
            self.advance()
        elif self.match(TokenKind.Lpar):
            self.advance()
            node = self.parse_expression()
            self.expect(TokenKind.Rpar)
        else:
            parser_error(self.filename, location, f"Unexpected {self.top.value}")

        return node

    def parse_callargs(self) -> list[Expr]:
        args: list[Expr] = []

        while not self.match(TokenKind.Rpar):
            args.append(self.parse_expression())
            if not self.match(TokenKind.Rpar):
                self.expect(TokenKind.Comma)

        return args
