from typing import cast

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
    Literal,
    IntLiteral,
    ComparisonOp,
    BinaryOp,
    Return,
    Program,
)

def parser_error(filename: str, location: Location, message: str) -> None:
    generic_error("Parser Error", filename, location, message)


class Parser:
    def __init__(self, source: str, filename: str) -> None:
        self.filename = filename
        self.tokens = tokenise(source, filename)
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
