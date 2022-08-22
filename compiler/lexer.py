import sys
from enum import Enum
from dataclasses import dataclass
from typing import Generator
from string import ascii_letters, digits

from .constants import RED, RESET, CYAN


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


class TokenKind(Enum):  # TODO: clean this up :)
    IntLiteral = 0
    Identifier = 1
    FloatLiteral = 2
    StringLiteral = 3

    Fun = 10
    Var = 11
    While = 12
    Return = 13
    If = 14
    Else = 15
    Continue = 16
    Break = 17
    Import = 18
    Extern = 19
    And = 49
    Or = 50
    Xor = 51
    Not = 52

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

    Eq = 34
    Ne = 35
    Ge = 36
    Le = 37
    Exclam = 38
    Slash = 39
    Percent = 43
    Shl = 44
    Shr = 45
    Ampersand = 46
    Pipe = 47
    Caret = 48
    Variadic = 53
    Dot = 54

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


def tokenise(source: str, filename: str) -> Generator[Token, None, None]:
    lvl, level, index, line, column = 0, 0, 0, 1, 1
    levels: list[int] = []

    keywords = {
        "fun": TokenKind.Fun,
        "var": TokenKind.Var,
        "while": TokenKind.While,
        "return": TokenKind.Return,
        "if": TokenKind.If,
        "else": TokenKind.Else,
        "continue": TokenKind.Continue,
        "break": TokenKind.Break,
        "import": TokenKind.Import,
        "extern": TokenKind.Extern,
        "and": TokenKind.And,
        "or": TokenKind.Or,
        "xor": TokenKind.Xor,
        "not": TokenKind.Not,
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
        "!": TokenKind.Exclam,
        "/": TokenKind.Slash,
        "%": TokenKind.Percent,
        "<<": TokenKind.Shl,
        ">>": TokenKind.Shr,
        "^": TokenKind.Caret,
        "&": TokenKind.Ampersand,
        "|": TokenKind.Pipe,
        "...": TokenKind.Variadic,
        ".": TokenKind.Dot,

        "==": TokenKind.Eq,
        "!=": TokenKind.Ne,
        ">=": TokenKind.Ge,
        "<=": TokenKind.Le,
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

        elif source[index] in (digits + "."):
            number = source[index]
            column += 1
            while ((index := index + 1) < len(source)) and (source[index] in digits + "_" + "."):
                number += source[index]
                column += 1

            if (count := number.count(".")) > 1:
                tokeniser_error(filename, Location(line, column - len(number)), f"Floating point number can only have 1 decimal, place found {count}")

            yield Token(
                TokenKind.IntLiteral if "." not in number else TokenKind.FloatLiteral,
                number.replace("_", ""),
                Location(line, column - len(number))
            )

        elif source[index] in symbols.keys():
            symbol = source[index]
            column += 1
            while ((index := index + 1) < len(source)) and symbols.get((symbol + source[index])) is not None:
                symbol += source[index]
                column += 1

            yield Token(symbols[symbol], symbol, Location(line, column - len(symbol)))

        elif source[index] == '"':
            string = source[index]
            column += 1
            while ((index := index + 1) < len(source)) and source[index] != '"':
                string += source[index]
                column += 1

            if (index >= len(source)) or (source[index] != '"'):
                tokeniser_error(filename, Location(line, column), "Un-closed string literal")

            string += source[index]
            column += 1
            index += 1

            yield Token(TokenKind.StringLiteral, string, Location(line, column - len(symbol)))

        elif source[index] == " ":
            index += 1
            column += 1

        else:
            tokeniser_error(filename, Location(line, column), f"Unexpected character `{source[index]}`")

    yield Token(TokenKind.Eof, None, Location(line, column))
