```bash
  █████████                             ██              █████████                                      ███  ████
 ███░░░░░███                           ███             ███░░░░░███                                    ░░░  ░░███
░███    ░░░   ██████   █████████████  ░░░   █████     ███     ░░░   ██████  █████████████   ████████  ████  ░███   ██████  ████████
░░█████████  ░░░░░███ ░░███░░███░░███      ███░░     ░███          ███░░███░░███░░███░░███ ░░███░░███░░███  ░███  ███░░███░░███░░███
 ░░░░░░░░███  ███████  ░███ ░███ ░███     ░░█████    ░███         ░███ ░███ ░███ ░███ ░███  ░███ ░███ ░███  ░███ ░███████  ░███ ░░░
 ███    ░███ ███░░███  ░███ ░███ ░███      ░░░░███   ░░███     ███░███ ░███ ░███ ░███ ░███  ░███ ░███ ░███  ░███ ░███░░░   ░███
░░█████████ ░░████████ █████░███ █████     ██████     ░░█████████ ░░██████  █████░███ █████ ░███████  █████ █████░░██████  █████
 ░░░░░░░░░   ░░░░░░░░ ░░░░░ ░░░ ░░░░░     ░░░░░░       ░░░░░░░░░   ░░░░░░  ░░░░░ ░░░ ░░░░░  ░███░░░  ░░░░░ ░░░░░  ░░░░░░  ░░░░░
                                                                                            ░███
                                                                                            █████
                                                                                           ░░░░░
```

# Sam's Compiler

An AWESOME compiler that's written for a bit of fun.
Attempting to make something that can self-host 😵‍💫

## Running

* Requires Python 3.10+
* Install requirements `pip install -r requirements.txt`
* Run the compiler `python main.py <filename>`
* Run the executable `./bin/<filename>`

## Basic Feature Roadmap

The following are some of the basic features we're hoping to add to the language.

- [x] New statement constructs
  - [x] If-Else statements
  - [x] Continue and Break
- [x] Operators
  - [x] Comparison
  - [x] Division
  - [x] Modulo
  - [x] Unary
    - [x] Negative
  - [x] Bitwise
    - [x] Shifts
    - [x] And
    - [x] Xor
    - [x] Or
  - [x] Logical
    - [x] Not
    - [x] And
    - [x] Xor
    - [x] Or
- [ ] New primitive data-types
  - [x] Floating points (f32)
    - [x] Annotation
    - [x] Literal
    - [x] Casting floats-ints and ints-floats
    - [x] Arithmetic
  - [ ] Strings
    - [x] Annotation
    - [x] Literal
    - [ ] Escape characters
    - [x] Indexing
- [x] Extern functions
- [ ] "Imports" or "Modules"
  - [x] "import" keyword
  - [x] Relative to compilation file not compiler
  - [x] namespace system? access using `<module name>.<function>`?
  - [ ] libc extern libraries as a stdlib
- [x] Structures
  - [x] Define syntax
  - [x] Automatically generated constructors
  - [x] Get attributes
- [x] Pointers and References

### Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/samnaylor">
        <img src="https://avatars.githubusercontent.com/u/57679802?v=4" style="border-radius: 100%; height: auto;" width="100px;" />
        <br />
        <sub>Sam Naylor</sub>
      </a>
      <br />
    </td>
    <td align="center">
      <a href="https://github.com/Wraith29">
        <img src="https://avatars.githubusercontent.com/u/55260788?v=4" style="border-radius: 100%; height: auto;" width="100px;" />
        <br />
        <sub>Isaac Naylor</sub>
      </a>
      <br />
    </td>
  </tr>
</table>
