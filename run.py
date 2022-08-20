from argparse import ArgumentParser
from subprocess import CalledProcessError, check_output

def main() -> None:
    argparser = ArgumentParser()
    argparser.add_argument("name", type = str, help = "The program to be run and checked.")
    args = argparser.parse_args()

    try:
        check_output([f"./bin/{args.name}"])
        out = check_output(["echo", "$?"])
    except CalledProcessError as cpe:
        print("Program returned code: %d" % cpe.returncode)

if __name__ == "__main__":
    main()