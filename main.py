from argparse import ArgumentParser

from compiler import compile

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("filename", type = str, help = "The file to be compiled")
    args = argparser.parse_args()

    raise SystemExit(compile(args.filename))
