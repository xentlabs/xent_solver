import argparse

def call_parser(default_description: str, default_conf: str, default_data: str, default_output: str):
    parser = argparse.ArgumentParser(description=default_description)
    parser.add_argument("-c", "--conf", type=str, default=default_conf, help="Path to config JSON")
    parser.add_argument("-d", "--data", type=str, default=default_data, help="Path to data JSON")
    parser.add_argument("-o", "--output", type=str, default=default_output, help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("-g", "--gpus", nargs="+", type=int, default=None, help="GPU device ids: space-separated integers (e.g., '0 1 2') or omit for auto")
    args = parser.parse_args()

    if args.gpus is None:
        args.gpus = "auto"

    return args.conf, args.data, args.output, args.verbose, args.gpus