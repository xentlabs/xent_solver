from pathlib import Path

from src.dsl.template import ProgramSpecTemplate, nex, Symbols, Fixed, Variable
from src.pipeline import run_generic

DESCRIPTION = "Hacking template (optimize suffix to elicit target response)"
CONFIG = Path("configs/gcg/test.json")

# You can use either JSON or Python data files
# DATA = Path("data/hacking/targets.json")
DATA = Path("data/hacking/targets.py")

OUTPUT_DIR = Path("outputs/hacking_results")


def build_spec_for_map(mp):
    # mp comes from the "details" field in your data file
    s = Symbols({
        "user_prompt": Fixed(mp["user_prompt"]),
        "suffix": Variable(mp["suffix_len"]),
        "target": Fixed(mp["target_output"]),
    })

    # We want to maximize P(target | user_prompt + suffix)
    # nex(A, B) corresponds to the Negative Cross Entropy of A given B
    # Maximizing nex is equivalent to minimizing the Loss(Target | Context)
    objective = nex(s["target"], s["user_prompt"] + s["suffix"])

    return ProgramSpecTemplate(objective=objective, goal="maximize")


if __name__ == "__main__":
    run_generic(build_spec_for_map, DESCRIPTION, CONFIG, DATA, OUTPUT_DIR)

