from pathlib import Path

from src.dsl.template import ProgramSpecTemplate, xed, Symbols, Fixed, Variable
from src.pipeline import run_generic

DESCRIPTION = "Condense program (thin wrapper over generic runner)"
CONFIG = Path("configs/gcg/test.json")
DATA = Path("data/condense/test.json")
OUTPUT_DIR = Path("outputs/condense_program")


def build_spec_for_map(mp):
    s = Symbols({
        "prefix": Variable(mp["prefix_len"]),
        "prefix_intro": Fixed(mp["prefix_intro"]),
        "text_intro": Fixed(mp["text_intro"]),
        "junction": Fixed(mp["junction"]),
        "text": Fixed(mp["text"])
    })

    comp1 = xed(s["text"], s["prefix_intro"] + s["prefix"] + s["junction"] + s["text_intro"])
    comp2 = xed(s["prefix"], s["text_intro"] + s["text"] + s["junction"] + s["prefix_intro"])

    return ProgramSpecTemplate(objective=3.14 * comp1 - 5 * comp2, goal="maximize")


if __name__ == "__main__":
    run_generic(build_spec_for_map, DESCRIPTION, CONFIG, DATA, OUTPUT_DIR)
