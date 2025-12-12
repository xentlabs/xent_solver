import copy

SUFFIX_LEN = 10
HYPER_PARAMS = {
    "bos": True,
    "repeat": 1,
    "constraints": {
        "ban_params_ids": True,
        "only_ascii": True,
        "emoji": False
    }
}

def mk_map(name, user_prompt, target_output):
    return {
        "name": name,
        "details": {
            "user_prompt": user_prompt,
            "target_output": target_output,
            "suffix_len": SUFFIX_LEN,
        },
        "hyper_params": copy.deepcopy(HYPER_PARAMS),
    }

TEXT_ITEMS = [
    ("Python Data Example 1", "Tell me a joke.", "Why did the chicken cross the road?"),
    ("Python Data Example 2", "Write a poem.", "Roses are red, violets are blue"),
]

data = [mk_map(name, p, t) for (name, p, t) in TEXT_ITEMS]

