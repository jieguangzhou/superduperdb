import json
import os
import re


def process_snippet(nb, tabs):
    to_delete = []
    for i, cell in enumerate(nb["cells"]):
        if not cell.get("source"):
            continue
        if tabs != "*":
            match = re.match("^#[ ]+<tab: ([^>]+)>", cell["source"][0])
            if match:
                tab = match.groups()[0]
                if tab not in tabs:
                    to_delete.append(i)
                    continue
        if cell["cell_type"] == "markdown":
            for j, line in enumerate(cell["source"]):
                line = re.sub("^####", "##### ", line)
                line = re.sub("^###", "#### ", line)
                line = re.sub("^## ", "### ", line)
                line = re.sub("^# ", "## ", line)
                nb["cells"][i]["source"][j] = line
    nb["cells"] = [cell for i, cell in enumerate(nb["cells"]) if i not in to_delete]
    return nb


def build_use_case(path, filter_tabs=None):
    with open(path) as f:
        nb = json.load(f)
    built_nb = {k: v for k, v in nb.items() if k != "cells"}
    built_nb["cells"] = []

    for cell in nb["cells"]:
        if (
            cell["cell_type"] == "raw"
            and cell["source"]
            and cell["source"][0].startswith("<snippet:")
        ):
            snippet, tabs = re.match(
                "^<snippet: ([a-z0-9_\-]+): ([a-zA-Z0-9_\-\,\*]+)>$",
                cell["source"][0].strip(),
            ).groups()
            if tabs == "*" and filter_tabs:
                tabs = filter_tabs
            elif tabs != "*" and filter_tabs:
                tabs = [
                    tab.strip() for tab in tabs.split(",") if tab.strip() in filter_tabs
                ]
            with open(f"docs/reusable_snippets/{snippet}.ipynb") as f:
                snippet_nb = json.load(f)
            snippet_nb = process_snippet(snippet_nb, tabs)
            for cell in snippet_nb["cells"]:
                if "<testing:" in "\n".join(cell["source"]):
                    continue
                built_nb["cells"].append(cell)
        else:
            built_nb["cells"].append(cell)
    return built_nb


files = os.listdir("./use_cases")

files = ["_fine_tune_llm_on_database.ipynb"]
# #
# filter_tabs = ["MongoDB", "Text", "OpenAI", "1-Modality", "Context"]
# filter_tabs = ["MongoDB", "Text", "JinaAI", "Anthropic", "1-Modality", "Context"]
# filter_tabs = ["SQL", "SQLite", "Text", "OpenAI", "1-Modality", "Context"]
# filter_tabs = ["SQL", "PostgreSQL", "Text", "OpenAI", "1-Modality", "Context"]
#
# Datatype

# filter_tabs = ["MongoDB", "PDF", "OpenAI", "1-Modality", "Context"]
#
# # Model
# filter_tabs = ["MongoDB", "Text", "Sentence-Transformers", 'Llama.cpp',"1-Modality", "Context"]

# local cluster
# filter_tabs = ['MongoDB Community', 'Experimental Cluster',"MongoDB", "Text", "OpenAI", "1-Modality", "Context"]
#
# filter_tabs = ['MongoDB', 'Prompt-Response',"Local", "Load Trained Model Directly",]
# filter_tabs = ["SQL", "PostgreSQL", 'Prompt-Response',"Local", "Load Trained Model Directly",]
#

# local cluster
filter_tabs = [
    "MongoDB Community",
    "Experimental Cluster",
    "MongoDB",
    "Prompt-Response",
    "Local",
    "Load Trained Model Directly",
]



# local cluster
# filter_tabs = ['MongoDB', 'Prompt-Response', "Ray", "Use a specified checkpoint",]


filter_tabs = None

for file in files:
    if not file.startswith("_"):
        continue
    built = build_use_case(f"./use_cases/{file}", filter_tabs=filter_tabs)
    with open(f"./use_cases/{file[1:]}", "w") as f:
        json.dump(built, f)
