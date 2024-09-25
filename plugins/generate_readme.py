#!/usr/bin/env python3
"""
Script to generate or update README.md files for Python plugin projects.

This script reads information from pyproject.toml and source code files to
generate a standardized README.md for each plugin in the plugins directory.
"""

import ast
import os
import re
import argparse
from pathlib import Path
from typing import Dict, List

import toml


def extract_pyproject_info(pyproject_path: Path) -> Dict[str, str]:
    """
    Extract necessary information from pyproject.toml.

    Args:
        pyproject_path (Path): Path to the pyproject.toml file.

    Returns:
        Dict[str, str]: A dictionary containing the extracted information.
    """
    with pyproject_path.open("r", encoding="utf-8") as f:
        pyproject_data = toml.load(f)
    project = pyproject_data.get("project", {})
    name = project.get("name", "")
    description = project.get("description", "")
    urls = project.get("urls", {})
    source_url = urls.get("source", "")
    return {
        "name": name,
        "description": description,
        "source_url": source_url,
        "api_docs_url": f"/docs/api/plugins/{name}",
    }


def parse_examples(docstring: str) -> List[Dict[str, str]]:
    """
    Parse the docstring to extract examples.

    Args:
        docstring (str): The docstring of a class or method.

    Returns:
        List[Dict[str, str]]: A list of examples, each example is a dict with 'title' and 'content'.
    """
    examples = []
    lines = docstring.split('\n')
    in_examples_section = False
    current_example_title = ''
    current_example_content = []
    example_started = False

    for line in lines:
        stripped_line = line.strip()
        if stripped_line == 'Examples:':
            in_examples_section = True
            continue
        if in_examples_section:
            example_title_match = re.match(r'^\s*Example: (.*)\s*$', line)
            if example_title_match:
                # Save previous example if any
                if current_example_title and current_example_content:
                    examples.append(
                        {
                            'title': current_example_title,
                            'content': '\n'.join(current_example_content),
                        }
                    )
                    current_example_content = []
                current_example_title = example_title_match.group(1)
                example_started = True
                continue
            elif example_started:
                # Collect code block lines
                if line.strip() == '':
                    continue
                else:
                    if ">>>" in line:
                        current_example_content.append(line.split(">>>")[1].strip())
                    else:
                        # Non-indented line, end of code block
                        example_started = False
                        continue
            else:
                continue
    # Save last example
    if current_example_title and current_example_content:
        examples.append(
            {
                'title': current_example_title,
                'content': '\n'.join(current_example_content),
            }
        )
    return examples


def get_classes_with_docstrings(
    package_dir: Path, package_name: str
) -> List[Dict[str, str]]:
    """
    Parse Python files to extract classes and their docstrings.

    Args:
        package_dir (Path): Path to the package directory containing Python modules.
        package_name (str): The package's importable name.

    Returns:
        List[Dict[str, str]]: A list of dictionaries with class information.
    """
    classes_info = []
    for py_file in package_dir.rglob("*.py"):
        with py_file.open("r", encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read(), filename=str(py_file))
            except SyntaxError:
                continue  # Skip files with syntax errors
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                if class_name.startswith("_"):
                    continue
                module_path = py_file.relative_to(package_dir.parent)
                module_name = ".".join(module_path.with_suffix("").parts)
                full_class_name = f"{module_name}.{class_name}"
                docstring = ast.get_docstring(node)
                if docstring:
                    summary_line = docstring.strip().split("\n")[0]
                    examples = parse_examples(docstring)
                else:
                    summary_line = ""
                    examples = []

                if '# noqa' in summary_line:
                    continue
                classes_info.append(
                    {
                        "class_name": f"{class_name}",
                        "full_class_name": f"`{full_class_name}`",
                        "description": summary_line,
                        "examples": examples,
                    }
                )
    return classes_info


def generate_readme_content(
    info: Dict[str, str], classes_info: List[Dict[str, str]]
) -> str:
    """
    Generate the README.md content based on extracted information.

    Args:
        info (Dict[str, str]): Extracted information from pyproject.toml.
        classes_info (List[Dict[str, str]]): Extracted classes and their docstrings.

    Returns:
        str: The generated README.md content.
    """
    lines = []
    # Title
    title = f"# {info['name']}"
    lines.append(title)
    lines.append("")
    # Description
    lines.append(info["description"])
    lines.append("")
    # Installation
    lines.append("## Installation")
    lines.append("")
    lines.append("```bash")
    lines.append(f"pip install {info['name']}")
    lines.append("```")
    lines.append("")
    # API Section
    lines.append("## API")
    lines.append("")
    lines.append(f"- [Code]({info['source_url']})")
    lines.append(f"- [API-docs]({info['api_docs_url']})")
    lines.append("")
    # Classes Table
    if classes_info:
        lines.append("| Class | Description |")
        lines.append("|---|---|")
        for cls in classes_info:
            class_name = cls["full_class_name"]
            description = cls["description"]
            lines.append(f"| {class_name} | {description} |")
        lines.append("")
    # Examples Section
    examples_size = sum(len(cls["examples"]) for cls in classes_info)
    if examples_size:
        lines.append("## Examples")
        lines.append("")
    for cls in classes_info:
        if cls["examples"]:
            class_heading = f"### {cls['class_name']}"
            lines.append(class_heading)
            lines.append("")
            for example in cls["examples"]:
                example_title = example["title"]
                lines.append(f"#### {example_title}")
                lines.append("")
                lines.append("```python")
                lines.append(example["content"])
                lines.append("```")
                lines.append("")
    lines.append("")
    return "\n".join(lines)


def update_readme(readme_path: Path, new_content: str) -> None:
    """
    Update the README.md file with new content, preserving existing sections.

    Args:
        readme_path (Path): Path to the README.md file.
        new_content (str): The new content to insert into the README.md.
    """
    auto_gen_start = "<!-- Auto-generated content start -->\n"
    auto_gen_end = "<!-- Auto-generated content end -->\n"
    if readme_path.exists():
        with readme_path.open("r", encoding="utf-8") as f:
            existing_content = f.read()
        # Split the content to preserve non-generated parts
        pattern = re.compile(
            re.escape(auto_gen_start) + ".*?" + re.escape(auto_gen_end), re.DOTALL
        )
        new_full_content = pattern.sub(
            auto_gen_start + new_content + "\n" + auto_gen_end, existing_content
        )
    else:
        # Create a new README.md with placeholders
        new_full_content = auto_gen_start + new_content + "\n" + auto_gen_end
        new_full_content += "\n<!-- Add your additional content below -->\n"
    with readme_path.open("w", encoding="utf-8") as f:
        f.write(new_full_content)


def process_plugin(plugin_dir: Path) -> None:
    """
    Process a single plugin directory to generate or update its README.md.

    Args:
        plugin_dir (Path): Path to the plugin directory.
    """
    print(f"Processing plugin at {plugin_dir}")
    pyproject_path = plugin_dir / "pyproject.toml"
    if not pyproject_path.exists():
        print(f"pyproject.toml not found in {plugin_dir}, skipping.")
        return
    info = extract_pyproject_info(pyproject_path)
    # Find the package directory
    package_name = info["name"].replace("-", "_")
    package_dir = plugin_dir / package_name
    if not package_dir.exists():
        print(f"Package directory {package_dir} not found, skipping.")
        return
    classes_info = get_classes_with_docstrings(package_dir, package_name)
    new_readme_content = generate_readme_content(info, classes_info)
    readme_path = plugin_dir / "README.md"
    update_readme(readme_path, new_readme_content)
    print(f"Updated README.md at {readme_path}")


def main():
    """
    Main function to process all plugins in the plugins directory or a specific plugin.
    """
    parser = argparse.ArgumentParser(
        description="Generate or update README.md files for plugins."
    )
    parser.add_argument(
        "plugin_path",
        nargs="?",
        default=None,
        help="Path to the plugin directory (e.g., plugins/openai)",
    )
    args = parser.parse_args()

    if args.plugin_path:
        plugin_dir = Path(args.plugin_path)
        if not plugin_dir.is_dir():
            print(f"Plugin directory {plugin_dir} does not exist.")
            return
        process_plugin(plugin_dir)
    else:
        plugins_dir = Path("plugins")
        for plugin_path in plugins_dir.iterdir():
            if plugin_path.is_dir():
                if plugin_path.name in {"dummy", "template"}:
                    continue
                process_plugin(plugin_path)


if __name__ == "__main__":
    main()
