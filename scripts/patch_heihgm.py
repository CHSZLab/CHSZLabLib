#!/usr/bin/env python3
"""Patch HeiHGM sources to replace Abseil dependencies with std equivalents.

Called at CMake configure time to strip absl/protobuf from Bmatching and
Streaming repos so they compile without external dependencies.
"""

import os
import re
import sys


def patch_file(filepath, replacements):
    """Apply a list of (old, new) string replacements to a file."""
    with open(filepath, "r") as f:
        content = f.read()
    original = content
    for old, new in replacements:
        content = content.replace(old, new)
    if content != original:
        with open(filepath, "w") as f:
            f.write(content)
        return True
    return False


def patch_absl_includes(filepath):
    """Replace absl includes with std equivalents."""
    with open(filepath, "r") as f:
        content = f.read()
    original = content

    # Replace absl::string_view with std::string_view
    content = content.replace('absl::string_view', 'std::string_view')

    # Replace absl::flat_hash_map with std::unordered_map
    content = content.replace('absl::flat_hash_map', 'std::unordered_map')

    # Replace absl::flat_hash_set with std::unordered_set
    content = content.replace('absl::flat_hash_set', 'std::unordered_set')

    # Replace absl::StripLeadingAsciiWhitespace
    # This modifies string in-place: remove leading whitespace
    content = content.replace(
        'absl::StripLeadingAsciiWhitespace(&str)',
        'str.erase(0, str.find_first_not_of(" \\t\\n\\r\\f\\v"))'
    )

    # Replace absl include lines with std equivalents
    content = re.sub(
        r'#include\s+"absl/strings/string_view\.h"',
        '#include <string_view>',
        content,
    )
    content = re.sub(
        r'#include\s+"absl/strings/ascii\.h"',
        '// absl/strings/ascii.h removed',
        content,
    )
    content = re.sub(
        r'#include\s+"absl/strings/str_split\.h"',
        '// absl/strings/str_split.h removed',
        content,
    )
    content = re.sub(
        r'#include\s+"absl/container/flat_hash_map\.h"',
        '#include <unordered_map>',
        content,
    )
    content = re.sub(
        r'#include\s+"absl/container/flat_hash_set\.h"',
        '#include <unordered_set>',
        content,
    )
    # Remove remaining absl includes (status, flags, etc.) - not needed
    content = re.sub(
        r'#include\s+"absl/[^"]*"[^\n]*\n',
        '',
        content,
    )

    if content != original:
        with open(filepath, "w") as f:
            f.write(content)
        return True
    return False


def patch_exit_calls(filepath):
    """Replace exit() calls with throw std::runtime_error()."""
    with open(filepath, "r") as f:
        content = f.read()
    original = content

    # Replace exit(1) and exit(0) with throw
    content = content.replace(
        'exit(1)',
        'throw std::runtime_error("fatal error in HeiHGM")'
    )
    content = content.replace(
        'exit(0)',
        'throw std::runtime_error("fatal error in HeiHGM")'
    )

    # Ensure stdexcept is included if we added throws
    if content != original and '#include <stdexcept>' not in content:
        # Add after the first #include or #pragma
        content = re.sub(
            r'(#(?:include|pragma)[^\n]*\n)',
            r'\1#include <stdexcept>\n',
            content,
            count=1,
        )

    if content != original:
        with open(filepath, "w") as f:
            f.write(content)
        return True
    return False


def main():
    if len(sys.argv) < 2:
        print("Usage: patch_heihgm.py <repo_dir>", file=sys.stderr)
        sys.exit(1)

    repo_dir = sys.argv[1]
    if not os.path.isdir(repo_dir):
        print(f"Directory not found: {repo_dir}", file=sys.stderr)
        sys.exit(1)

    patched = 0
    for root, dirs, files in os.walk(repo_dir):
        # Skip .git, BUILD, runner, tools, and third_party dirs
        dirs[:] = [d for d in dirs if d not in (
            '.git', 'runner', 'tools', 'third_party',
        )]
        for fname in files:
            if not fname.endswith(('.h', '.cc', '.cpp', '.hpp')):
                continue
            filepath = os.path.join(root, fname)
            if patch_absl_includes(filepath):
                patched += 1
            if patch_exit_calls(filepath):
                patched += 1

    print(f"Patched {patched} files in {repo_dir}")


if __name__ == "__main__":
    main()
