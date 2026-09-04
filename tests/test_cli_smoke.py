"""Smoke tests for the command line surface.

These exist because of a real failure. A batch edit left a syntax error in
``cli.py``, and the whole suite still went green: every other test imports the
library modules directly, so nothing in the project ever imported ``cli``. Six
commands were broken and 107 passing tests said otherwise.

The gap was structural, not a missed assertion, so the fix is structural too.
Three layers, cheapest first:

1. every module in the package byte-compiles -- catches a syntax error anywhere,
   including in modules nothing else imports;
2. every module actually imports -- catches bad names and circular imports that
   compiling cannot see, including in modules the CLI only imports lazily
   inside a command function;
3. the parser is built and every subcommand is dispatched to ``--help``, and the
   ``python -m razerpay_fraud`` entry point is executed in a real subprocess.

Nothing here runs a pipeline. A smoke test that takes a minute is a smoke test
people stop running.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import pkgutil
import py_compile
import subprocess
import sys
import unittest
from pathlib import Path

import razerpay_fraud
from razerpay_fraud import cli

PACKAGE_DIR = Path(razerpay_fraud.__file__).parent

#: Every subcommand the CLI is expected to expose. A new command must be added
#: here deliberately, so deleting one by accident fails rather than passes.
EXPECTED_COMMANDS = {"demo", "simulate", "replay", "sweep", "live", "explain"}


def _module_names() -> list[str]:
    return [
        f"razerpay_fraud.{m.name}"
        for m in pkgutil.iter_modules([str(PACKAGE_DIR)])
        # __main__ raises SystemExit on import by design; it is covered by the
        # subprocess test instead.
        if m.name != "__main__"
    ]


class TestPackageIntegrity(unittest.TestCase):
    def test_every_module_byte_compiles(self):
        """A syntax error anywhere in the package fails here, imported or not."""
        for path in sorted(PACKAGE_DIR.glob("*.py")):
            with self.subTest(module=path.name):
                try:
                    py_compile.compile(str(path), doraise=True)
                except py_compile.PyCompileError as exc:
                    self.fail(f"{path.name} does not compile: {exc}")

    def test_every_module_imports(self):
        """Catches bad names and circular imports that compiling cannot see.

        This matters most for modules the CLI imports lazily inside a command
        function (``live``, ``sweep``): a broken import there would otherwise
        only surface when a user ran that command.
        """
        for name in _module_names():
            with self.subTest(module=name):
                importlib.import_module(name)


class TestParser(unittest.TestCase):
    def setUp(self):
        self.parser = cli.build_parser()

    def _subcommands(self) -> dict:
        actions = [
            a for a in self.parser._subparsers._group_actions
            if hasattr(a, "choices") and a.choices
        ]
        self.assertTrue(actions, "parser exposes no subcommands at all")
        return dict(actions[0].choices)

    def test_expected_subcommands_are_present(self):
        self.assertEqual(EXPECTED_COMMANDS, set(self._subcommands()))

    def test_every_subcommand_dispatches_to_a_callable(self):
        """A subparser with no ``func`` default crashes only when invoked."""
        for name, subparser in self._subcommands().items():
            with self.subTest(command=name):
                func = subparser.get_default("func")
                self.assertIsNotNone(func, f"'{name}' has no func default")
                self.assertTrue(callable(func))
                self.assertTrue(
                    hasattr(cli, func.__name__),
                    f"'{name}' dispatches to {func.__name__}, missing from cli",
                )

    def test_every_subcommand_help_renders(self):
        """``--help`` exercises the whole subparser, including its defaults."""
        for name in self._subcommands():
            with self.subTest(command=name):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    with self.assertRaises(SystemExit) as caught:
                        self.parser.parse_args([name, "--help"])
                self.assertEqual(caught.exception.code, 0)
                self.assertIn("usage:", buf.getvalue())

    def test_no_arguments_is_an_error_not_a_traceback(self):
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            with self.assertRaises(SystemExit) as caught:
                self.parser.parse_args([])
        self.assertNotEqual(caught.exception.code, 0)

    def test_an_unknown_command_is_rejected(self):
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            with self.assertRaises(SystemExit):
                self.parser.parse_args(["definitely-not-a-command"])


class TestEntryPoint(unittest.TestCase):
    """The real thing: ``python -m razerpay_fraud`` in a fresh interpreter.

    Slower than the in-process tests, so it runs a couple of invocations rather
    than one per command. It is what fails if ``__main__.py`` or the package's
    import-time behaviour breaks, which no in-process test can see.
    """

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "razerpay_fraud", *args],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PACKAGE_DIR.parent),
        )

    def test_module_entry_point_prints_usage(self):
        result = self._run("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout)
        for name in EXPECTED_COMMANDS:
            self.assertIn(name, result.stdout, f"'{name}' missing from top-level help")

    def test_a_subcommand_help_works_through_the_entry_point(self):
        # 'live' is the useful one to check: cli imports it lazily, so this is
        # the only path that proves that import actually resolves.
        result = self._run("live", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--speed", result.stdout)

    def test_a_bad_command_exits_nonzero_without_a_traceback(self):
        result = self._run("nope")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
