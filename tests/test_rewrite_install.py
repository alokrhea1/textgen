"""Installer contracts; no package installation or model downloads."""
from pathlib import Path
import json
from subprocess import CompletedProcess
import tempfile
import unittest
from unittest.mock import patch

import one_click

ROOT = Path(__file__).resolve().parents[1]


class RewriteInstallTests(unittest.TestCase):
    def test_every_profile_installs_shared_rewrite_dependencies(self):
        for profile in ["full", "portable"]:
            for requirement in (ROOT / "requirements" / profile).glob("*.txt"):
                with self.subTest(profile=profile, requirement=requirement.name):
                    lines = one_click.read_requirements(requirement)
                    self.assertIn("sentence-transformers==6.1.0", lines)
                    self.assertIn("sentencepiece==0.2.1", lines)
                    self.assertIn("protobuf==6.33.5", lines)
                    self.assertEqual(lines.count("transformers==5.6.*"), 1)
                    self.assertFalse(any(line.startswith("-r ") for line in lines))
                    torch_lines = [line for line in lines if line.startswith("torch==")]
                    if profile == "full":
                        self.assertEqual(torch_lines, [])
                    elif requirement.name in {"requirements.txt", "requirements_ik.txt"}:
                        self.assertEqual(len(torch_lines), 1)
                        self.assertIn("2.6.0+cu124", torch_lines[0])
                    elif "cuda131" in requirement.name:
                        self.assertEqual(len(torch_lines), 1)
                        self.assertIn("2.9.0+cu128", torch_lines[0])
                    elif requirement.name == "requirements_amd.txt":
                        self.assertEqual(len(torch_lines), 1)
                        self.assertIn('platform_system == "Windows"', torch_lines[0])
                        rocm_lines = [line for line in lines if "torch-2.9.1%2Brocm7.2" in line]
                        self.assertEqual(len(rocm_lines), 4)
                    else:
                        self.assertEqual(len(torch_lines), 2)
                        self.assertTrue(any("2.9.0+cpu" in line for line in torch_lines))

    def test_native_intel_macos_fails_before_installing_packages(self):
        with patch.object(one_click, "is_macos", return_value=True), \
                patch.object(one_click, "is_x86_64", return_value=True), \
                patch.object(one_click, "run_cmd") as run:
            with self.assertRaisesRegex(SystemExit, "Native macOS Intel"):
                one_click.install_webui()
            with self.assertRaisesRegex(SystemExit, "Use Linux"):
                one_click.update_requirements()
            run.assert_not_called()

    def test_nested_includes_resolve_from_containing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "main.txt").write_text("-r nested/child.txt\nrequests\n")
            (root / "nested" / "child.txt").write_text("--requirement ../shared.txt\n")
            (root / "shared.txt").write_text("sentence-transformers==6.1.0\n")
            self.assertEqual(one_click.read_requirements(root / "main.txt"),
                             ["sentence-transformers==6.1.0", "requests"])
            (root / "shared.txt").write_text("-r main.txt\n")
            with self.assertRaisesRegex(ValueError, "Circular"):
                one_click.read_requirements(root / "main.txt")

    def test_gemma_override_has_only_one_transformers_pin(self):
        lines = one_click.read_requirements(ROOT / "requirements" / "rewrite-gemma4.txt")
        self.assertEqual([line for line in lines if line.startswith("transformers==")],
                         ["transformers==5.10.4"])
        self.assertIn("sentence-transformers==6.1.0", lines)

    def test_known_gemma_override_survives_updates_and_can_be_disabled(self):
        with patch.dict(one_click.os.environ, {}, clear=True), \
                patch.object(one_click.importlib.metadata, "version", return_value="5.10.4"), \
                patch.object(one_click, "print_big_message"):
            state = {"rewrite_gemma4": False}
            self.assertEqual(one_click.select_rewrite_requirements(["transformers==5.6.*"], state),
                             ["transformers==5.10.4"])
            self.assertTrue(state["rewrite_gemma4"])
            with patch.dict(one_click.os.environ, {"REWRITE_GEMMA4": "0"}):
                self.assertEqual(one_click.select_rewrite_requirements(["transformers==5.6.*"], state),
                                 ["transformers==5.6.*"])
            self.assertFalse(state["rewrite_gemma4"])

    def test_arbitrary_transformers_versions_do_not_select_gemma(self):
        with patch.dict(one_click.os.environ, {}, clear=True), \
                patch.object(one_click.importlib.metadata, "version", return_value="5.11.0"):
            state = {}
            self.assertEqual(one_click.select_rewrite_requirements(["transformers==5.6.*"], state),
                             ["transformers==5.6.*"])
            self.assertFalse(state["rewrite_gemma4"])

    @patch.object(one_click, "is_windows", return_value=False)
    def test_amd_installer_uses_published_rocm_torch_version(self, _windows):
        self.assertIn("torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4",
                      one_click.get_pytorch_install_command("AMD"))
        self.assertIn("torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4",
                      one_click.get_pytorch_update_command("AMD"))

    def test_windows_amd_uses_cpu_torch_without_changing_gguf_backend(self):
        with patch.object(one_click, "is_windows", return_value=True):
            self.assertIn("/whl/cpu", one_click.get_pytorch_install_command("AMD"))
            self.assertIn("/whl/cpu", one_click.get_pytorch_update_command("AMD"))
            self.assertTrue(one_click.get_requirements_file("AMD").endswith("requirements_amd.txt"))

    def test_gemma_selection_is_saved_before_extension_mutations(self):
        with patch.dict(one_click.os.environ, {}, clear=True), \
                patch.object(one_click.importlib.metadata, "version", return_value="5.10.4"), \
                patch.object(one_click, "load_state", return_value={}), \
                patch.object(one_click, "save_state") as save, \
                patch.object(one_click, "print_big_message"):
            one_click.preserve_rewrite_selection()
            save.assert_called_once_with({"rewrite_gemma4": True})

    def test_container_and_colab_routes_use_the_fork(self):
        for variant in ["nvidia", "cpu", "amd", "intel"]:
            dockerfile = (ROOT / "docker" / variant / "Dockerfile").read_text()
            compose = (ROOT / "docker" / variant / "docker-compose.yml").read_text()
            self.assertIn("COPY . /home/app/textgen", dockerfile)
            self.assertNotIn("git clone", dockerfile)
            self.assertIn("context: ../..", compose)
            self.assertIn(f"dockerfile: docker/{variant}/Dockerfile", compose)
        ignored = (ROOT / ".dockerignore").read_text().splitlines()
        self.assertIn("**/user_data/**", ignored)
        self.assertIn("portable_env", ignored)
        self.assertNotIn("!user_data/settings.yaml", ignored)
        self.assertNotIn("!user_data/mcp.json", ignored)
        self.assertNotIn("!user_data/CMD_FLAGS.txt", ignored)
        trt = (ROOT / "docker" / "TensorRT-LLM" / "Dockerfile").read_text()
        self.assertIn("scripts/setup_tensorrt.py", trt)
        self.assertIn("--runtime-dir /opt/tensorrt_runtime", trt)
        self.assertNotIn("tensorrt_llm==1.1.0", trt)
        notebook = json.loads((ROOT / "Colab-TextGen-GPU.ipynb").read_text())
        source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("git clone https://github.com/alokrhea1/textgen", source)
        self.assertNotIn("git clone https://github.com/oobabooga", source)

    def test_fork_updates_tracking_branch_without_resetting_origin(self):
        commands = []
        def run(command, **kwargs):
            commands.append(command)
            return CompletedProcess(command, 0, b"https://github.com/alokrhea1/textgen.git\n", b"")
        with patch.object(one_click, "run_cmd", side_effect=run), patch.object(one_click, "print_big_message"):
            one_click.update_repository()
        self.assertEqual(commands, ["git remote get-url origin", "git pull --autostash --ff-only"])

    def test_source_archive_never_bootstraps_upstream_git(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(one_click, "script_dir", directory), \
                patch.object(one_click.sys, "version_info", (3, 0)), \
                patch.object(one_click, "print_big_message"), patch.object(one_click, "run_cmd") as run:
            self.assertEqual(one_click.get_current_commit(), "source-archive")
            with self.assertRaises(SystemExit):
                one_click.update_requirements()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
