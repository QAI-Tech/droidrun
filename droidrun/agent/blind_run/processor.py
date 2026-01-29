"""
Blind Run post-processor for DroidRun.

Generates blind_run_ss/ screenshots, blind_run_log.json, and graph_blind.json
from trajectory data after a test execution completes.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("droidrun")


class BlindRunProcessor:
    """
    Post-execution processor that generates blind run artifacts:
    - blind_run_ss/: Screenshots after each interaction (offset by 1 from trajectory screenshots)
    - blind_run_log.json: Log mapping each step to screenshot, action, interaction, reasoning
    - graph_blind.json: Graph with nodes (screen states) and edges (actions)
    """

    def __init__(
        self,
        shared_state,
        trajectory,
        trajectory_folder: Path,
        llm=None,
    ):
        self.shared_state = shared_state
        self.trajectory = trajectory
        self.trajectory_folder = Path(trajectory_folder)
        self.llm = llm

        self.blind_run_ss_folder = self.trajectory_folder / "blind_run_ss"
        self.blind_run_log_path = self.trajectory_folder / "blind_run_log.json"
        self.graph_blind_path = self.trajectory_folder / "graph_blind.json"

        self._blind_run_log: List[Dict[str, Any]] = []
        self._screenshot_bytes_map: Dict[int, bytes] = {}

    def process(self) -> dict:
        """Run all blind run post-processing. Returns paths to generated files."""
        result = {
            "blind_run_ss_folder": None,
            "blind_run_log": None,
            "graph_blind": None,
        }

        try:
            self._save_blind_run_screenshots()
            result["blind_run_ss_folder"] = str(self.blind_run_ss_folder)
        except Exception as e:
            logger.error(f"Failed to save blind run screenshots: {e}")

        try:
            self._generate_blind_run_log()
            result["blind_run_log"] = str(self.blind_run_log_path)
        except Exception as e:
            logger.error(f"Failed to generate blind run log: {e}")

        try:
            self._generate_graph()
            result["graph_blind"] = str(self.graph_blind_path)
        except Exception as e:
            logger.error(f"Failed to generate graph_blind.json: {e}")

        return result

    def _save_blind_run_screenshots(self):
        """
        Save screenshots to blind_run_ss/ folder.

        Screenshots are captured BEFORE each LLM call, so:
        - screenshot[0] = state before action[0] (initial screen)
        - screenshot[i+1] = state after action[i]

        For blind_run_ss we want:
        - 0000.png = screenshot[0] (initial screen before any action)
        - 0001.png = screenshot[1] (state after action[0])
        - etc.

        We read from the already-written screenshots/ folder on disk.
        """
        self.blind_run_ss_folder.mkdir(parents=True, exist_ok=True)

        screenshots_folder = self.trajectory_folder / "screenshots"
        if not screenshots_folder.is_dir():
            logger.warning("No screenshots folder found, skipping blind run screenshots")
            return

        total_screenshots = self.trajectory.screenshot_count
        num_actions = len(self.shared_state.action_history)

        # Copy screenshots: we want screenshot[0] through screenshot[num_actions]
        # screenshot[0] = initial state, screenshot[i+1] = state after action[i]
        screenshots_to_copy = min(total_screenshots, num_actions + 1)

        copied = 0
        for i in range(screenshots_to_copy):
            src = screenshots_folder / f"{i:04d}.png"
            dst = self.blind_run_ss_folder / f"{i:04d}.png"

            if src.exists():
                shutil.copy2(str(src), str(dst))
                # Cache bytes for graph generation
                self._screenshot_bytes_map[i] = src.read_bytes()
                copied += 1
            else:
                logger.warning(f"Screenshot {src.name} not found, skipping")

        logger.info(f"Saved {copied} blind run screenshots to {self.blind_run_ss_folder}")

    def _extract_interaction_summary(self, thought: str) -> str:
        """Extract a concise interaction description from the full thought text."""
        if not thought:
            return "Unknown interaction"

        # Take the first meaningful line as the interaction summary
        lines = [line.strip() for line in thought.strip().split("\n") if line.strip()]
        if lines:
            first_line = lines[0]
            # Remove common prefixes
            for prefix in [
                "(Navigation) Agent Analysis:",
                "(STEP",
                "Agent Analysis:",
                "**",
            ]:
                if first_line.startswith(prefix):
                    first_line = first_line[len(prefix) :].strip()
                    # Remove trailing ** if markdown
                    if first_line.endswith("**"):
                        first_line = first_line[:-2].strip()
                    break
            # Truncate if too long
            if len(first_line) > 150:
                first_line = first_line[:147] + "..."
            return first_line

        return "Unknown interaction"

    def _generate_blind_run_log(self):
        """
        Generate blind_run_log.json from shared_state action/summary history.

        Each entry maps:
        - step: step index
        - screenshot: relative path to blind_run_ss/XXXX.png (state after this action)
        - action: raw code from action_history
        - interaction: human-readable summary from thought
        - reasoning: full thought text
        """
        self._blind_run_log = []

        for i, action_entry in enumerate(self.shared_state.action_history):
            thought = action_entry.get("thought", "")
            code = action_entry.get("code", "")

            # Screenshot after this action = screenshot[i+1]
            screenshot_idx = i + 1
            screenshot_path = f"blind_run_ss/{screenshot_idx:04d}.png"

            # Check if the screenshot file exists
            if not (self.blind_run_ss_folder / f"{screenshot_idx:04d}.png").exists():
                screenshot_path = None

            entry = {
                "step": i,
                "screenshot": screenshot_path,
                "action": code,
                "interaction": self._extract_interaction_summary(thought),
                "reasoning": thought,
            }
            self._blind_run_log.append(entry)

        # Write to disk
        with open(self.blind_run_log_path, "w", encoding="utf-8") as f:
            json.dump(self._blind_run_log, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Generated blind_run_log.json with {len(self._blind_run_log)} entries"
        )

    def _generate_graph(self):
        """Generate graph_blind.json using GraphAgent."""
        from droidrun.agent.blind_run.graph_agent import GraphAgent

        # Build screenshot bytes list for graph agent
        # Node 0 = initial screen (screenshot[0]), Node i+1 = screen after action[i] (screenshot[i+1])
        num_actions = len(self.shared_state.action_history)
        screenshot_bytes_list = []
        for i in range(num_actions + 1):
            screenshot_bytes_list.append(self._screenshot_bytes_map.get(i))

        graph_agent = GraphAgent(
            llm=self.llm,
            blind_run_log=self._blind_run_log,
            screenshot_bytes_list=screenshot_bytes_list,
        )
        graph_data = graph_agent.generate_graph()

        # Write to disk
        with open(self.graph_blind_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Generated graph_blind.json with {len(graph_data.get('nodes', []))} nodes "
            f"and {len(graph_data.get('edges', []))} edges"
        )
