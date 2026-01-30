"""
GraphAgent for blind run post-processing.

Produces graph_blind.json with nodes (screen states) and edges (actions)
from blind run log data and screenshots.
"""

import asyncio
import base64
import io
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("droidrun")

INITIAL_SCREEN_PROMPT = """You are analyzing an Android app screenshot. This is the initial screen state before any test actions were performed.

Provide a concise 1-3 word name for this screen.
Examples: "Home", "Login", "Onboarding", "Settings", "Search"

Respond in this exact JSON format:
{"screen_name": "<1-3 word screen name>"}"""

STEP_ANALYSIS_PROMPT = """You are analyzing a UI test step on an Android app.

You are given:
1. BEFORE screenshot: the screen state before the action
2. AFTER screenshot: the screen state after the action
3. The code that was executed
4. The agent's reasoning

Code executed:
{code}

Agent reasoning:
{reasoning}

Provide:
- screen_name: A concise 1-3 word name for the AFTER screen (e.g. "Home", "Settings", "Login Form", "Search Results")
- edge_description: A short professional description (max 10-15 words) of the action performed (e.g. "Tap on Settings icon", "Enter email address in login field", "Scroll down to Logout button")

Respond in this exact JSON format:
{{"screen_name": "<1-3 word screen name>", "edge_description": "<10-15 word action description>"}}"""

STEP_ANALYSIS_NO_IMAGES_PROMPT = """You are analyzing a UI test step on an Android app.

Code executed:
{code}

Agent reasoning:
{reasoning}

Provide:
- screen_name: A concise 1-3 word name for the screen after this action (e.g. "Home", "Settings", "Login Form", "Search Results")
- edge_description: A short professional description (max 10-15 words) of the action performed (e.g. "Tap on Settings icon", "Enter email address in login field", "Scroll down to Logout button")

Respond in this exact JSON format:
{{"screen_name": "<1-3 word screen name>", "edge_description": "<10-15 word action description>"}}"""


class GraphAgent:
    """
    Post-execution agent that generates a graph representation of a blind run.

    Each node represents a screen state (with screenshot + LLM-generated description).
    Each edge represents an action transitioning between screen states.

    Uses a single LLM call per step with before + after screenshots for better
    context and efficiency.
    """

    def __init__(
        self,
        llm,
        blind_run_log: List[Dict[str, Any]],
        screenshot_bytes_list: List[Optional[bytes]],
    ):
        """
        Args:
            llm: LlamaIndex LLM instance for generating descriptions.
            blind_run_log: List of blind run log entries (step, action, interaction, reasoning).
            screenshot_bytes_list: List of screenshot PNG bytes.
                Index 0 = initial screen (before any action).
                Index i+1 = screen after action i.
        """
        self.llm = llm
        self.blind_run_log = blind_run_log
        self.screenshot_bytes_list = screenshot_bytes_list
        self._run_id = str(uuid.uuid4())[:8]

    def generate_graph(self) -> dict:
        """
        Produce the graph_blind.json structure.

        Returns:
            Dict with "nodes" and "edges" lists matching the graph export schema.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an async context, create a new loop in a thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self._async_generate_graph())
                return future.result()
        else:
            return asyncio.run(self._async_generate_graph())

    async def _async_generate_graph(self) -> dict:
        """Async implementation of graph generation.

        Runs all LLM calls concurrently using asyncio.gather for speed.
        """
        nodes = []
        edges = []

        # Prepare all LLM tasks to run concurrently
        # Task 0: initial screen description
        # Tasks 1..N: step analysis (screen_name + edge_description)
        tasks = [self._get_initial_screen_description()]

        for i, entry in enumerate(self.blind_run_log):
            node_idx = i + 1
            before_bytes = self.screenshot_bytes_list[i] if i < len(self.screenshot_bytes_list) else None
            after_bytes = self.screenshot_bytes_list[node_idx] if node_idx < len(self.screenshot_bytes_list) else None

            tasks.append(
                self._analyze_step(
                    entry=entry,
                    before_screenshot=before_bytes,
                    after_screenshot=after_bytes,
                )
            )

        # Run all LLM calls concurrently
        logger.info(f"Running {len(tasks)} LLM calls concurrently for graph generation...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process initial screen result
        initial_description = "Initial Screen"
        if not isinstance(results[0], Exception):
            initial_description = results[0]
        else:
            logger.warning(f"Initial screen LLM failed: {results[0]}")

        initial_node_id = f"node-{self._run_id}-0"
        initial_node = self._build_node(
            node_id=initial_node_id,
            index=0,
            description=initial_description,
            screenshot_bytes=self.screenshot_bytes_list[0] if self.screenshot_bytes_list else None,
        )
        nodes.append(initial_node)

        # Process step results
        for i, entry in enumerate(self.blind_run_log):
            node_idx = i + 1
            node_id = f"node-{self._run_id}-{node_idx}"
            result = results[node_idx]  # results[0] is initial, results[1..N] are steps

            if isinstance(result, Exception):
                logger.warning(f"Step {i} LLM failed: {result}")
                screen_name, edge_description = self._fallback_step(entry)
            else:
                screen_name, edge_description = result

            after_bytes = self.screenshot_bytes_list[node_idx] if node_idx < len(self.screenshot_bytes_list) else None

            node = self._build_node(
                node_id=node_id,
                index=node_idx,
                description=screen_name,
                screenshot_bytes=after_bytes,
            )
            nodes.append(node)

            # Edge from previous node to this node
            source_id = nodes[node_idx - 1]["id"]
            edge_id = f"edge-{self._run_id}-{node_idx}"
            edge = self._build_edge(
                edge_id=edge_id,
                source_id=source_id,
                target_id=node_id,
                entry=entry,
                description=edge_description,
            )
            edges.append(edge)

        return {"nodes": nodes, "edges": edges}

    async def _get_initial_screen_description(self) -> str:
        """Get LLM description for the initial screen (before any action)."""
        if self.llm is None:
            return "Initial Screen"

        try:
            from llama_index.core.base.llms.types import (
                ChatMessage,
                ImageBlock,
                MessageRole,
                TextBlock,
            )

            blocks = [TextBlock(text=INITIAL_SCREEN_PROMPT)]

            # Add screenshot if available
            screenshot_bytes = self.screenshot_bytes_list[0] if self.screenshot_bytes_list else None
            if screenshot_bytes:
                blocks.append(ImageBlock(image=screenshot_bytes))

            messages = [ChatMessage(role=MessageRole.USER, blocks=blocks)]

            response = await asyncio.wait_for(
                self.llm.achat(messages=messages),
                timeout=30,
            )

            if response and response.message and response.message.content:
                return self._parse_screen_name(response.message.content)

        except asyncio.TimeoutError:
            logger.warning("LLM timeout for initial screen description")
        except Exception as e:
            logger.warning(f"LLM failed for initial screen description: {e}")

        return "Initial Screen"

    async def _analyze_step(
        self,
        entry: Dict[str, Any],
        before_screenshot: Optional[bytes],
        after_screenshot: Optional[bytes],
    ) -> Tuple[str, str]:
        """
        Analyze a single step using LLM with before/after screenshots + code + reasoning.

        Returns:
            Tuple of (screen_name, edge_description)
        """
        if self.llm is None:
            return self._fallback_step(entry)

        try:
            from llama_index.core.base.llms.types import (
                ChatMessage,
                ImageBlock,
                MessageRole,
                TextBlock,
            )

            code = entry.get("action", "Unknown")
            reasoning = entry.get("reasoning", "")[:500]

            has_images = before_screenshot is not None or after_screenshot is not None

            if has_images:
                prompt_text = STEP_ANALYSIS_PROMPT.format(
                    code=code,
                    reasoning=reasoning,
                )
                blocks = []

                # Add before screenshot
                if before_screenshot:
                    blocks.append(TextBlock(text="BEFORE screenshot:"))
                    blocks.append(ImageBlock(image=before_screenshot))
                else:
                    blocks.append(TextBlock(text="BEFORE screenshot: (not available)"))

                # Add after screenshot
                if after_screenshot:
                    blocks.append(TextBlock(text="AFTER screenshot:"))
                    blocks.append(ImageBlock(image=after_screenshot))
                else:
                    blocks.append(TextBlock(text="AFTER screenshot: (not available)"))

                # Add the analysis prompt
                blocks.append(TextBlock(text=prompt_text))
            else:
                # No images available, use text-only prompt
                prompt_text = STEP_ANALYSIS_NO_IMAGES_PROMPT.format(
                    code=code,
                    reasoning=reasoning,
                )
                blocks = [TextBlock(text=prompt_text)]

            messages = [ChatMessage(role=MessageRole.USER, blocks=blocks)]

            response = await asyncio.wait_for(
                self.llm.achat(messages=messages),
                timeout=30,
            )

            if response and response.message and response.message.content:
                return self._parse_step_response(response.message.content, entry)

        except asyncio.TimeoutError:
            logger.warning(f"LLM timeout for step {entry.get('step', '?')}")
        except Exception as e:
            logger.warning(f"LLM failed for step {entry.get('step', '?')}: {e}")

        return self._fallback_step(entry)

    def _parse_screen_name(self, response_text: str) -> str:
        """Parse screen name from LLM JSON response."""
        try:
            # Try to extract JSON from response
            text = response_text.strip()
            # Handle cases where LLM wraps in markdown code block
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            data = json.loads(text)
            name = data.get("screen_name", "").strip().strip("\"'")
            if name and len(name) <= 50:
                return name
        except (json.JSONDecodeError, AttributeError, KeyError):
            # Try to extract just the name from plain text
            text = response_text.strip().strip("\"'").strip()
            if text and len(text) <= 50:
                return text

        return "Initial Screen"

    def _parse_step_response(
        self, response_text: str, entry: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Parse screen_name and edge_description from LLM JSON response."""
        try:
            text = response_text.strip()
            # Handle markdown code block wrapping
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            data = json.loads(text)
            screen_name = data.get("screen_name", "").strip().strip("\"'")
            edge_desc = data.get("edge_description", "").strip().strip("\"'").rstrip(".")

            if not screen_name:
                screen_name = self._fallback_step(entry)[0]
            if not edge_desc:
                edge_desc = self._fallback_step(entry)[1]

            # Truncate if needed
            if len(screen_name) > 50:
                screen_name = screen_name[:47] + "..."
            if len(edge_desc) > 80:
                edge_desc = edge_desc[:77] + "..."

            return screen_name, edge_desc

        except (json.JSONDecodeError, AttributeError, KeyError):
            logger.warning(
                f"Failed to parse LLM JSON for step {entry.get('step', '?')}, "
                f"raw response: {response_text[:200]}"
            )

        return self._fallback_step(entry)

    def _fallback_step(self, entry: Dict[str, Any]) -> Tuple[str, str]:
        """Generate fallback descriptions without LLM."""
        screen_name = f"Screen {entry.get('step', '?')}"
        edge_desc = entry.get("interaction", "Perform action")
        return screen_name, edge_desc

    def _build_node(
        self,
        node_id: str,
        index: int,
        description: str,
        screenshot_bytes: Optional[bytes],
    ) -> dict:
        """Build a node dict matching the graph export schema."""
        image_data = None
        if screenshot_bytes:
            image_data = self._encode_screenshot(screenshot_bytes)

        return {
            "id": node_id,
            "type": "customNode",
            "position": {"x": index * 500, "y": 0},
            "data": {
                "description": description,
                "image": image_data,
            },
        }

    def _build_edge(
        self,
        edge_id: str,
        source_id: str,
        target_id: str,
        entry: Dict[str, Any],
        description: str,
    ) -> dict:
        """Build an edge dict matching the graph export schema."""
        return {
            "id": edge_id,
            "source": source_id,
            "target": target_id,
            "sourceHandle": "right-source",
            "targetHandle": "left-target",
            "type": "customEdge",
            "data": {
                "business_logic": entry.get("action", ""),
                "curvature": 0,
                "description": description,
                "source_anchor": "right-source",
                "target_anchor": "left-target",
            },
        }

    def _encode_screenshot(self, screenshot_bytes: bytes) -> str:
        """Encode screenshot bytes to base64 JPEG data URI."""
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(screenshot_bytes))
            # Convert to JPEG for smaller size
            jpeg_buffer = io.BytesIO()
            img.convert("RGB").save(jpeg_buffer, format="JPEG", quality=75)
            jpeg_bytes = jpeg_buffer.getvalue()
            b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"
        except Exception as e:
            logger.warning(f"Failed to encode screenshot as JPEG, using PNG: {e}")
            b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            return f"data:image/png;base64,{b64}"
