"""
GraphAgent for blind run post-processing.

Produces graph_blind.json with nodes (screen states) and edges (actions)
from blind run log data and screenshots.
"""

import asyncio
import base64
import io
import logging
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger("droidrun")

SCREEN_DESCRIPTION_PROMPT = """You are analyzing an Android app screenshot. This screenshot was taken after performing the following action:

Action: {action}
Context: {reasoning}

Provide a concise 2-5 word description of the screen state shown in this screenshot.
Examples: "Home Screen", "Search Results Page", "Login Form", "Product Details", "Shopping Cart"

Description:"""

INITIAL_SCREEN_PROMPT = """You are analyzing an Android app screenshot. This is the initial screen state before any test actions were performed.

Provide a concise 2-5 word description of the screen state shown in this screenshot.
Examples: "Home Screen", "Search Results Page", "Login Form", "Product Details", "Shopping Cart"

Description:"""

EDGE_DESCRIPTION_PROMPT = """You are describing a UI interaction on an Android app. Given the action code and the agent's reasoning, provide a concise professional description of what the user did.

Action code:
{action}

Agent reasoning:
{reasoning}

Write a short, professional action label (3-8 words) describing this interaction.
Examples: "Tap Settings button", "Enter email address", "Scroll down to Logout", "Select Deutschland from list", "Allow notification permission", "Navigate back to Home"

Label:"""


class GraphAgent:
    """
    Post-execution agent that generates a graph representation of a blind run.

    Each node represents a screen state (with screenshot + LLM-generated description).
    Each edge represents an action transitioning between screen states.
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
        """Async implementation of graph generation."""
        nodes = []
        edges = []

        num_actions = len(self.blind_run_log)

        # Generate node for initial screen state (before any action)
        initial_node_id = f"node-{self._run_id}-0"
        initial_description = await self._get_screen_description(
            step_index=0, is_initial=True
        )
        initial_node = self._build_node(
            node_id=initial_node_id,
            index=0,
            description=initial_description,
            screenshot_bytes=self.screenshot_bytes_list[0] if self.screenshot_bytes_list else None,
        )
        nodes.append(initial_node)

        # Generate nodes and edges for each action
        for i, entry in enumerate(self.blind_run_log):
            node_idx = i + 1  # Node index (0 is initial)
            node_id = f"node-{self._run_id}-{node_idx}"

            # Get screen description via LLM
            description = await self._get_screen_description(
                step_index=node_idx, is_initial=False, entry=entry
            )

            # Get edge description via LLM
            edge_description = await self._get_edge_description(entry)

            # Get screenshot bytes for this node
            screenshot_bytes = None
            if node_idx < len(self.screenshot_bytes_list):
                screenshot_bytes = self.screenshot_bytes_list[node_idx]

            node = self._build_node(
                node_id=node_id,
                index=node_idx,
                description=description,
                screenshot_bytes=screenshot_bytes,
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

    async def _get_edge_description(self, entry: Dict[str, Any]) -> str:
        """
        Use LLM to generate a concise, professional action label for an edge.

        Falls back to the interaction field if LLM is unavailable.
        """
        if self.llm is None:
            return entry.get("interaction", "Perform action")

        try:
            from llama_index.core.base.llms.types import ChatMessage, MessageRole

            prompt_text = EDGE_DESCRIPTION_PROMPT.format(
                action=entry.get("action", "Unknown"),
                reasoning=entry.get("reasoning", "")[:500],
            )

            messages = [ChatMessage(role=MessageRole.USER, content=prompt_text)]

            response = await asyncio.wait_for(
                self.llm.achat(messages=messages),
                timeout=30,
            )

            if response and response.message and response.message.content:
                label = response.message.content.strip()
                # Clean up: remove quotes, extra whitespace, trailing punctuation
                label = label.strip('"\'').strip().rstrip(".")
                # Truncate if too long
                if len(label) > 60:
                    label = label[:57] + "..."
                return label

        except asyncio.TimeoutError:
            logger.warning(f"LLM timeout for edge description at step {entry.get('step', '?')}")
        except Exception as e:
            logger.warning(f"LLM failed for edge description at step {entry.get('step', '?')}: {e}")

        return entry.get("interaction", "Perform action")

    async def _get_screen_description(
        self,
        step_index: int,
        is_initial: bool = False,
        entry: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Use LLM to generate a short description of the screen state.

        Falls back to a default description if LLM is unavailable or fails.
        """
        if self.llm is None:
            return self._fallback_description(step_index, is_initial, entry)

        try:
            from llama_index.core.base.llms.types import ChatMessage, MessageRole

            # Build prompt
            if is_initial:
                prompt_text = INITIAL_SCREEN_PROMPT
            else:
                prompt_text = SCREEN_DESCRIPTION_PROMPT.format(
                    action=entry.get("action", "Unknown") if entry else "Unknown",
                    reasoning=entry.get("reasoning", "")[:300] if entry else "",
                )

            messages = [ChatMessage(role=MessageRole.USER, content=prompt_text)]

            # Add screenshot as image if available and LLM supports it
            screenshot_bytes = None
            if step_index < len(self.screenshot_bytes_list):
                screenshot_bytes = self.screenshot_bytes_list[step_index]

            if screenshot_bytes:
                # Try multimodal message with image
                try:
                    b64_image = base64.b64encode(screenshot_bytes).decode("utf-8")
                    messages = [
                        ChatMessage(
                            role=MessageRole.USER,
                            content=[
                                {"type": "text", "text": prompt_text},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{b64_image}"
                                    },
                                },
                            ],
                        )
                    ]
                except Exception:
                    # Fall back to text-only if multimodal fails
                    pass

            response = await asyncio.wait_for(
                self.llm.achat(messages=messages),
                timeout=30,
            )

            if response and response.message and response.message.content:
                description = response.message.content.strip()
                # Clean up: remove quotes, extra whitespace
                description = description.strip('"\'').strip()
                # Truncate if too long
                if len(description) > 50:
                    description = description[:47] + "..."
                return description

        except asyncio.TimeoutError:
            logger.warning(f"LLM timeout for screen description at step {step_index}")
        except Exception as e:
            logger.warning(f"LLM failed for screen description at step {step_index}: {e}")

        return self._fallback_description(step_index, is_initial, entry)

    def _fallback_description(
        self,
        step_index: int,
        is_initial: bool,
        entry: Optional[Dict[str, Any]],
    ) -> str:
        """Generate a fallback description without LLM."""
        if is_initial:
            return "Initial Screen"
        if entry:
            interaction = entry.get("interaction", "")
            if interaction:
                return f"After: {interaction[:40]}"
        return f"Screen State {step_index}"
