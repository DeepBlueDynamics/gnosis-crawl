"""MCP server for grubcrawler.

Exposes grub crawl, screenshot, and Ghost Protocol tools to Claude Code
and any other MCP client. Mounted at /mcp in main.py.

Connect from Claude Code by adding to .claude/settings.json:

  {
    "mcpServers": {
      "grubcrawler": {
        "url": "http://localhost:6792/mcp",
        "transport": "streamable-http"
      }
    }
  }

Or for the deployed service:

  {
    "mcpServers": {
      "grubcrawler": {
        "url": "https://grub.nuts.services/mcp",
        "transport": "streamable-http",
        "headers": { "Authorization": "Bearer <your-token>" }
      }
    }
  }
"""

from __future__ import annotations

import base64
import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "grubcrawler",
    instructions=(
        "Stealth web crawler with bot-bypass capabilities. "
        "Use grub_crawl to fetch page markdown, grub_screenshot for screenshots, "
        "and grub_diagnose on blocked URLs to identify what's blocking and what to try next. "
        "Set warmup=true on sites with Cloudflare or session-based bot detection."
    ),
    streamable_http_path="/",  # mounted at /mcp in FastAPI, so /mcp → /
    stateless_http=True,       # no session state needed for a crawl service
)


@mcp.tool()
async def grub_crawl(
    url: str,
    javascript: bool = True,
    warmup: bool = False,
    timeout: int = 30,
) -> str:
    """Crawl a URL and return its content as markdown.

    Args:
        url: The URL to crawl.
        javascript: Enable JavaScript rendering (default true). Set false for
            plain HTML pages to speed things up.
        warmup: Visit the site homepage first to establish a session before
            crawling the target URL. Helps bypass Cloudflare and similar
            session-based bot detection. Slower but more effective.
        timeout: Navigation timeout in seconds (default 30).

    Returns:
        Page content as markdown text.
    """
    from app.crawler import get_crawler_engine

    engine = await get_crawler_engine()
    result = await engine.crawl_url(
        url=url,
        javascript=javascript,
        warmup=warmup,
        timeout=timeout,
    )

    if not result.get("success"):
        error = result.get("error", "crawl failed")
        blocked = result.get("blocked") or result.get("content_quality") == "blocked"
        if blocked:
            return (
                f"[BLOCKED] Could not retrieve content from {url}.\n"
                f"Error: {error}\n"
                f"Tip: try with warmup=true, or use grub_diagnose to identify the block type."
            )
        return f"[ERROR] {error}"

    markdown = result.get("markdown") or result.get("content", "")
    if not markdown or len(markdown.strip()) < 50:
        return (
            f"[THIN CONTENT] Page returned very little text (chars={result.get('body_char_count', 0)}).\n"
            f"URL: {url}\n"
            f"Use grub_diagnose to check if the page is blocked."
        )
    return markdown


@mcp.tool()
async def grub_screenshot(
    url: str,
    javascript: bool = True,
    timeout: int = 30,
) -> str:
    """Take a full-page screenshot of a URL.

    Args:
        url: The URL to screenshot.
        javascript: Enable JavaScript (default true).
        timeout: Navigation timeout in seconds (default 30).

    Returns:
        Base64-encoded PNG screenshot.
    """
    from app.agent.ghost import capture_screenshot

    capture = await capture_screenshot(url, javascript=javascript, timeout=timeout)

    if not capture.success:
        raise ValueError(f"Screenshot failed for {url}: {capture.error}")

    return base64.b64encode(capture.image_bytes).decode("utf-8")


@mcp.tool()
async def grub_diagnose(
    url: str,
    timeout: int = 30,
) -> dict:
    """Run Ghost Protocol on a URL to diagnose why a crawl failed.

    Takes a screenshot of what the browser actually rendered (typically a
    block page, challenge, or cookie wall) and uses vision AI to identify
    the block type and recommend the next action.

    Args:
        url: The URL that failed to crawl.
        timeout: Navigation timeout in seconds (default 30).

    Returns:
        A dict with:
          - block_type: CLOUDFLARE | AKAMAI | CAPTCHA | LOGIN_WALL | COOKIE_CONSENT |
                        JS_GATE | RATE_LIMITED | ACCESS_DENIED | ERROR_PAGE | EMPTY | UNKNOWN
          - description: What was visible on the blocked page
          - action: Recommended next step (WARMUP | ACCEPT_COOKIES | INJECT_COOKIES |
                    RETRY_JS | LOGIN_REQUIRED | UNSOLVABLE)
          - action_reason: Why this action is recommended
    """
    from app.agent.ghost import run_ghost_protocol, create_ghost_provider
    from app.config import settings

    try:
        provider = create_ghost_provider()
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to initialize vision provider: {exc}",
            "block_type": "UNKNOWN",
            "action": "UNSOLVABLE",
        }

    diagnosis = await run_ghost_protocol(
        url,
        provider=provider,
        max_width=settings.agent_ghost_max_image_width,
        timeout=timeout,
    )

    return {
        "success": diagnosis.success,
        "url": diagnosis.url,
        "block_type": diagnosis.block_type,
        "description": diagnosis.description,
        "action": diagnosis.action,
        "action_reason": diagnosis.action_reason,
        "capture_ms": diagnosis.capture_ms,
        "diagnosis_ms": diagnosis.diagnosis_ms,
        "total_ms": diagnosis.total_ms,
        "error": diagnosis.error,
    }
