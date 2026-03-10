"""Tests for linkedin_mcp_server.dependencies — async lock + DI integration."""

import asyncio
from contextlib import AbstractAsyncContextManager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from linkedin_mcp_server.dependencies import _browser_lock, get_extractor


class TestGetExtractorIsContextManager:
    """Verify get_extractor() returns an AbstractAsyncContextManager, not a raw async_generator."""

    def test_returns_async_context_manager(self):
        """The @asynccontextmanager decorator must be present so uncalled_for
        recognises the return value and enters/exits it properly."""
        result = get_extractor()
        assert isinstance(result, AbstractAsyncContextManager), (
            f"get_extractor() returned {type(result).__name__}, "
            "expected AbstractAsyncContextManager. "
            "Missing @asynccontextmanager decorator?"
        )

    def test_not_raw_async_generator(self):
        import inspect

        result = get_extractor()
        assert not inspect.isasyncgen(result), (
            "get_extractor() returned a raw async_generator — "
            "uncalled_for's Depends() will inject it as-is instead of resolving it"
        )


class TestGetExtractorYieldsExtractor:
    """Verify get_extractor() yields a real LinkedInExtractor under the lock."""

    @pytest.mark.asyncio
    async def test_yields_extractor(self):
        mock_page = MagicMock()
        mock_browser = MagicMock()
        mock_browser.page = mock_page

        with (
            patch(
                "linkedin_mcp_server.dependencies.get_or_create_browser",
                new=AsyncMock(return_value=mock_browser),
            ),
            patch(
                "linkedin_mcp_server.dependencies.ensure_authenticated",
                new=AsyncMock(),
            ),
        ):
            async with get_extractor() as extractor:
                from linkedin_mcp_server.scraping import LinkedInExtractor

                assert isinstance(extractor, LinkedInExtractor)

    @pytest.mark.asyncio
    async def test_lock_is_held_during_yield(self):
        """The browser lock must be held while the extractor is in use."""
        mock_page = MagicMock()
        mock_browser = MagicMock()
        mock_browser.page = mock_page

        with (
            patch(
                "linkedin_mcp_server.dependencies.get_or_create_browser",
                new=AsyncMock(return_value=mock_browser),
            ),
            patch(
                "linkedin_mcp_server.dependencies.ensure_authenticated",
                new=AsyncMock(),
            ),
        ):
            async with get_extractor() as _extractor:
                assert _browser_lock.locked(), "Lock should be held during yield"
            assert not _browser_lock.locked(), "Lock should be released after exit"


class TestConcurrency:
    """Verify concurrent tool calls are serialised by the lock."""

    @pytest.mark.asyncio
    async def test_concurrent_calls_are_serialised(self):
        """Multiple concurrent get_extractor() calls must not overlap."""
        mock_page = MagicMock()
        mock_browser = MagicMock()
        mock_browser.page = mock_page

        execution_log: list[str] = []

        async def fake_get_browser():
            return mock_browser

        with (
            patch(
                "linkedin_mcp_server.dependencies.get_or_create_browser",
                side_effect=fake_get_browser,
            ),
            patch(
                "linkedin_mcp_server.dependencies.ensure_authenticated",
                new=AsyncMock(),
            ),
        ):

            async def worker(name: str):
                async with get_extractor() as _ext:
                    execution_log.append(f"{name}_start")
                    await asyncio.sleep(0.05)
                    execution_log.append(f"{name}_end")

            await asyncio.gather(worker("a"), worker("b"), worker("c"))

        # With serialisation, starts and ends must alternate (no interleaving)
        starts = [e for e in execution_log if e.endswith("_start")]
        ends = [e for e in execution_log if e.endswith("_end")]
        assert len(starts) == 3
        assert len(ends) == 3

        # Each start must be immediately followed by its own end
        for i in range(0, len(execution_log), 2):
            name = execution_log[i].removesuffix("_start")
            assert execution_log[i + 1] == f"{name}_end", (
                f"Calls were interleaved: {execution_log}"
            )
