"""
Remote API Integration Tests for grub-crawl
Tests deployed API with various scenarios (requires environment configuration)

Run with: pytest -v -m remote tests/test_remote_integration.py
Skip with: pytest -v -m "not remote"
"""
import pytest
import requests
import os
import time
from typing import Optional, Dict, Any

# Mark all tests as remote integration tests
pytestmark = pytest.mark.remote


# Configuration from environment
API_BASE_URL = os.getenv("GRUB_CRAWL_API_URL", "https://grub.nuts.services/")
CUSTOMER_ID = os.getenv("GRUB_CRAWL_CUSTOMER_ID", "kordless")
BEARER_TOKEN = os.getenv("GRUB_CRAWL_BEARER_TOKEN")  # Optional

# Test URLs
TEST_URLS = {
    "simple": "https://example.com",
    "complex": "https://news.ycombinator.com",
    "with_js": "https://www.github.com"
}


@pytest.fixture(scope="module")
def api_client():
    """Create configured API client session"""
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    # Set auth header if token provided
    if BEARER_TOKEN:
        session.headers.update({"Authorization": f"Bearer {BEARER_TOKEN}"})

    # Strip trailing slash from base URL
    base_url = API_BASE_URL.rstrip('/')

    return {
        "session": session,
        "base_url": base_url,
        "customer_id": CUSTOMER_ID,
        "bearer_token": BEARER_TOKEN
    }


@pytest.fixture(scope="module")
def check_api_configured():
    """Skip tests if API URL not configured"""
    if not API_BASE_URL or "your-deployed-url" in API_BASE_URL:
        pytest.skip("API_BASE_URL not configured. Set GRUB_CRAWL_API_URL environment variable.")


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check(self, api_client, check_api_configured):
        """Health endpoint should return 200 and service info"""
        response = api_client["session"].get(f"{api_client['base_url']}/health")

        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"
        assert "service" in data
        assert "version" in data


class TestSingleCrawl:
    """Test single URL crawling"""

    def test_simple_url_with_customer_id(self, api_client, check_api_configured):
        """Crawl simple URL with customer_id"""
        payload = {
            "url": TEST_URLS["simple"],
            "customer_id": api_client["customer_id"],
            "options": {
                "javascript": False,
                "screenshot": False,
                "timeout": 30
            }
        }

        response = api_client["session"].post(
            f"{api_client['base_url']}/api/crawl",
            json=payload
        )

        assert response.status_code == 200
        data = response.json()
        assert data.get("success") == True
        assert data.get("url") == TEST_URLS["simple"]
        assert len(data.get("html", "")) > 0
        assert len(data.get("markdown", "")) > 0
        assert data.get("metadata", {}).get("customer_identifier") is not None

    def test_without_customer_id_anonymous(self, api_client, check_api_configured):
        """Crawl without customer_id should use anonymous or auth email"""
        payload = {
            "url": TEST_URLS["simple"],
            "options": {"javascript": False}
        }

        response = api_client["session"].post(
            f"{api_client['base_url']}/api/crawl",
            json=payload
        )

        assert response.status_code == 200
        data = response.json()
        customer_id = data.get("metadata", {}).get("customer_identifier")

        if api_client["bearer_token"]:
            # With auth: should use authenticated user email
            assert customer_id != "anonymous@grub-crawl.local"
        else:
            # Without auth: should use anonymous
            assert customer_id == "anonymous@grub-crawl.local"


class TestMarkdownEndpoint:
    """Test markdown-only endpoint"""

    def test_markdown_only_crawl(self, api_client, check_api_configured):
        """Markdown endpoint should return only markdown"""
        payload = {
            "url": TEST_URLS["simple"],
            "customer_id": api_client["customer_id"],
            "options": {
                "javascript": False,
                "timeout": 30
            }
        }

        response = api_client["session"].post(
            f"{api_client['base_url']}/api/markdown",
            json=payload
        )

        assert response.status_code == 200
        data = response.json()
        assert data.get("success") == True
        assert len(data.get("markdown", "")) > 0
        # Markdown-only should not include full HTML
        assert "html" not in data or data.get("html") == ""


class TestBatchCrawl:
    """Test batch crawling"""

    @pytest.fixture
    def batch_urls(self):
        """URLs for batch testing"""
        return [TEST_URLS["simple"], TEST_URLS["complex"]]

    def test_batch_crawl_multiple_urls(self, api_client, check_api_configured, batch_urls):
        """Batch crawl should process multiple URLs"""
        payload = {
            "urls": batch_urls,
            "customer_id": api_client["customer_id"],
            "options": {
                "javascript": False,
                "screenshot": False,
                "max_concurrent": 3
            }
        }

        response = api_client["session"].post(
            f"{api_client['base_url']}/api/batch",
            json=payload
        )

        assert response.status_code == 200
        data = response.json()
        assert data.get("success") == True
        assert "job_id" in data or "summary" in data
        assert data.get("total_urls") == len(batch_urls)


class TestSessionFiles:
    """Test session file management"""

    @pytest.fixture
    def crawl_result(self, api_client, check_api_configured):
        """Create a crawl to get session_id"""
        payload = {
            "url": TEST_URLS["simple"],
            "customer_id": api_client["customer_id"],
            "options": {"javascript": False}
        }

        response = api_client["session"].post(
            f"{api_client['base_url']}/api/crawl",
            json=payload
        )

        assert response.status_code == 200
        return response.json()

    def test_list_session_files(self, api_client, check_api_configured, crawl_result):
        """Should list files for a session"""
        session_id = crawl_result.get("metadata", {}).get("session_id")

        if not session_id:
            pytest.skip("No session_id in crawl result")

        response = api_client["session"].get(
            f"{api_client['base_url']}/api/sessions/{session_id}/files",
            params={"customer_id": api_client["customer_id"]}
        )

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "files" in data
        assert len(data.get("files", [])) >= 0


class TestAuthentication:
    """Test authentication behavior"""

    def test_with_bearer_token(self, api_client, check_api_configured):
        """Authenticated requests should work"""
        if not api_client["bearer_token"]:
            pytest.skip("Bearer token not configured")

        payload = {
            "url": TEST_URLS["simple"],
            "options": {"javascript": False}
        }

        response = api_client["session"].post(
            f"{api_client['base_url']}/api/crawl",
            json=payload
        )

        assert response.status_code == 200
        data = response.json()
        # With auth, customer_identifier should be user email from token
        customer_id = data.get("metadata", {}).get("customer_identifier")
        assert customer_id is not None
        assert customer_id != "anonymous@grub-crawl.local"

    def test_without_bearer_token_requires_customer_id(self, check_api_configured):
        """Without auth, customer_id should be required or default to anonymous"""
        # Create session without bearer token
        session = requests.Session()
        session.headers.update({"Content-Type": "application/json"})

        payload = {
            "url": TEST_URLS["simple"],
            "options": {"javascript": False}
        }

        response = session.post(
            f"{API_BASE_URL.rstrip('/')}/api/crawl",
            json=payload
        )

        assert response.status_code == 200
        data = response.json()
        # Should use anonymous identifier
        customer_id = data.get("metadata", {}).get("customer_identifier")
        assert customer_id == "anonymous@grub-crawl.local"


@pytest.mark.slow
class TestStorageDebug:
    """Test storage debug endpoint (if available)"""

    def test_storage_debug_info(self, api_client, check_api_configured):
        """Storage debug should return customer storage info"""
        response = api_client["session"].get(
            f"{api_client['base_url']}/api/debug/storage",
            params={"customer_id": api_client["customer_id"]}
        )

        # This endpoint may not exist in all deployments
        if response.status_code == 404:
            pytest.skip("Storage debug endpoint not available")

        assert response.status_code == 200
        data = response.json()
        assert "customer_hash" in data
        assert "storage_root" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "remote"])
