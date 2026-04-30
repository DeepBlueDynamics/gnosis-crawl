"""
API routes for Grub Crawler service
"""
import uuid
import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Header, Request
from datetime import datetime

from app.models import (
    CrawlRequest, CrawlResult,
    MarkdownRequest, MarkdownResult,
    RawHtmlRequest, RawHtmlResult,
    BatchRequest, BatchResult,
    JobStatus, JobListResponse,
    CacheSearchRequest, CacheUpsertRequest, CachePruneRequest,
)
from app.auth import get_current_user, get_user_email, get_customer_identifier
from app.config import settings
from app.crawler import get_crawler_engine
from fastapi.responses import Response, JSONResponse
from app.storage import CrawlStorageService
from app.cache_store import RemoteCacheStore
from app.proxy import resolve_proxy

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()


async def get_optional_user_email(authorization: str = Header(None)) -> Optional[str]:
    """
    Get user email if auth is enabled and token is provided, otherwise return None.
    Used for routes that support both authenticated and unauthenticated access.
    """
    if settings.disable_auth:
        return None
    
    if not authorization or not authorization.startswith("Bearer "):
        return None
    
    try:
        user = await get_current_user(authorization)
        if "email" in user:
            return user["email"]
        elif user.get("subject", "").startswith("user:"):
            return user["subject"][5:]
        else:
            return None
    except:
        return None


def _crawl_result_to_payload(result: Any, include_html: bool = False) -> Dict[str, Any]:
    """Map internal CrawlResult to stable API response fields."""
    payload = {
        "success": bool(result.success),
        "url": result.url,
        "final_url": result.final_url or result.url,
        "status_code": result.status_code,
        "markdown": result.markdown,
        "markdown_plain": result.markdown_plain,
        "content": result.content,
        "quarantined": bool(getattr(result, "quarantined", False)),
        "quarantine_reason": getattr(result, "quarantine_reason", "") or None,
        "policy_flags": list(getattr(result, "policy_flags", []) or []),
        "blocked": bool(result.blocked),
        "block_reason": result.block_reason or None,
        "captcha_detected": bool(result.captcha_detected),
        "http_error_family": result.http_error_family or None,
        "render_mode": result.render_mode,
        "wait_strategy": result.wait_strategy,
        "timings_ms": result.timings_ms or {},
        "body_char_count": int(result.body_char_count or 0),
        "body_word_count": int(result.body_word_count or 0),
        "visible_char_count": int(getattr(result, "visible_char_count", 0) or 0),
        "visible_word_count": int(getattr(result, "visible_word_count", 0) or 0),
        "visible_similarity": getattr(result, "visible_similarity", None),
        "content_quality": result.content_quality,
        "extractor_version": result.extractor_version,
        "normalized_url": result.normalized_url,
        "content_hash": result.content_hash,
        "screenshot_url": result.screenshot_path or "",
        "error": result.error_message or None,
    }
    if include_html:
        payload["html"] = result.html
    return payload


@router.post("/crawl", response_model=CrawlResult)
async def crawl_single_url(
    request: CrawlRequest,
    user_email: Optional[str] = Depends(get_optional_user_email)
):
    """
    Crawl a single URL and return HTML + markdown
    Synchronous operation that returns results immediately
    Supports both authenticated (via token) and unauthenticated (via customer_id) access
    """
    try:
        # Resolve customer identifier (prioritizes customer_id from request, falls back to user_email)
        customer_identifier = get_customer_identifier(request.customer_id, user_email)
        
        # Get crawler engine for this customer
        crawler = await get_crawler_engine(customer_identifier)
        
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Perform crawl with request options
        javascript_enabled = request.javascript_enabled if request.javascript_enabled is not None else request.options.javascript
        javascript_payload = request.javascript_payload or request.options.javascript_payload

        # Resolve proxy (per-request overrides env-based default)
        proxy = resolve_proxy(getattr(request.options, 'proxy', None))

        result = await crawler.crawl_url(
            url=str(request.url),
            javascript=javascript_enabled,
            screenshot=request.options.screenshot,
            screenshot_mode=request.options.screenshot_mode,
            timeout=request.options.timeout,
            javascript_payload=javascript_payload,
            dedupe_tables=request.options.dedupe_tables,
            wait_until=request.options.wait_until,
            wait_for_selector=request.options.wait_for_selector,
            wait_after_load_ms=request.options.wait_after_load_ms,
            retry_with_js_if_thin=request.options.retry_with_js_if_thin,
            session_id=session_id,
            proxy=proxy,
            cookies=request.options.cookies,
            warmup=request.options.warmup,
        )

        if result.success:
            saved_filename = None
            try:
                if session_id:
                    saved_filename = await crawler._save_crawl_result(result, session_id)
            except Exception:
                saved_filename = None
            return CrawlResult(
                success=True,
                url=result.url,
                html=result.html,
                markdown=result.markdown,
                markdown_plain=result.markdown_plain,
                content=result.content,
                final_url=result.final_url or result.url,
                status_code=result.status_code,
                quarantined=bool(getattr(result, "quarantined", False)),
                quarantine_reason=getattr(result, "quarantine_reason", None),
                policy_flags=list(getattr(result, "policy_flags", []) or []),
                visible_char_count=int(getattr(result, "visible_char_count", 0) or 0),
                visible_word_count=int(getattr(result, "visible_word_count", 0) or 0),
                visible_similarity=getattr(result, "visible_similarity", None),
                blocked=result.blocked,
                block_reason=result.block_reason or None,
                captcha_detected=result.captcha_detected,
                http_error_family=result.http_error_family or None,
                render_mode=result.render_mode,
                wait_strategy=result.wait_strategy,
                timings_ms=result.timings_ms or {},
                body_char_count=result.body_char_count,
                body_word_count=result.body_word_count,
                content_quality=result.content_quality,
                extractor_version=result.extractor_version,
                normalized_url=result.normalized_url,
                content_hash=result.content_hash,
                screenshot_url=result.screenshot_path or "",
                metadata={
                    "title": result.title,
                    "customer_identifier": customer_identifier,
                    "processing_time": result.processing_time,
                    "browser_time": result.browser_time,
                    "markdown_time": result.markdown_time,
                    "page_info": result.page_info,
                    "options": request.options.dict(),
                    "session_id": session_id,
                    "storage": {
                        "result_file": saved_filename,
                        "screenshots": result.screenshot_path if isinstance(result.screenshot_path, list) else ([result.screenshot_path] if result.screenshot_path else [])
                    }
                },
                crawled_at=datetime.utcnow()
            )
        else:
            return CrawlResult(
                success=False,
                url=result.url,
                final_url=result.final_url or result.url,
                status_code=result.status_code,
                quarantined=bool(getattr(result, "quarantined", False)),
                quarantine_reason=getattr(result, "quarantine_reason", None),
                policy_flags=list(getattr(result, "policy_flags", []) or []),
                visible_char_count=int(getattr(result, "visible_char_count", 0) or 0),
                visible_word_count=int(getattr(result, "visible_word_count", 0) or 0),
                visible_similarity=getattr(result, "visible_similarity", None),
                blocked=result.blocked,
                block_reason=result.block_reason or None,
                captcha_detected=result.captcha_detected,
                http_error_family=result.http_error_family or None,
                render_mode=result.render_mode,
                wait_strategy=result.wait_strategy,
                timings_ms=result.timings_ms or {},
                body_char_count=result.body_char_count,
                body_word_count=result.body_word_count,
                content_quality=result.content_quality,
                extractor_version=result.extractor_version,
                normalized_url=result.normalized_url,
                content_hash=result.content_hash,
                screenshot_url=result.screenshot_path or "",
                crawled_at=datetime.utcnow(),
                error=result.error_message,
                metadata={
                    "customer_identifier": customer_identifier,
                    "processing_time": result.processing_time,
                    "options": request.options.dict(),
                    "session_id": session_id
                }
            )
        
    except Exception as e:
        logger.error(f"Failed to crawl {request.url}: {e}", exc_info=True)
        return CrawlResult(
            success=False,
            url=str(request.url),
            crawled_at=datetime.utcnow(),
            error=str(e),
            metadata={"customer_identifier": customer_identifier}
        )


@router.post("/markdown", response_model=MarkdownResult)
async def crawl_markdown_only(
    request: MarkdownRequest,
    x_client_timeout: Optional[str] = Header(None, alias="X-Client-Timeout"),
    user_email: Optional[str] = Depends(get_optional_user_email)
):
    """
    Crawl a URL and return only markdown content
    Optimized for markdown extraction
    Supports both authenticated (via token) and unauthenticated (via customer_id) access
    """
    try:
        # Resolve customer identifier
        customer_identifier = get_customer_identifier(request.customer_id, user_email)
        
        # Get crawler engine for this customer
        crawler = await get_crawler_engine(customer_identifier)
        cache_store = RemoteCacheStore(customer_identifier)
        
        # Perform crawl(s) with stable response contract
        url_candidates = request.urls or ([request.url] if request.url else [])
        javascript_enabled = request.javascript_enabled if request.javascript_enabled is not None else request.options.javascript
        javascript_payload = request.javascript_payload or request.options.javascript_payload

        # Resolve proxy (per-request overrides env-based default)
        proxy = resolve_proxy(getattr(request.options, 'proxy', None))

        # Parse client timeout budget from X-Client-Timeout header
        client_timeout_seconds = int(x_client_timeout) if x_client_timeout and x_client_timeout.isdigit() else None

        per_url_results: List[Dict[str, Any]] = []
        for target in url_candidates:
            crawl_result = await crawler.crawl_url(
                url=str(target),
                javascript=javascript_enabled,
                screenshot=False,
                timeout=request.options.timeout,
                javascript_payload=javascript_payload,
                dedupe_tables=request.options.dedupe_tables,
                wait_until=request.options.wait_until,
                wait_for_selector=request.options.wait_for_selector,
                wait_after_load_ms=request.options.wait_after_load_ms,
                retry_with_js_if_thin=request.options.retry_with_js_if_thin,
                proxy=proxy,
                client_timeout_seconds=client_timeout_seconds,
                cookies=request.options.cookies,
                warmup=request.options.warmup,
            )
            payload = _crawl_result_to_payload(crawl_result, include_html=False)
            cache_doc = None
            if crawl_result.success:
                cache_doc = cache_store.upsert(
                    url=crawl_result.url,
                    markdown=crawl_result.markdown or "",
                    markdown_plain=crawl_result.markdown_plain or "",
                    content=crawl_result.content or "",
                    quality=crawl_result.content_quality or "empty",
                    status_code=crawl_result.status_code,
                    extractor_version=crawl_result.extractor_version,
                    normalized_url=crawl_result.normalized_url,
                    content_hash=crawl_result.content_hash,
                    metadata={
                        "final_url": crawl_result.final_url or crawl_result.url,
                        "blocked": crawl_result.blocked,
                        "block_reason": crawl_result.block_reason,
                        "captcha_detected": crawl_result.captcha_detected,
                    },
                )
                payload["doc_id"] = cache_doc.get("doc_id")
                payload["source_status"] = cache_doc.get("source_status")

            per_url_results.append(payload)

        if not per_url_results:
            return MarkdownResult(
                success=False,
                url=str(request.url) if request.url else "",
                crawled_at=datetime.utcnow(),
                error="No URL provided",
                metadata={"customer_identifier": customer_identifier}
            )

        if len(per_url_results) == 1:
            single = per_url_results[0]
            return MarkdownResult(
                success=bool(single.get("success")),
                url=single.get("url", ""),
                final_url=single.get("final_url"),
                status_code=single.get("status_code"),
                markdown=single.get("markdown"),
                markdown_plain=single.get("markdown_plain"),
                content=single.get("content"),
                quarantined=bool(single.get("quarantined")),
                quarantine_reason=single.get("quarantine_reason"),
                policy_flags=list(single.get("policy_flags") or []),
                blocked=bool(single.get("blocked")),
                block_reason=single.get("block_reason"),
                captcha_detected=bool(single.get("captcha_detected")),
                http_error_family=single.get("http_error_family"),
                render_mode=single.get("render_mode"),
                wait_strategy=single.get("wait_strategy"),
                timings_ms=single.get("timings_ms") or {},
                body_char_count=int(single.get("body_char_count") or 0),
                body_word_count=int(single.get("body_word_count") or 0),
                visible_char_count=int(single.get("visible_char_count") or 0),
                visible_word_count=int(single.get("visible_word_count") or 0),
                visible_similarity=single.get("visible_similarity"),
                content_quality=single.get("content_quality"),
                extractor_version=single.get("extractor_version"),
                normalized_url=single.get("normalized_url"),
                content_hash=single.get("content_hash"),
                metadata={
                    "customer_identifier": customer_identifier,
                    "options": request.options.dict(),
                    "session_id": request.session_id,
                    "doc_id": single.get("doc_id"),
                    "source_status": single.get("source_status"),
                },
                crawled_at=datetime.utcnow(),
                error=single.get("error")
            )

        all_success = all(item.get("success") for item in per_url_results)
        joined_markdown = "\n\n---\n\n".join(
            [f"## {item.get('url')}\n\n{item.get('markdown') or ''}" for item in per_url_results]
        )
        joined_plain = "\n\n---\n\n".join(
            [f"## {item.get('url')}\n\n{item.get('markdown_plain') or ''}" for item in per_url_results]
        )
        aggregate_content = "\n\n---\n\n".join(
            [f"## {item.get('url')}\n\n{item.get('content') or ''}" for item in per_url_results]
        )
        first = per_url_results[0]
        quality_values = [item.get("content_quality") for item in per_url_results if item.get("content_quality")]
        aggregate_quality = "sufficient" if all(q == "sufficient" for q in quality_values) else "minimal"
        aggregate_quarantined = any(bool(item.get("quarantined")) for item in per_url_results)
        aggregate_flags: List[str] = []
        for item in per_url_results:
            for flag in (item.get("policy_flags") or []):
                if flag not in aggregate_flags:
                    aggregate_flags.append(flag)

        return MarkdownResult(
            success=all_success,
            url=first.get("url", ""),
            final_url=first.get("final_url"),
            status_code=first.get("status_code"),
            markdown=joined_markdown,
            markdown_plain=joined_plain,
            content=aggregate_content,
            quarantined=aggregate_quarantined,
            quarantine_reason="one_or_more_urls_quarantined" if aggregate_quarantined else None,
            policy_flags=aggregate_flags,
            blocked=any(bool(item.get("blocked")) for item in per_url_results),
            block_reason=next((item.get("block_reason") for item in per_url_results if item.get("block_reason")), None),
            captcha_detected=any(bool(item.get("captcha_detected")) for item in per_url_results),
            http_error_family=first.get("http_error_family"),
            render_mode=first.get("render_mode"),
            wait_strategy=first.get("wait_strategy"),
            timings_ms=first.get("timings_ms") or {},
            body_char_count=sum(int(item.get("body_char_count") or 0) for item in per_url_results),
            body_word_count=sum(int(item.get("body_word_count") or 0) for item in per_url_results),
            visible_char_count=sum(int(item.get("visible_char_count") or 0) for item in per_url_results),
            visible_word_count=sum(int(item.get("visible_word_count") or 0) for item in per_url_results),
            visible_similarity=None,
            content_quality=aggregate_quality,
            extractor_version=first.get("extractor_version"),
            normalized_url=first.get("normalized_url"),
            content_hash=first.get("content_hash"),
            metadata={
                "customer_identifier": customer_identifier,
                "options": request.options.dict(),
                "session_id": request.session_id,
                "results": per_url_results
            },
            crawled_at=datetime.utcnow(),
            error=None if all_success else "One or more URLs failed"
        )
        
    except Exception as e:
        logger.error(f"Failed to crawl markdown for {request.url}: {e}", exc_info=True)
        return MarkdownResult(
            success=False,
            url=str(request.url),
            crawled_at=datetime.utcnow(),
            error=str(e),
            metadata={"customer_identifier": customer_identifier}
        )


@router.post("/raw", response_model=RawHtmlResult)
async def crawl_raw_html(
    request: RawHtmlRequest,
    user_email: Optional[str] = Depends(get_optional_user_email)
):
    """
    Crawl a URL and return raw HTML content.
    Supports JavaScript execution and custom payload injection.
    """
    customer_identifier = None
    try:
        customer_identifier = get_customer_identifier(request.customer_id, user_email)
        crawler = await get_crawler_engine(customer_identifier)
        javascript_enabled = request.javascript_enabled if request.javascript_enabled is not None else request.options.javascript
        javascript_payload = request.javascript_payload or request.options.javascript_payload

        proxy = resolve_proxy(getattr(request.options, 'proxy', None))

        result = await crawler.crawl_raw_html(
            url=str(request.url),
            javascript=javascript_enabled,
            timeout=request.options.timeout,
            javascript_payload=javascript_payload,
            proxy=proxy
        )

        metadata = {
            "customer_identifier": customer_identifier,
            "options": request.options.dict(),
            "session_id": request.session_id,
            "page_info": result.get("page_info"),
            "processing_time": result.get("processing_time")
        }

        if result.get("success"):
            return RawHtmlResult(
                success=True,
                url=str(request.url),
                html=result.get("html"),
                metadata=metadata,
                crawled_at=datetime.utcnow()
            )
        else:
            return RawHtmlResult(
                success=False,
                url=str(request.url),
                error=result.get("error"),
                metadata=metadata,
                crawled_at=datetime.utcnow()
            )
    except Exception as e:
        logger.error(f"Failed to fetch raw HTML for {request.url}: {e}", exc_info=True)
        return RawHtmlResult(
            success=False,
            url=str(request.url),
            crawled_at=datetime.utcnow(),
            error=str(e),
            metadata={"customer_identifier": customer_identifier}
        )


@router.post("/batch", response_model=BatchResult)
async def crawl_batch_urls(
    request: BatchRequest,
    user_email: Optional[str] = Depends(get_optional_user_email)
):
    """
    Start a batch crawl job for multiple URLs
    Returns results immediately (synchronous batch processing)
    Supports both authenticated (via token) and unauthenticated (via customer_id) access
    """
    try:
        # Resolve customer identifier
        customer_identifier = get_customer_identifier(request.customer_id, user_email)
        
        # Get crawler engine for this customer
        crawler = await get_crawler_engine(customer_identifier)
        
        # Generate session ID for this batch
        session_id = str(uuid.uuid4())
        
        # Convert URLs to strings
        url_list = [str(url) for url in request.urls]
        javascript_enabled = request.javascript_enabled if request.javascript_enabled is not None else request.options.javascript
        javascript_payload = request.javascript_payload or request.options.javascript_payload
        
        logger.info(f"Starting batch crawl for {len(url_list)} URLs (customer: {customer_identifier})")
        
        # Resolve proxy
        proxy = resolve_proxy(getattr(request.options, 'proxy', None))

        # Perform batch crawl (synchronous)
        batch_result = await crawler.batch_crawl(
            urls=url_list,
            javascript=javascript_enabled,
            screenshot=request.options.screenshot,
            max_concurrent=request.concurrent,  # Fixed: it's on BatchRequest, not options
            session_id=session_id,
            javascript_payload=javascript_payload,
            dedupe_tables=request.options.dedupe_tables,
            wait_until=request.options.wait_until,
            wait_for_selector=request.options.wait_for_selector,
            wait_after_load_ms=request.options.wait_after_load_ms,
            retry_with_js_if_thin=request.options.retry_with_js_if_thin,
            proxy=proxy
        )

        
        return BatchResult(
            success=True,
            job_id=session_id,
            total_urls=len(url_list),
            message=f"Batch crawl completed: {batch_result['summary']['success']}/{batch_result['summary']['total']} successful",
            results=batch_result["results"],
            summary=batch_result["summary"]
        )
        
    except Exception as e:
        logger.error(f"Failed to execute batch crawl: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/search")
async def cache_search(
    request: CacheSearchRequest,
    customer_id: Optional[str] = None,
    user_email: Optional[str] = Depends(get_optional_user_email)
):
    customer_identifier = get_customer_identifier(customer_id, user_email)
    store = RemoteCacheStore(customer_identifier)
    matches = store.search(
        query=request.query,
        domain=request.domain,
        url_prefix=request.url_prefix,
        min_similarity=request.min_similarity,
        max_results=request.max_results,
        quality_in=request.quality_in or ["sufficient"],
        since_ts=request.since_ts
    )
    return {
        "success": True,
        "matches": matches,
        "count": len(matches),
        "query": request.query,
        "min_similarity": request.min_similarity
    }


@router.get("/cache/list")
async def cache_list(
    domain: Optional[str] = None,
    quality: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    customer_id: Optional[str] = None,
    user_email: Optional[str] = Depends(get_optional_user_email)
):
    customer_identifier = get_customer_identifier(customer_id, user_email)
    store = RemoteCacheStore(customer_identifier)
    result = store.list_docs(domain=domain, quality=quality, limit=limit, offset=offset)
    return {
        "success": True,
        "docs": result["docs"],
        "count": len(result["docs"]),
        "limit": result["limit"],
        "offset": result["offset"]
    }


@router.get("/cache/doc/{doc_id}")
async def cache_get_doc(
    doc_id: str,
    customer_id: Optional[str] = None,
    user_email: Optional[str] = Depends(get_optional_user_email)
):
    customer_identifier = get_customer_identifier(customer_id, user_email)
    store = RemoteCacheStore(customer_identifier)
    doc = store.get_doc(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "success": True,
        "doc_id": doc.get("doc_id"),
        "url": doc.get("url"),
        "markdown": doc.get("markdown"),
        "quality": doc.get("quality"),
        "char_count": doc.get("char_count", 0),
        "updated_at": doc.get("updated_at"),
        "content_hash": doc.get("content_hash"),
        "source_status": doc.get("source_status"),
    }


@router.post("/cache/upsert")
async def cache_upsert(
    request: CacheUpsertRequest,
    customer_id: Optional[str] = None,
    user_email: Optional[str] = Depends(get_optional_user_email)
):
    customer_identifier = get_customer_identifier(customer_id, user_email)
    store = RemoteCacheStore(customer_identifier)
    upserted = store.upsert(
        url=request.url,
        markdown=request.markdown,
        markdown_plain=request.markdown_plain,
        content=request.content,
        quality=request.quality,
        status_code=request.status_code,
        extractor_version=request.extractor_version or "",
        normalized_url=request.normalized_url,
        content_hash=request.content_hash,
        metadata=request.metadata
    )
    return {"success": True, "doc": upserted}


@router.post("/cache/prune")
async def cache_prune(
    request: CachePruneRequest,
    customer_id: Optional[str] = None,
    user_email: Optional[str] = Depends(get_optional_user_email)
):
    customer_identifier = get_customer_identifier(customer_id, user_email)
    store = RemoteCacheStore(customer_identifier)
    result = store.prune(domain=request.domain, ttl_hours=request.ttl_hours, dry_run=request.dry_run)
    return {"success": True, **result}


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(
    job_id: str,
    user_email: str = Depends(get_user_email)
):
    """
    Get status and results for a specific job
    """
    try:
        # TODO: Implement actual job status retrieval
        # For now, return a mock response for Phase 1
        
        return JobStatus(
            job_id=job_id,
            status="completed",
            progress=1.0,
            total_urls=3,
            completed_urls=3,
            results=[
                CrawlResult(
                    success=True,
                    url="https://example.com",
                    html="<html><body>Mock content</body></html>",
                    markdown="# Mock Content",
                    metadata={"title": "Example"},
                    crawled_at=datetime.utcnow()
                )
            ],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to get job status for {job_id}: {e}")
        raise HTTPException(status_code=404, detail="Job not found")


@router.get("/jobs", response_model=JobListResponse)
async def list_user_jobs(
    user_email: str = Depends(get_user_email)
):
    """
    List all jobs for the authenticated user
    """
    try:
        # TODO: Implement actual job listing
        # For now, return a mock response for Phase 1
        
        mock_jobs = [
            {
                "job_id": "mock-job-1",
                "status": "completed",
                "total_urls": 5,
                "completed_urls": 5,
                "created_at": datetime.utcnow().isoformat()
            },
            {
                "job_id": "mock-job-2", 
                "status": "running",
                "total_urls": 10,
                "completed_urls": 7,
                "created_at": datetime.utcnow().isoformat()
            }
        ]
        
        return JobListResponse(
            jobs=mock_jobs,
            total=len(mock_jobs)
        )
        
    except Exception as e:
        logger.error(f"Failed to list jobs for user {user_email}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/sessions/{session_id}/files")
async def list_session_files(
    session_id: str, 
    prefix: Optional[str] = None, 
    customer_id: Optional[str] = None,
    user_email: Optional[str] = Depends(get_optional_user_email)
):
    """List files stored for a session. Optional prefix (e.g., 'results', 'screenshots')."""
    customer_identifier = get_customer_identifier(customer_id, user_email)
    storage = CrawlStorageService(customer_identifier)
    try:
        pref = prefix or ''
        files = await storage.list_files(pref or '', session_id)
        
        # Add storage path info for debugging
        storage_path = None
        if hasattr(storage, '_storage_root'):
            storage_path = str(storage.get_session_path(session_id))
        
        return {
            "session_id": session_id, 
            "prefix": pref, 
            "files": files,
            "storage_info": {
                "customer_identifier": customer_identifier,
                "customer_hash": storage._user_hash,
                "storage_path": storage_path,
                "is_cloud": storage._is_cloud
            }
        }
    except Exception as e:
        logger.error(f"Failed to list session files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug/storage")
async def debug_storage(
    customer_id: Optional[str] = None,
    user_email: Optional[str] = Depends(get_optional_user_email)
):
    """Debug endpoint to inspect storage structure and contents."""
    import os
    from pathlib import Path
    
    customer_identifier = get_customer_identifier(customer_id, user_email)
    storage = CrawlStorageService(customer_identifier)
    
    result = {
        "customer_identifier": customer_identifier,
        "customer_hash": storage._user_hash,
        "is_cloud": storage._is_cloud,
        "storage_root": str(storage._storage_root) if hasattr(storage, '_storage_root') else "N/A"
    }
    
    # List all directories and files in storage
    if hasattr(storage, '_storage_root'):
        storage_root = storage._storage_root
        customer_path = storage_root / storage._user_hash
        
        result["customer_path"] = str(customer_path)
        result["customer_path_exists"] = customer_path.exists()
        
        if customer_path.exists():
            # List all sessions
            sessions = []
            for session_dir in customer_path.iterdir():
                if session_dir.is_dir():
                    session_info = {
                        "session_id": session_dir.name,
                        "path": str(session_dir),
                        "files": []
                    }
                    
                    # List files recursively
                    for root, dirs, files in os.walk(session_dir):
                        for file in files:
                            file_path = Path(root) / file
                            session_info["files"].append({
                                "name": file,
                                "relative_path": str(file_path.relative_to(session_dir)),
                                "size": file_path.stat().st_size
                            })
                    
                    sessions.append(session_info)
            
            result["sessions"] = sessions
            result["total_sessions"] = len(sessions)
    
    return result


@router.get("/sessions/{session_id}/file")
async def get_session_file(
    session_id: str, 
    path: str, 
    customer_id: Optional[str] = None,
    user_email: Optional[str] = Depends(get_optional_user_email)
):
    """Fetch a stored file for a session by relative path (e.g., 'results/abc.json')."""
    customer_identifier = get_customer_identifier(customer_id, user_email)
    storage = CrawlStorageService(customer_identifier)
    try:
        data = await storage.get_file(path, session_id)
        # Best-effort content type based on extension
        if path.endswith('.json'):
            return Response(content=data, media_type='application/json')
        elif path.endswith('.png'):
            return Response(content=data, media_type='image/png')
        elif path.endswith('.txt') or path.endswith('.md'):
            return Response(content=data, media_type='text/plain')
        return Response(content=data, media_type='application/octet-stream')
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(f"Failed to fetch session file '{path}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Catch-all inside the /api router — returns endpoint hint instead of bare 404.
# Starlette's Mount intercepts /api/* before the app-level catch-all can fire.
_API_ENDPOINT_HINT = {
    "endpoints": {
        "crawl":    "POST /api/crawl          {url, options?}",
        "markdown": "POST /api/markdown       {url, options?}",
        "batch":    "POST /api/batch          {urls: [...]}",
        "jobs":     "POST /api/jobs/crawl     {url, session_id?}",
        "agent":    "POST /api/agent/run      {task, config?}",
        "tools":    "GET  /tools              list available AHP tools",
        "view":     "GET  /view?url=...       render page as HTML",
        "download": "GET  /download?url=...   fetch binary/file",
        "health":   "GET  /health",
        "docs":     "GET  /docs",
    },
    "hint": "All POST endpoints accept JSON. See /docs for full schema.",
}

@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def api_not_found(path: str, request: Request):
    return JSONResponse(
        status_code=404,
        content={
            "error": "http_error",
            "status": 404,
            "details": {"message": f"Not Found: /api/{path}"},
            "grub": _API_ENDPOINT_HINT,
        },
    )
