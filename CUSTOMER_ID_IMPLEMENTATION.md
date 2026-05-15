# Customer ID Implementation

## Overview
Added support for optional `customer_id` field in API requests while maintaining backward compatibility with existing authentication-based user identification.

## Changes Made

### 1. Models (`app/models.py`)
Added `customer_id: Optional[str] = None` to all request models:
- `CrawlRequest`
- `MarkdownRequest`
- `BatchRequest`

### 2. Authentication (`app/auth.py`)
Added new helper function:
```python
def get_customer_identifier(customer_id: Optional[str] = None, user_email: Optional[str] = None) -> str
```
This function resolves the customer identifier by:
1. First checking if `customer_id` is provided (highest priority)
2. Falling back to `user_email` from authentication
3. Defaulting to `"anonymous@gnosis-crawl.local"` if neither is available

### 3. Routes (`app/routes.py`)
Created `get_optional_user_email()` dependency that:
- Returns `None` when auth is disabled (`settings.disable_auth=True`)
- Returns `None` when no auth header is provided
- Returns user email when valid token is provided
- Never raises auth errors (allows unauthenticated access)

Updated all route handlers:
- `/crawl` - Single URL crawl
- `/markdown` - Markdown-only crawl
- `/batch` - Batch crawl
- `/sessions/{session_id}/files` - List session files
- `/sessions/{session_id}/file` - Get session file

All routes now:
- Accept optional `customer_id` in request body (for POST) or query param (for GET)
- Use `get_optional_user_email()` instead of `get_user_email()`
- Call `get_customer_identifier(request.customer_id, user_email)` to resolve identifier
- Use `customer_identifier` for storage operations
- Include `customer_identifier` in response metadata

## Usage Examples

### With Authentication (Existing Behavior)
```bash
curl -X POST https://api.example.com/crawl \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "options": {"javascript": true}
  }'
```
Storage path: `storage/{hash(user_email)}/{session_id}/`

### Without Authentication (New Behavior)
```bash
curl -X POST https://api.example.com/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "customer_id": "client-xyz-123",
    "options": {"javascript": true}
  }'
```
Storage path: `storage/{hash(client-xyz-123)}/{session_id}/`

### With Customer ID Override (Even When Authenticated)
```bash
curl -X POST https://api.example.com/crawl \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "customer_id": "custom-partition-id",
    "options": {"javascript": true}
  }'
```
Storage path: `storage/{hash(custom-partition-id)}/{session_id}/`
Note: customer_id takes priority over authenticated user email

### Session File Access Without Auth
```bash
# List files
curl "https://api.example.com/sessions/{session_id}/files?customer_id=client-xyz-123"

# Get file
curl "https://api.example.com/sessions/{session_id}/file?path=results/abc123.json&customer_id=client-xyz-123"
```

## Backward Compatibility

✅ **Fully backward compatible** - All existing functionality preserved:
- Authenticated requests work exactly as before
- Token validation unchanged
- Auth middleware unchanged
- Storage path structure unchanged
- When auth is enabled and no `customer_id` is provided, behavior is identical to previous version

## Configuration

For self-hosted/unauthenticated deployments, set `DISABLE_AUTH=true` in your `.env`:
```env
DISABLE_AUTH=true
```

This allows requests without auth tokens to succeed using only `customer_id` for storage partitioning.

## Response Changes

Metadata now includes `customer_identifier` instead of `user_email`:
```json
{
  "success": true,
  "url": "https://example.com",
  "metadata": {
    "customer_identifier": "client-xyz-123",
    "session_id": "uuid-here",
    ...
  }
}
```

## Storage Behavior

The customer identifier (from either `customer_id` or `user_email`) is:
1. Hashed to 12 characters using SHA256
2. Used as the top-level storage directory
3. Provides privacy and consistent bucketing per customer
4. Works identically for both authenticated and unauthenticated access

## Testing

To test unauthenticated access:
1. Set `DISABLE_AUTH=true` in environment
2. Send requests with `customer_id` field
3. Verify storage appears in `storage/{hash(customer_id)}/` directory
4. Verify responses include `customer_identifier` in metadata
