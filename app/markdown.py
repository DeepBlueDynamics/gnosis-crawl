"""
Markdown generation engine for Grub Crawler
Ported from gnosis-wraith with enhanced filtering and processing
"""
import re
import urllib.parse as urlparse
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from bs4 import BeautifulSoup, Tag, NavigableString, Comment
import logging

logger = logging.getLogger(__name__)

# Try to import the Rust native extension for fast markdown generation
try:
    from grub_md import generate_markdown as _rust_generate
    _HAS_RUST = True
    logger.info("grub_md Rust extension loaded — using fast path")
except ImportError:
    _HAS_RUST = False

# Pre-compile regex patterns
LINK_PATTERN = re.compile(r'!?\[([^\]]+)\]\(([^)]+?)(?:\s+"([^"]*)")?\)')


@dataclass
class MarkdownResult:
    """Result of markdown generation process."""
    raw_markdown: str
    markdown_with_citations: str
    references_markdown: str
    clean_markdown: str = ""
    markdown_references: str = ""
    markdown_plain: str = ""
    links: List[Dict[str, Any]] = None
    images: List[Dict[str, Any]] = None
    urls: List[str] = None
    
    def __post_init__(self):
        if self.links is None:
            self.links = []
        if self.images is None:
            self.images = []
        if self.urls is None:
            self.urls = []
    
    def __str__(self):
        return self.clean_markdown or self.raw_markdown


class HTMLToMarkdownConverter:
    """Convert HTML to Markdown with enhanced filtering."""
    
    def __init__(self, base_url: str = "", dedupe_tables: bool = True):
        self.base_url = base_url
        self.dedupe_tables = dedupe_tables
        self._layout_table_depth = 0
        self.ignore_links = False
        self.ignore_images = False
        self.ignore_emphasis = False
        self.single_line_break = True
        self.mark_code = True
        
    def convert(self, html: str) -> str:
        """Convert HTML to markdown."""
        if not html or not html.strip():
            return ""
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements
            self._remove_unwanted_elements(soup)
            
            # Process the soup
            markdown = self._process_element(soup)
            
            # Clean up the markdown
            markdown = self._clean_markdown(markdown)
            
            return markdown
            
        except Exception as e:
            logger.error(f"Error converting HTML to markdown: {e}")
            return f"Error converting content: {str(e)}"
    
    def _remove_unwanted_elements(self, soup: BeautifulSoup) -> None:
        """Remove script, style, and other unwanted elements."""
        unwanted_tags = [
            'script', 'style', 'noscript', 'iframe', 'object', 'embed',
            'form', 'input', 'button', 'select', 'textarea'
        ]
        
        for tag_name in unwanted_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Remove common hidden/a11y-only patterns that are frequently abused for prompt injection.
        hidden_selectors = [
            '.sr-only', '.sr_only', '.srOnly',
            '.visually-hidden', '.visually_hidden',
            '.screen-reader-only', '.screen_reader_only',
            '.a11y-only', '.a11y_only',
            '[hidden]',
        ]
        for selector in hidden_selectors:
            for element in soup.select(selector):
                element.decompose()
    
    def _process_element(self, element) -> str:
        """Process a BeautifulSoup element and return markdown."""
        if isinstance(element, NavigableString):
            return str(element).strip()
        
        if not isinstance(element, Tag):
            return ""
        
        tag_name = element.name.lower()
        text = ""
        
        # Handle different HTML tags
        if tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            level = int(tag_name[1])
            content = self._get_text_content(element)
            text = f"\n{'#' * level} {content}\n\n"
            
        elif tag_name == 'p':
            content = self._process_children(element)
            text = f"{content}\n\n"
            
        elif tag_name == 'br':
            text = "\n"
            
        elif tag_name in ['strong', 'b']:
            if not self.ignore_emphasis:
                content = self._process_children(element)
                text = f"**{content}**"
            else:
                text = self._process_children(element)
                
        elif tag_name in ['em', 'i']:
            if not self.ignore_emphasis:
                content = self._process_children(element)
                text = f"*{content}*"
            else:
                text = self._process_children(element)
                
        elif tag_name == 'a':
            text = self._process_link(element)
            
        elif tag_name == 'img':
            text = self._process_image(element)
            
        elif tag_name in ['ul', 'ol']:
            text = self._process_list(element, ordered=(tag_name == 'ol'))
            
        elif tag_name == 'li':
            content = self._process_children(element)
            text = f"- {content}\n"
            
        elif tag_name == 'blockquote':
            content = self._process_children(element)
            lines = content.split('\n')
            quoted_lines = [f"> {line}" for line in lines if line.strip()]
            text = '\n'.join(quoted_lines) + '\n\n'
            
        elif tag_name in ['code', 'tt']:
            if self.mark_code:
                content = self._get_text_content(element)
                text = f"`{content}`"
            else:
                text = self._get_text_content(element)
                
        elif tag_name == 'pre':
            content = self._get_text_content(element)
            if self.mark_code:
                text = f"```\n{content}\n```\n\n"
            else:
                text = f"{content}\n\n"
                
        elif tag_name in ['div', 'section', 'article', 'main', 'aside', 'header', 'footer', 'nav']:
            text = self._process_children(element)
            
        elif tag_name in ['table', 'thead', 'tbody', 'tfoot']:
            text = self._process_table(element)
            
        elif tag_name == 'tr':
            if self.dedupe_tables and self._layout_table_depth > 0:
                return self._process_children(element)
            cells = []
            for cell in element.find_all(['td', 'th']):
                cells.append(self._get_text_content(cell))
            text = "| " + " | ".join(cells) + " |\n"
            
        else:
            # Default: process children
            text = self._process_children(element)
        
        return text
    
    def _process_children(self, element) -> str:
        """Process all children of an element."""
        result = ""
        for child in element.children:
            result += self._process_element(child)
        return result
    
    def _get_text_content(self, element) -> str:
        """Get clean text content from element."""
        if isinstance(element, NavigableString):
            return str(element).strip()
        
        text = ""
        for child in element.children:
            if isinstance(child, NavigableString):
                text += str(child)
            else:
                text += self._get_text_content(child)
        
        return ' '.join(text.split())  # Normalize whitespace
    
    def _process_link(self, element) -> str:
        """Process an anchor tag."""
        if self.ignore_links:
            return self._get_text_content(element)
        
        href = element.get('href', '')
        text = self._get_text_content(element)
        
        if not href or not text:
            return text
        
        # Resolve relative URLs
        if self.base_url and href:
            href = urlparse.urljoin(self.base_url, href)
        
        return f"[{text}]({href})"
    
    def _process_image(self, element) -> str:
        """Process an image tag."""
        if self.ignore_images:
            return ""
        
        src = element.get('src', '')
        alt = element.get('alt', 'Image')
        title = element.get('title', '')
        
        if not src:
            return ""
        
        # Resolve relative URLs
        if self.base_url and src:
            src = urlparse.urljoin(self.base_url, src)
        
        if title:
            return f'![{alt}]({src} "{title}")'
        else:
            return f'![{alt}]({src})'
    
    def _process_list(self, element, ordered: bool = False) -> str:
        """Process ul/ol list."""
        result = ""
        counter = 1
        
        for li in element.find_all('li', recursive=False):
            content = self._process_children(li).strip()
            if ordered:
                result += f"{counter}. {content}\n"
                counter += 1
            else:
                result += f"- {content}\n"
        
        return result + "\n"
    
    def _process_table(self, element) -> str:
        """Process table element - skip layout tables."""
        # Heuristics to detect layout tables
        has_nested_table = element.find('table') is not None

        rows = element.find_all('tr', recursive=False)
        if not rows:
            sections = element.find_all(['thead', 'tbody', 'tfoot'], recursive=False)
            if sections:
                rows = []
                for section in sections:
                    rows.extend(section.find_all('tr', recursive=False))
        if not rows:
            if has_nested_table:
                return self._process_children(element)
            return ""

        first_row = rows[0]
        first_row_cells = first_row.find_all(['td', 'th'], recursive=False)

        # Cells with block-level children usually mean layout usage
        block_like_tags = ['div', 'p', 'ul', 'ol', 'table', 'article', 'section', 'header', 'footer', 'nav', 'aside']
        has_block_children = False
        for cell in first_row_cells:
            if cell.find(block_like_tags):
                has_block_children = True
                break

        # Few columns but many rows often indicates layout/spacer tables
        looks_like_layout = bool(first_row_cells and len(first_row_cells) <= 2 and len(rows) >= 15)

        if has_nested_table or has_block_children or looks_like_layout:
            if self.dedupe_tables:
                self._layout_table_depth += 1
                try:
                    return self._process_children(element)
                finally:
                    self._layout_table_depth -= 1
            return self._process_children(element)
        
        # Actual data table - convert to markdown
        markdown_rows = []
        for row in rows:
            cells = []
            for cell in row.find_all(['td', 'th'], recursive=False):
                cells.append(self._get_text_content(cell))
            if cells:
                markdown_rows.append("| " + " | ".join(cells) + " |")
        
        if not markdown_rows:
            return ""
        
        if first_row and first_row.find('th', recursive=False):
            header_sep = "| " + " | ".join(["---"] * len(first_row_cells)) + " |"
            markdown_rows.insert(1, header_sep)
        
        return "\n".join(markdown_rows) + "\n\n"
    
    def _clean_markdown(self, markdown: str) -> str:
        """Clean up generated markdown."""
        if not markdown:
            return ""
        
        # Remove excessive whitespace
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        markdown = re.sub(r' +', ' ', markdown)
        
        # Clean up list formatting
        markdown = re.sub(r'\n- \n', '\n', markdown)
        markdown = re.sub(r'\n\d+\. \n', '\n', markdown)
        
        return markdown.strip()


class ContentFilter:
    """Filter and clean HTML content for better markdown generation."""
    
    def __init__(self, readability_threshold: float = 0.3):
        self.readability_threshold = readability_threshold
    
    def filter_content(self, html: str) -> str:
        """Apply content filtering to HTML."""
        if not html:
            return ""
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove navigation, ads, etc.
            self._remove_navigation_elements(soup)
            
            # Find main content area
            main_content = self._extract_main_content(soup)
            
            return str(main_content) if main_content else str(soup)
            
        except Exception as e:
            logger.error(f"Error filtering content: {e}")
            return html
    
    def _remove_navigation_elements(self, soup: BeautifulSoup) -> None:
        """Remove navigation, ads, and other non-content elements."""
        # Common non-content selectors
        unwanted_selectors = [
            'nav', 'header', 'footer', 'aside', '.nav', '.navigation',
            '.sidebar', '.menu', '.ads', '.advertisement', '.social',
            '.share', '.comments', '.related', '.popup', '.modal'
        ]

        # Hidden/a11y-only patterns are frequently abused for prompt injection.
        # These are safe to remove for summarization-oriented extraction.
        unwanted_selectors.extend([
            '.sr-only', '.sr_only', '.srOnly',
            '.visually-hidden', '.visually_hidden',
            '.screen-reader-only', '.screen_reader_only',
            '.a11y-only', '.a11y_only',
            '[hidden]',
        ])
        
        for selector in unwanted_selectors:
            for element in soup.select(selector):
                element.decompose()
    
    def _extract_main_content(self, soup: BeautifulSoup):
        """Try to extract the main content area."""
        # Look for common main content selectors
        main_selectors = [
            'main', 'article', '.content', '.main-content',
            '.post-content', '.entry-content', '#content', '#main'
        ]
        
        for selector in main_selectors:
            main_element = soup.select_one(selector)
            if main_element:
                return main_element
        
        # Fallback: return body
        return soup.find('body') or soup


class MarkdownGenerator:
    """Main markdown generation class."""
    
    def __init__(self, content_filter: Optional[ContentFilter] = None):
        self.content_filter = content_filter or ContentFilter()
    
    def generate_markdown(
        self,
        html: str,
        base_url: str = "",
        dedupe_tables: bool = True
    ) -> MarkdownResult:
        """Generate comprehensive markdown result from HTML."""
        if not html or not html.strip():
            return MarkdownResult(
                raw_markdown="",
                markdown_with_citations="",
                references_markdown="",
                clean_markdown=""
            )

        # Fast path: use Rust native extension if available
        if _HAS_RUST:
            try:
                d = _rust_generate(html, base_url, dedupe_tables)
                return MarkdownResult(**d)
            except Exception as e:
                logger.warning(f"Rust grub_md failed, falling back to Python: {e}")

        try:
            # Filter content for better extraction
            filtered_html = self.content_filter.filter_content(html)
            
            # Convert to markdown
            converter = HTMLToMarkdownConverter(
                base_url=base_url,
                dedupe_tables=dedupe_tables
            )
            raw_markdown = converter.convert(filtered_html)

            if self._should_fallback(html, raw_markdown, base_url):
                logger.info("Markdown fallback triggered; retrying without filtering")
                fallback_converter = HTMLToMarkdownConverter(
                    base_url=base_url,
                    dedupe_tables=dedupe_tables
                )
                raw_markdown = fallback_converter.convert(html)
            
            # Extract links and generate citations
            links, markdown_with_citations = self._extract_links_and_generate_citations(
                raw_markdown, base_url
            )
            
            # Generate references section
            references_markdown = self._generate_references_section(links)
            
            # Generate different markdown variants
            clean_markdown = self._clean_markdown_for_readability(raw_markdown)
            markdown_plain = self._strip_links_from_markdown(raw_markdown)
            
            # Extract images
            images = self._extract_images_from_markdown(raw_markdown)
            
            # Extract all URLs
            urls = [link['url'] for link in links if 'url' in link]
            
            return MarkdownResult(
                raw_markdown=raw_markdown,
                markdown_with_citations=markdown_with_citations,
                references_markdown=references_markdown,
                clean_markdown=clean_markdown,
                markdown_references=markdown_with_citations + "\n\n" + references_markdown,
                markdown_plain=markdown_plain,
                links=links,
                images=images,
                urls=urls
            )
            
        except Exception as e:
            logger.error(f"Error generating markdown: {e}")
            return MarkdownResult(
                raw_markdown=f"Error generating markdown: {str(e)}",
                markdown_with_citations="",
                references_markdown="",
                clean_markdown=""
            )

    def _should_fallback(self, html: str, markdown: str, base_url: str) -> bool:
        """Detect when markdown extraction is too sparse and retry without filtering."""
        html_len = len(html or "")
        md_len = len(markdown or "")
        if md_len == 0:
            return True
        if html_len < 5000:
            return False
        if md_len < 400:
            return True
        if md_len / max(html_len, 1) < 0.01:
            return True
        if "news.ycombinator.com" in (base_url or "") and "item?id=" not in markdown:
            return True
        return False
    
    def _extract_links_and_generate_citations(self, markdown: str, base_url: str) -> Tuple[List[Dict], str]:
        """Extract links and replace with numbered citations."""
        links = []
        citation_counter = 1
        result_markdown = markdown
        
        for match in LINK_PATTERN.finditer(markdown):
            link_text = match.group(1)
            link_url = match.group(2)
            link_title = match.group(3) or ""
            
            # Resolve relative URLs
            if base_url and link_url:
                full_url = urlparse.urljoin(base_url, link_url)
            else:
                full_url = link_url
            
            # Store link info
            link_info = {
                'text': link_text,
                'url': full_url,
                'title': link_title,
                'citation_number': citation_counter
            }
            links.append(link_info)
            
            # Replace with citation
            full_match = match.group(0)
            citation = f"{link_text}[{citation_counter}]"
            result_markdown = result_markdown.replace(full_match, citation, 1)
            
            citation_counter += 1
        
        return links, result_markdown
    
    def _generate_references_section(self, links: List[Dict]) -> str:
        """Generate a references section from extracted links."""
        if not links:
            return ""
        
        references = ["## References\n"]
        for link in links:
            ref_line = f"[{link['citation_number']}]: {link['url']}"
            if link.get('title'):
                ref_line += f' "{link["title"]}"'
            references.append(ref_line)
        
        return "\n".join(references)
    
    def _clean_markdown_for_readability(self, markdown: str) -> str:
        """Clean markdown for better readability."""
        if not markdown:
            return ""
        
        # Remove excessive empty lines
        cleaned = re.sub(r'\n{3,}', '\n\n', markdown)
        
        # Remove empty list items
        cleaned = re.sub(r'\n- \n', '\n', cleaned)
        
        # Clean up spacing around headers
        cleaned = re.sub(r'\n+(#{1,6})', r'\n\n\1', cleaned)
        cleaned = re.sub(r'(#{1,6}.*)\n+', r'\1\n\n', cleaned)
        
        return cleaned.strip()
    
    def _strip_links_from_markdown(self, markdown: str) -> str:
        """Remove all links from markdown, keeping only text."""
        # Replace [text](url) with text
        plain = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', markdown)
        
        # Replace ![alt](url) with alt
        plain = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', plain)
        
        return plain
    
    def _extract_images_from_markdown(self, markdown: str) -> List[Dict]:
        """Extract image information from markdown."""
        images = []
        image_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)(?:\s+"([^"]*)")?\)')
        
        for match in image_pattern.finditer(markdown):
            images.append({
                'alt': match.group(1),
                'url': match.group(2),
                'title': match.group(3) or ""
            })
        
        return images
