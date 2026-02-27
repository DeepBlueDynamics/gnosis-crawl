use ego_tree::NodeId;
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use regex::Regex;
use scraper::{ElementRef, Html, Node, Selector};
use std::collections::HashSet;
use url::Url;

// ---------------------------------------------------------------------------
// Selectors (compiled once)
// ---------------------------------------------------------------------------

macro_rules! sel {
    ($s:expr) => {
        Selector::parse($s).expect(concat!("bad selector: ", $s))
    };
}

static SEL_MAIN: Lazy<Vec<Selector>> = Lazy::new(|| {
    vec![
        sel!("main"),
        sel!("article"),
        sel!(".content"),
        sel!(".main-content"),
        sel!(".post-content"),
        sel!(".entry-content"),
        sel!("#content"),
        sel!("#main"),
        sel!("body"),
    ]
});

static SEL_TABLE: Lazy<Selector> = Lazy::new(|| sel!("table"));
static SEL_TR: Lazy<Selector> = Lazy::new(|| sel!("tr"));
static SEL_THEAD_TBODY_TFOOT: Lazy<Selector> = Lazy::new(|| sel!("thead, tbody, tfoot"));
static SEL_TD_TH: Lazy<Selector> = Lazy::new(|| sel!("td, th"));
static SEL_LI: Lazy<Selector> = Lazy::new(|| sel!("li"));

/// Tags whose entire subtree we skip.
const SKIP_TAGS: &[&str] = &[
    "script", "style", "noscript", "iframe", "object", "embed", "form", "input", "button",
    "select", "textarea",
];

/// Nav / clutter tags to remove during content filtering.
const NAV_TAGS: &[&str] = &["nav", "header", "footer", "aside"];

/// Nav / clutter CSS classes to remove.
const NAV_CLASSES: &[&str] = &[
    "nav",
    "navigation",
    "sidebar",
    "menu",
    "ads",
    "advertisement",
    "social",
    "share",
    "comments",
    "related",
    "popup",
    "modal",
];

/// Hidden / a11y-only CSS classes to remove.
const HIDDEN_CLASSES: &[&str] = &[
    "sr-only",
    "sr_only",
    "srOnly",
    "visually-hidden",
    "visually_hidden",
    "screen-reader-only",
    "screen_reader_only",
    "a11y-only",
    "a11y_only",
];

/// Block-level tags that signal a table cell is used for layout.
const BLOCK_LIKE_TAGS: &[&str] = &[
    "div", "p", "ul", "ol", "table", "article", "section", "header", "footer", "nav", "aside",
];

// ---------------------------------------------------------------------------
// Link / image regex for the citation pass (matches Python's LINK_PATTERN)
// ---------------------------------------------------------------------------

static RE_LINK: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"!?\[([^\]]+)\]\(([^)]+?)(?:\s+"([^"]*)")?\)"#).unwrap());

static RE_IMAGE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"!\[([^\]]*)\]\(([^)]+?)(?:\s+"([^"]*)")?\)"#).unwrap());

// ---------------------------------------------------------------------------
// Collected link / image structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct LinkInfo {
    text: String,
    url: String,
    title: String,
    citation_number: usize,
}

#[derive(Debug, Clone)]
struct ImageInfo {
    alt: String,
    url: String,
    title: String,
}

// ---------------------------------------------------------------------------
// Helper: should an element be skipped entirely?
// ---------------------------------------------------------------------------

fn should_skip(el: &ElementRef) -> bool {
    let tag = el.value().name();

    // Skip tags
    if SKIP_TAGS.contains(&tag) {
        return true;
    }

    // Hidden attribute
    if el.value().attr("hidden").is_some() {
        return true;
    }

    // Hidden / a11y-only classes
    if let Some(cls_attr) = el.value().attr("class") {
        for cls in cls_attr.split_whitespace() {
            if HIDDEN_CLASSES.contains(&cls) {
                return true;
            }
        }
    }

    false
}

/// Check if an element is nav/clutter that should be removed during content
/// filtering (before main-content detection).
fn is_nav_clutter(el: &ElementRef) -> bool {
    let tag = el.value().name();
    if NAV_TAGS.contains(&tag) {
        return true;
    }
    if let Some(cls_attr) = el.value().attr("class") {
        for cls in cls_attr.split_whitespace() {
            if NAV_CLASSES.contains(&cls) {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Resolve a potentially-relative URL against a base.
// ---------------------------------------------------------------------------

fn resolve_url(href: &str, base: &Option<Url>) -> String {
    if href.is_empty() {
        return String::new();
    }
    if let Some(base_url) = base {
        match base_url.join(href) {
            Ok(u) => u.to_string(),
            Err(_) => href.to_string(),
        }
    } else {
        href.to_string()
    }
}

// ---------------------------------------------------------------------------
// Core tree-walk: emit markdown into a buffer
// ---------------------------------------------------------------------------

struct Walker<'a> {
    base_url: Option<Url>,
    dedupe_tables: bool,
    layout_table_depth: usize,
    /// Set of node IDs that belong to nav/clutter elements (pre-computed).
    skip_ids: &'a HashSet<NodeId>,
}

impl<'a> Walker<'a> {
    fn walk(&mut self, el: ElementRef, buf: &mut String) {
        // Skip entirely?
        if should_skip(&el) {
            return;
        }
        if self.skip_ids.contains(&el.id()) {
            return;
        }

        let tag = el.value().name();

        match tag {
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                let level = tag.as_bytes()[1] - b'0';
                let text = get_text_content(&el);
                if !text.is_empty() {
                    buf.push('\n');
                    for _ in 0..level {
                        buf.push('#');
                    }
                    buf.push(' ');
                    buf.push_str(&text);
                    buf.push_str("\n\n");
                }
            }
            "p" => {
                let start = buf.len();
                self.walk_children(&el, buf);
                if buf.len() > start {
                    buf.push_str("\n\n");
                }
            }
            "br" => {
                buf.push('\n');
            }
            "strong" | "b" => {
                let content = self.children_to_string(&el);
                if !content.is_empty() {
                    buf.push_str("**");
                    buf.push_str(&content);
                    buf.push_str("**");
                }
            }
            "em" | "i" => {
                let content = self.children_to_string(&el);
                if !content.is_empty() {
                    buf.push('*');
                    buf.push_str(&content);
                    buf.push('*');
                }
            }
            "a" => {
                self.handle_link(&el, buf);
            }
            "img" => {
                self.handle_image(&el, buf);
            }
            "ul" => {
                self.handle_list(&el, false, buf);
            }
            "ol" => {
                self.handle_list(&el, true, buf);
            }
            "li" => {
                // Only reached if <li> appears outside <ul>/<ol>
                let content = self.children_to_string(&el);
                let trimmed = content.trim();
                if !trimmed.is_empty() {
                    buf.push_str("- ");
                    buf.push_str(trimmed);
                    buf.push('\n');
                }
            }
            "blockquote" => {
                let content = self.children_to_string(&el);
                for line in content.lines() {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        buf.push_str("> ");
                        buf.push_str(trimmed);
                        buf.push('\n');
                    }
                }
                buf.push('\n');
            }
            "code" | "tt" => {
                // If inside <pre>, don't add backticks (pre handles it)
                if el
                    .parent()
                    .and_then(|p| p.value().as_element())
                    .map_or(false, |p| p.name() == "pre")
                {
                    let text = get_text_content(&el);
                    buf.push_str(&text);
                } else {
                    let text = get_text_content(&el);
                    if !text.is_empty() {
                        buf.push('`');
                        buf.push_str(&text);
                        buf.push('`');
                    }
                }
            }
            "pre" => {
                let text = get_raw_text(&el);
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    buf.push_str("```\n");
                    buf.push_str(trimmed);
                    buf.push_str("\n```\n\n");
                }
            }
            "table" | "thead" | "tbody" | "tfoot" => {
                self.handle_table(&el, buf);
            }
            "tr" => {
                if self.dedupe_tables && self.layout_table_depth > 0 {
                    self.walk_children(&el, buf);
                } else {
                    let cells = direct_children_by_sel(&el, &SEL_TD_TH);
                    if !cells.is_empty() {
                        buf.push_str("| ");
                        for (i, cell) in cells.iter().enumerate() {
                            if i > 0 {
                                buf.push_str(" | ");
                            }
                            buf.push_str(&get_text_content(cell));
                        }
                        buf.push_str(" |\n");
                    }
                }
            }
            // Container elements — just recurse
            _ => {
                self.walk_children(&el, buf);
            }
        }
    }

    fn walk_children(&mut self, el: &ElementRef, buf: &mut String) {
        for child in el.children() {
            match child.value() {
                Node::Element(_) => {
                    if let Some(child_el) = ElementRef::wrap(child) {
                        self.walk(child_el, buf);
                    }
                }
                Node::Text(t) => {
                    let s = t.text.trim();
                    if !s.is_empty() {
                        buf.push_str(s);
                    }
                }
                _ => {}
            }
        }
    }

    /// Walk children into a temporary String (used for inline contexts).
    fn children_to_string(&mut self, el: &ElementRef) -> String {
        let mut tmp = String::new();
        self.walk_children(el, &mut tmp);
        tmp
    }

    fn handle_link(&mut self, el: &ElementRef, buf: &mut String) {
        let text = get_text_content(el);
        let href = el.value().attr("href").unwrap_or("");
        if text.is_empty() && href.is_empty() {
            return;
        }
        if text.is_empty() || href.is_empty() {
            // Just emit the text (or nothing)
            buf.push_str(&text);
            return;
        }
        let resolved = resolve_url(href, &self.base_url);
        buf.push('[');
        buf.push_str(&text);
        buf.push_str("](");
        buf.push_str(&resolved);
        buf.push(')');
    }

    fn handle_image(&mut self, el: &ElementRef, buf: &mut String) {
        let src = el.value().attr("src").unwrap_or("");
        if src.is_empty() {
            return;
        }
        let alt = el.value().attr("alt").unwrap_or("Image");
        let title = el.value().attr("title").unwrap_or("");
        let resolved = resolve_url(src, &self.base_url);
        buf.push_str("![");
        buf.push_str(alt);
        buf.push_str("](");
        buf.push_str(&resolved);
        if !title.is_empty() {
            buf.push_str(" \"");
            buf.push_str(title);
            buf.push('"');
        }
        buf.push(')');
    }

    fn handle_list(&mut self, el: &ElementRef, ordered: bool, buf: &mut String) {
        let items = direct_children_by_sel(el, &SEL_LI);
        let mut counter = 1usize;
        for li in &items {
            let content = self.children_to_string(li);
            let trimmed = content.trim();
            if !trimmed.is_empty() {
                if ordered {
                    buf.push_str(&counter.to_string());
                    buf.push_str(". ");
                    counter += 1;
                } else {
                    buf.push_str("- ");
                }
                buf.push_str(trimmed);
                buf.push('\n');
            }
        }
        buf.push('\n');
    }

    fn handle_table(&mut self, el: &ElementRef, buf: &mut String) {
        let tag = el.value().name();

        // For thead/tbody/tfoot wrappers, just walk their rows
        if tag != "table" {
            self.walk_children(el, buf);
            return;
        }

        // Gather rows
        let has_nested_table = el.select(&SEL_TABLE).next().is_some();
        let mut rows: Vec<ElementRef> = direct_children_by_sel(el, &SEL_TR);

        if rows.is_empty() {
            // Look inside thead/tbody/tfoot
            let sections = direct_children_by_sel(el, &SEL_THEAD_TBODY_TFOOT);
            for sec in &sections {
                rows.extend(direct_children_by_sel(sec, &SEL_TR));
            }
        }
        if rows.is_empty() {
            if has_nested_table {
                self.walk_children(el, buf);
            }
            return;
        }

        // Layout detection
        let first_row = &rows[0];
        let first_row_cells = direct_children_by_sel(first_row, &SEL_TD_TH);

        let has_block_children = first_row_cells.iter().any(|cell| {
            cell.children().any(|c| {
                if let Some(ce) = ElementRef::wrap(c) {
                    BLOCK_LIKE_TAGS.contains(&ce.value().name())
                } else {
                    false
                }
            })
        });

        let looks_like_layout =
            !first_row_cells.is_empty() && first_row_cells.len() <= 2 && rows.len() >= 15;

        if has_nested_table || has_block_children || looks_like_layout {
            if self.dedupe_tables {
                self.layout_table_depth += 1;
                self.walk_children(el, buf);
                self.layout_table_depth -= 1;
            } else {
                self.walk_children(el, buf);
            }
            return;
        }

        // Data table — emit markdown table
        let mut md_rows: Vec<String> = Vec::new();
        let mut first_has_th = false;
        let mut first_cell_count = 0;

        for (i, row) in rows.iter().enumerate() {
            let cells = direct_children_by_sel(row, &SEL_TD_TH);
            if cells.is_empty() {
                continue;
            }
            let mut parts: Vec<String> = Vec::new();
            for cell in &cells {
                parts.push(get_text_content(cell));
                if i == 0 {
                    if cell.value().name() == "th" {
                        first_has_th = true;
                    }
                }
            }
            if i == 0 {
                first_cell_count = parts.len();
            }
            let row_str = format!("| {} |", parts.join(" | "));
            md_rows.push(row_str);
        }

        if md_rows.is_empty() {
            return;
        }

        if first_has_th && first_cell_count > 0 {
            let sep = format!(
                "| {} |",
                vec!["---"; first_cell_count].join(" | ")
            );
            md_rows.insert(1, sep);
        }

        for row_str in &md_rows {
            buf.push_str(row_str);
            buf.push('\n');
        }
        buf.push('\n');
    }
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/// Get direct children matching a selector (direct children only, not all descendants).
fn direct_children_by_sel<'a>(parent: &ElementRef<'a>, _sel: &Selector) -> Vec<ElementRef<'a>> {
    parent
        .children()
        .filter_map(ElementRef::wrap)
        .filter(|c| _sel.matches(c))
        .collect()
}

/// Recursively extract text content (normalised whitespace).
fn get_text_content(el: &ElementRef) -> String {
    let mut parts: Vec<String> = Vec::new();
    collect_text(el, &mut parts);
    let joined = parts.join("");
    // Normalise whitespace
    joined.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn collect_text(el: &ElementRef, parts: &mut Vec<String>) {
    for child in el.children() {
        match child.value() {
            Node::Text(t) => {
                parts.push(t.text.to_string());
            }
            Node::Element(_) => {
                if let Some(child_el) = ElementRef::wrap(child) {
                    collect_text(&child_el, parts);
                }
            }
            _ => {}
        }
    }
}

/// Get raw text preserving whitespace (for <pre> blocks).
fn get_raw_text(el: &ElementRef) -> String {
    let mut parts: Vec<String> = Vec::new();
    collect_text(el, &mut parts);
    parts.join("")
}

// ---------------------------------------------------------------------------
// Clean markdown (same rules as Python _clean_markdown)
// ---------------------------------------------------------------------------

static RE_MULTI_NL: Lazy<Regex> = Lazy::new(|| Regex::new(r"\n{3,}").unwrap());
static RE_MULTI_SP: Lazy<Regex> = Lazy::new(|| Regex::new(r" {2,}").unwrap());
static RE_EMPTY_LI: Lazy<Regex> = Lazy::new(|| Regex::new(r"\n- \n").unwrap());
static RE_EMPTY_OL: Lazy<Regex> = Lazy::new(|| Regex::new(r"\n\d+\. \n").unwrap());
static RE_HEADER_BEFORE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\n+(#{1,6})").unwrap());
static RE_HEADER_AFTER: Lazy<Regex> = Lazy::new(|| Regex::new(r"(#{1,6}.*)\n+").unwrap());

fn clean_markdown(md: &str) -> String {
    let s = RE_MULTI_NL.replace_all(md, "\n\n");
    let s = RE_MULTI_SP.replace_all(&s, " ");
    let s = RE_EMPTY_LI.replace_all(&s, "\n");
    let s = RE_EMPTY_OL.replace_all(&s, "\n");
    s.trim().to_string()
}

fn clean_markdown_readable(md: &str) -> String {
    let s = RE_MULTI_NL.replace_all(md, "\n\n");
    let s = RE_EMPTY_LI.replace_all(&s, "\n");
    let s = RE_HEADER_BEFORE.replace_all(&s, "\n\n$1");
    let s = RE_HEADER_AFTER.replace_all(&s, "$1\n\n");
    s.trim().to_string()
}

// ---------------------------------------------------------------------------
// Post-processing: citations, references, plain, images
// ---------------------------------------------------------------------------

fn extract_links_and_citations(md: &str, base_url: &Option<Url>) -> (Vec<LinkInfo>, String) {
    let mut links: Vec<LinkInfo> = Vec::new();
    let mut citation_counter = 1usize;
    let mut result = md.to_string();

    // We need to collect matches first to avoid borrow issues
    let matches: Vec<(String, String, String, String)> = RE_LINK
        .find_iter(md)
        .filter_map(|m| {
            let full = m.as_str().to_string();
            // Skip image links (start with !)
            if full.starts_with('!') {
                return None;
            }
            RE_LINK.captures(m.as_str()).map(|caps| {
                let text = caps.get(1).map_or("", |c| c.as_str()).to_string();
                let url = caps.get(2).map_or("", |c| c.as_str()).to_string();
                let title = caps.get(3).map_or("", |c| c.as_str()).to_string();
                (full, text, url, title)
            })
        })
        .collect();

    for (full, text, url, title) in matches {
        let resolved = if let Some(base) = base_url {
            match base.join(&url) {
                Ok(u) => u.to_string(),
                Err(_) => url.clone(),
            }
        } else {
            url.clone()
        };

        links.push(LinkInfo {
            text: text.clone(),
            url: resolved,
            title: title.clone(),
            citation_number: citation_counter,
        });

        let citation = format!("{}[{}]", text, citation_counter);
        result = result.replacen(&full, &citation, 1);
        citation_counter += 1;
    }

    (links, result)
}

fn generate_references(links: &[LinkInfo]) -> String {
    if links.is_empty() {
        return String::new();
    }
    let mut refs = String::from("## References\n");
    for link in links {
        refs.push_str(&format!("[{}]: {}", link.citation_number, link.url));
        if !link.title.is_empty() {
            refs.push_str(&format!(" \"{}\"", link.title));
        }
        refs.push('\n');
    }
    refs
}

fn strip_links(md: &str) -> String {
    // Replace [text](url) with text
    let s = Regex::new(r"\[([^\]]+)\]\([^)]+\)")
        .unwrap()
        .replace_all(md, "$1");
    // Replace ![alt](url) with alt
    let s = Regex::new(r"!\[([^\]]*)\]\([^)]+\)")
        .unwrap()
        .replace_all(&s, "$1");
    s.to_string()
}

fn extract_images(md: &str) -> Vec<ImageInfo> {
    RE_IMAGE
        .captures_iter(md)
        .map(|caps| ImageInfo {
            alt: caps.get(1).map_or("", |c| c.as_str()).to_string(),
            url: caps.get(2).map_or("", |c| c.as_str()).to_string(),
            title: caps.get(3).map_or("", |c| c.as_str()).to_string(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Main content detection
// ---------------------------------------------------------------------------

/// Pre-compute the set of node IDs that belong to nav/clutter subtrees so the
/// walker can skip them.
fn build_skip_set(doc: &Html) -> HashSet<NodeId> {
    let mut set = HashSet::new();

    for el in doc.root_element().children().filter_map(ElementRef::wrap) {
        collect_nav_ids(&el, &mut set);
    }

    set
}

fn collect_nav_ids(el: &ElementRef, set: &mut HashSet<NodeId>) {
    if should_skip(el) || is_nav_clutter(el) {
        add_subtree(el, set);
        return;
    }
    for child in el.children().filter_map(ElementRef::wrap) {
        collect_nav_ids(&child, set);
    }
}

fn add_subtree(el: &ElementRef, set: &mut HashSet<NodeId>) {
    set.insert(el.id());
    for child in el.descendants().filter_map(ElementRef::wrap) {
        set.insert(child.id());
    }
}

fn find_main_content<'a>(doc: &'a Html, skip_ids: &HashSet<NodeId>) -> Option<ElementRef<'a>> {
    for sel in SEL_MAIN.iter() {
        for el in doc.select(sel) {
            if !skip_ids.contains(&el.id()) {
                return Some(el);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Fallback logic (same as Python _should_fallback)
// ---------------------------------------------------------------------------

fn should_fallback(html: &str, md: &str, base_url: &str) -> bool {
    let html_len = html.len();
    let md_len = md.len();
    if md_len == 0 {
        return true;
    }
    if html_len < 5000 {
        return false;
    }
    if md_len < 400 {
        return true;
    }
    if (md_len as f64 / html_len.max(1) as f64) < 0.01 {
        return true;
    }
    if base_url.contains("news.ycombinator.com") && !md.contains("item?id=") {
        return true;
    }
    false
}

// ---------------------------------------------------------------------------
// Top-level pipeline
// ---------------------------------------------------------------------------

fn run_pipeline(html: &str, base_url: &str, dedupe_tables: bool) -> PipelineResult {
    let parsed_base: Option<Url> = if base_url.is_empty() {
        None
    } else {
        Url::parse(base_url).ok()
    };

    let doc = Html::parse_document(html);
    let skip_ids = build_skip_set(&doc);

    // Find main content node
    let main_node = find_main_content(&doc, &skip_ids);

    let mut walker = Walker {
        base_url: parsed_base.clone(),
        dedupe_tables,
        layout_table_depth: 0,
        skip_ids: &skip_ids,
    };

    let mut raw = String::with_capacity(html.len() / 4);
    if let Some(node) = main_node {
        walker.walk(node, &mut raw);
    }

    let raw = clean_markdown(&raw);

    // Fallback: if too sparse, re-walk the entire document
    let raw = if should_fallback(html, &raw, base_url) {
        let empty_skip = HashSet::new();
        let mut walker2 = Walker {
            base_url: parsed_base.clone(),
            dedupe_tables,
            layout_table_depth: 0,
            skip_ids: &empty_skip,
        };
        let mut full_buf = String::with_capacity(html.len() / 4);
        // Walk root element (usually <html>)
        let root = doc.root_element();
        walker2.walk(root, &mut full_buf);
        clean_markdown(&full_buf)
    } else {
        raw
    };

    // Post-processing
    let (links, md_with_citations) = extract_links_and_citations(&raw, &parsed_base);
    let references = generate_references(&links);
    let clean = clean_markdown_readable(&raw);
    let plain = strip_links(&raw);
    let images = extract_images(&raw);
    let urls: Vec<String> = links.iter().map(|l| l.url.clone()).collect();

    let md_references = if references.is_empty() {
        md_with_citations.clone()
    } else {
        format!("{}\n\n{}", md_with_citations, references)
    };

    PipelineResult {
        raw_markdown: raw,
        clean_markdown: clean,
        markdown_with_citations: md_with_citations,
        references_markdown: references,
        markdown_references: md_references,
        markdown_plain: plain,
        links,
        images,
        urls,
    }
}

struct PipelineResult {
    raw_markdown: String,
    clean_markdown: String,
    markdown_with_citations: String,
    references_markdown: String,
    markdown_references: String,
    markdown_plain: String,
    links: Vec<LinkInfo>,
    images: Vec<ImageInfo>,
    urls: Vec<String>,
}

// ---------------------------------------------------------------------------
// PyO3 bindings
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (html, base_url="", dedupe_tables=true))]
fn generate_markdown(py: Python<'_>, html: &str, base_url: &str, dedupe_tables: bool) -> PyResult<PyObject> {
    let result = run_pipeline(html, base_url, dedupe_tables);

    let dict = PyDict::new_bound(py);
    dict.set_item("raw_markdown", &result.raw_markdown)?;
    dict.set_item("clean_markdown", &result.clean_markdown)?;
    dict.set_item("markdown_with_citations", &result.markdown_with_citations)?;
    dict.set_item("references_markdown", &result.references_markdown)?;
    dict.set_item("markdown_references", &result.markdown_references)?;
    dict.set_item("markdown_plain", &result.markdown_plain)?;

    // Links
    let links_list = PyList::empty_bound(py);
    for link in &result.links {
        let d = PyDict::new_bound(py);
        d.set_item("text", &link.text)?;
        d.set_item("url", &link.url)?;
        d.set_item("title", &link.title)?;
        d.set_item("citation_number", link.citation_number)?;
        links_list.append(d)?;
    }
    dict.set_item("links", links_list)?;

    // Images
    let images_list = PyList::empty_bound(py);
    for img in &result.images {
        let d = PyDict::new_bound(py);
        d.set_item("alt", &img.alt)?;
        d.set_item("url", &img.url)?;
        d.set_item("title", &img.title)?;
        images_list.append(d)?;
    }
    dict.set_item("images", images_list)?;

    // URLs
    let urls_list = PyList::new_bound(py, &result.urls);
    dict.set_item("urls", &urls_list)?;

    Ok(dict.into())
}

#[pymodule]
fn grub_md(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_markdown, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_heading() {
        let r = run_pipeline("<h1>Hello</h1><p>World</p>", "", true);
        assert!(r.raw_markdown.contains("# Hello"));
        assert!(r.raw_markdown.contains("World"));
    }

    #[test]
    fn test_link_extraction() {
        let r = run_pipeline(
            r#"<p><a href="https://example.com">Example</a></p>"#,
            "",
            true,
        );
        assert!(r.raw_markdown.contains("[Example](https://example.com)"));
        assert_eq!(r.links.len(), 1);
        assert_eq!(r.links[0].url, "https://example.com");
        assert_eq!(r.links[0].citation_number, 1);
    }

    #[test]
    fn test_relative_url_resolution() {
        let r = run_pipeline(
            r#"<p><a href="/page">Link</a></p>"#,
            "https://example.com",
            true,
        );
        assert!(r.raw_markdown.contains("https://example.com/page"));
    }

    #[test]
    fn test_image() {
        let r = run_pipeline(
            r#"<img src="test.png" alt="Test Image" title="A test">"#,
            "",
            true,
        );
        assert!(r.raw_markdown.contains("![Test Image](test.png \"A test\")"));
        assert_eq!(r.images.len(), 1);
    }

    #[test]
    fn test_skip_script_style() {
        let r = run_pipeline(
            "<p>Keep</p><script>bad()</script><style>.x{}</style><p>Also keep</p>",
            "",
            true,
        );
        assert!(r.raw_markdown.contains("Keep"));
        assert!(r.raw_markdown.contains("Also keep"));
        assert!(!r.raw_markdown.contains("bad()"));
        assert!(!r.raw_markdown.contains(".x{}"));
    }

    #[test]
    fn test_code_and_pre() {
        let r = run_pipeline(
            "<p>Use <code>foo()</code> and:</p><pre>bar()\nbaz()</pre>",
            "",
            true,
        );
        assert!(r.raw_markdown.contains("`foo()`"));
        assert!(r.raw_markdown.contains("```\nbar()\nbaz()\n```"));
    }

    #[test]
    fn test_table_data() {
        let r = run_pipeline(
            "<table><tr><th>Name</th><th>Age</th></tr><tr><td>Alice</td><td>30</td></tr></table>",
            "",
            true,
        );
        assert!(r.raw_markdown.contains("| Name | Age |"));
        assert!(r.raw_markdown.contains("| --- | --- |"));
        assert!(r.raw_markdown.contains("| Alice | 30 |"));
    }

    #[test]
    fn test_empty_input() {
        let r = run_pipeline("", "", true);
        assert!(r.raw_markdown.is_empty());
    }

    #[test]
    fn test_plain_strips_links() {
        let r = run_pipeline(
            r#"<p><a href="https://example.com">Click</a> here</p>"#,
            "",
            true,
        );
        assert!(r.markdown_plain.contains("Click"));
        assert!(!r.markdown_plain.contains("example.com"));
    }

    #[test]
    fn test_citations() {
        let r = run_pipeline(
            r#"<p><a href="https://a.com">A</a> and <a href="https://b.com">B</a></p>"#,
            "",
            true,
        );
        assert!(r.markdown_with_citations.contains("A[1]"));
        assert!(r.markdown_with_citations.contains("B[2]"));
        assert!(r.references_markdown.contains("[1]: https://a.com"));
        assert!(r.references_markdown.contains("[2]: https://b.com"));
    }

    #[test]
    fn test_main_content_detection() {
        let html = r#"
            <html><body>
                <nav><a href="/home">Home</a></nav>
                <main><h1>Main Title</h1><p>Main content</p></main>
                <footer>Footer stuff</footer>
            </body></html>
        "#;
        let r = run_pipeline(html, "", true);
        assert!(r.raw_markdown.contains("Main Title"));
        assert!(r.raw_markdown.contains("Main content"));
        // Nav and footer should be filtered out
        assert!(!r.raw_markdown.contains("Home"));
        assert!(!r.raw_markdown.contains("Footer stuff"));
    }

    #[test]
    fn test_fallback_sparse() {
        // Large HTML but tiny main content → should trigger fallback
        let padding = "<div>x</div>".repeat(500);
        let html = format!(
            "<html><body><main><p>tiny</p></main><article>{}</article></body></html>",
            padding
        );
        let r = run_pipeline(&html, "", true);
        // Fallback should include the repeated text
        assert!(r.raw_markdown.contains("x"));
    }

    #[test]
    fn test_hidden_removed() {
        let html = r#"<p>Visible</p><span class="sr-only">Hidden</span><div hidden>Also hidden</div>"#;
        let r = run_pipeline(html, "", true);
        assert!(r.raw_markdown.contains("Visible"));
        assert!(!r.raw_markdown.contains("Hidden"));
        assert!(!r.raw_markdown.contains("Also hidden"));
    }
}
