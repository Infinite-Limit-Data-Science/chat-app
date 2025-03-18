1) Doc Enhancements 

Currently, we extract all text from all elements in the page, drilling down into the xml nodes of word docs and powerpoint docs and pdfs to extract text from paragraph nodes, table, table cell nodes, and capturing image nodes as well. In the context of "paragaph" nodes, it is the section of a word document you typically write in. Everything there is a paragraph node. The only distinction is between the "headings" element that microsoft adds as a convenience. But the xml tags all are this structure: <w:p>. In effect, examining XML tags alone (e.g., <w:p>, <w:tbl>, <w:drawing>) usually doesn’t help much in identifying “headers” vs. normal paragraphs, because Word (and PowerPoint) use the same tags for both headings and normal text (for instance, <w:p> is just a paragraph). The distinction between headings and normal text is typically found in styles or formatting (font size, bold, style ID, etc.), not at the high-level tag name.
All XML paragraphs in aforementioned document types (e.g. Word, Powerpoint) carry extra formatting properties within the <w:p> element beyond just plain text. For example, bulleted or numbered lists differ from normal body text by including numbering/bullet properties (<w:numPr>) in the paragraph’s XML or by referencing a list style. Even though both are <w:p>, Word sees the list properties and displays bullets or numbering accordingly. Example:

<w:p>
 <w:pPr>
   <w:numPr>
     <w:ilvl w:val="0"/>
     <w:numId w:val="1"/>
   </w:numPr>
  </w:pPr>
</w:p>

Visually, a list item in Word shows up with a bullet/number, indentation, etc., while a normal paragraph (lacking <w:numPr>) does not. But under the hood, they’re both <w:p> elements with different properties.

Below is a summary of why using the distribution of XML tags rarely helps for heading detection, and why it’s more fruitful to rely on the distribution of font sizes (and other formatting attributes) or the built-in heading styles.

Thus, if you do a raw count of the different XML tags: 
- You’ll see a bunch of <w:p> (paragraph) elements.
- Some <w:tbl> (tables).
- Possibly <w:drawing> or <w:pict> for images, etc.

But heading paragraphs do not get a special “heading” tag. They’re still <w:p>.

A .docx file (OpenXML) organizes text primarily in:
- Document Body (docx.Document(...).paragraphs where Tables appear as separate XML blocks, but each table cell also contains paragraphs.)
- Headers/Footers
Each Section in Word can have its own header/footer. In python-docx, you can access them via doc.sections and then section.header.paragraphs / section.footer.paragraphs.
- Shapes / Text Boxes
- Footnotes / Endnotes

For most typical Word documents, you’ll get 90%+ of the visible text with:
- doc.paragraphs
- for table in doc.tables: then table.cell(x,y).paragraphs
- for section in doc.sections: with section.header/footer.paragraphs
Each paragraph is composed of runs (<w:r>). A run can have formatting (font size, bold, italic, etc.).

So from a structural standpoint:

The paragraph is the minimal block-level container (like a line or block of text).
The run is the minimal inline container (like a styled span).

Hence, to do a global statistical approach, you eventually want to gather all those runs from all paragraphs—wherever they occur (body, table, header, shape).

Here’s some code showing how to gather paragraphs from body, tables, and headers/footers. (We’ll skip shapes/text boxes here, as that’s advanced.)

import docx
from docx.text.paragraph import Paragraph
from docx.table import Table

def get_all_paragraphs(doc):
    """
    Return a list of all paragraphs from:
      - Main body
      - Table cells in the main body
      - All headers & footers
    """
    all_paras = []

    # 1) Main body
    for block in doc.element.body:
        if block.tag.endswith("p"):
            all_paras.append(Paragraph(block, doc))
        elif block.tag.endswith("tbl"):
            table = Table(block, doc)
            all_paras.extend(get_paragraphs_from_table(table))

    # 2) Headers/footers
    for section in doc.sections:
        header = section.header
        footer = section.footer
        # header paragraphs
        for block in header._element:
            if block.tag.endswith("p"):
                all_paras.append(Paragraph(block, doc))
            elif block.tag.endswith("tbl"):
                table = Table(block, doc)
                all_paras.extend(get_paragraphs_from_table(table))
        # footer paragraphs
        for block in footer._element:
            if block.tag.endswith("p"):
                all_paras.append(Paragraph(block, doc))
            elif block.tag.endswith("tbl"):
                table = Table(block, doc)
                all_paras.extend(get_paragraphs_from_table(table))

    return all_paras

def get_paragraphs_from_table(table):
    """
    Recursively extract paragraphs from a docx.table.Table, including nested tables.
    """
    paragraphs = []
    for row in table.rows:
        for cell in row.cells:
            for block in cell._element:
                if block.tag.endswith("p"):
                    paragraphs.append(Paragraph(block, table._parent))
                elif block.tag.endswith("tbl"):
                    nested_table = Table(block, table._parent)
                    paragraphs.extend(get_paragraphs_from_table(nested_table))
    return paragraphs

doc = docx.Document("example.docx")
all_paras = get_all_paragraphs(doc)

print(f"Found {len(all_paras)} paragraphs (including tables, headers, footers).")

In python-docx, when you access low-level elements (like doc.element.body), you’re iterating over the actual XML elements in the .docx file. Each of these elements has a tag property that indicates its XML tag name (including the namespace). For example:

- A paragraph element in Word’s XML is <w:p>…</w:p> in the WordprocessingML namespace.
- A table element is <w:tbl>…</w:tbl>.

So when you see code like:

for block in doc.element.body:
    if block.tag.endswith("p"):
        # it's a paragraph
    elif block.tag.endswith("tbl"):
        # it's a table

block.tag is not a high-level python-docx property; it’s the raw XML tag from the underlying document structure.

In python-docx, you typically detect a heading by checking paragraph.style or scanning the underlying XML for <w:pStyle w:val="HeadingX">. But from a raw XML/tag perspective, it’s still just <w:p>.

internally, Heading 1, 2, or 3 paragraphs are still <w:p> in the XML. The “heading-ness” is indicated by a paragraph style:

<w:p>
  <w:pPr>
    <w:pStyle w:val="Heading1" />
  </w:pPr>
  <w:r>
    <w:t>My Heading Text</w:t>
  </w:r>
</w:p>

By default, Word’s built-in heading styles do use style identifiers like “Heading1”, “Heading2”, etc. (internally, there may be no space). However:

Yes, it’s “hard-coded” in the sense that the default built-in heading styles in Word carry known names/IDs (e.g., “Heading 1” style in the UI often corresponds to an internal style ID of “Heading1”).

But users can rename or customize these styles. 

Write python code that takes the paragraph of word doc and divides it by <w:pStyle w:val="Heading1"> or if there is no Heading1, check Heading2, and if no Heading2, check Heading3 and so forth.


import docx
from typing import List, Dict

def divide_by_headings(docx_path: str) -> List[Dict[str, List[str]]]:
    """
    Loads a Word document, segments it by heading paragraphs (Heading 1, 2, or 3),
    and returns a list of sections, each with:
      {
        "heading_level": int,
        "heading_text": str,
        "content": List[str]
      }
    """
    doc = docx.Document(docx_path)

    sections = []
    current_section = {
        "heading_level": 0,
        "heading_text": "Untitled",
        "content": []
    }

    for paragraph in doc.paragraphs:
        # Check if this paragraph is one of our heading styles
        style_name = paragraph.style.name if paragraph.style else ""
        if style_name in ["Heading 1", "Heading 2", "Heading 3"]:
            # Start a new section for each heading paragraph

            # First, store the old section if it has content
            if current_section["heading_text"] or current_section["content"]:
                sections.append(current_section)

            # Determine heading level (1, 2, or 3)
            level = 1 if style_name == "Heading 1" else 2 if style_name == "Heading 2" else 3

            current_section = {
                "heading_level": level,
                "heading_text": paragraph.text.strip(),
                "content": []
            }
        else:
            # It's a normal paragraph; add to current section content
            if paragraph.text.strip():
                current_section["content"].append(paragraph.text.strip())

    # Append the last section if it has data
    if current_section["heading_text"] or current_section["content"]:
        sections.append(current_section)

    return sections

if __name__ == "__main__":
    docx_file = "example.docx"
    result_sections = divide_by_headings(docx_file)

    # Print the results
    for idx, sec in enumerate(result_sections, start=1):
        print(f"Section {idx}: Level={sec['heading_level']} Heading={sec['heading_text']}")
        for c in sec["content"]:
            print("  ", c)

Now we have our sections structured:
{
  "heading_level": 1 or 2 or 3,
  "heading_text": "The heading's text",
  "content": [paragraph_string, paragraph_string, ...]
}

What is key here is we can later associate heading_text with each content item as the chunk overlap. 

If your document doesn’t contain a “Heading 1,” then paragraphs styled as “Heading 2” will start new sections with heading_level=2.

If you need to handle more heading levels (Heading 4, Heading 5, etc.), just extend the check:

if style_name.startswith("Heading "):
    # parse the number

What happens if you have Heading 1 followed by Heading 2?
The script closes off the current “section” (associated with Heading 1) and starts a new section for Heading 2.
This yields a flat list of sections:
Section with heading_level=1 + content
Section with heading_level=2 + content
Section with heading_level=1 again, etc.

So in the final data structure:

[
  {
    "heading_level": 1,
    "heading_text": "Some H1 text",
    "content": ["some paragraphs..."]
  },
  {
    "heading_level": 2,
    "heading_text": "Sub-heading text",
    "content": ["some paragraphs..."]
  },
  ...
]

Enhanced Approach (Hierarchical / Nested Headings)

Below is a sample implementation. This is more advanced than the simple “flat segmentation,” but it properly nests headings.

import docx
from docx.text.paragraph import Paragraph
from typing import List

class HeadingNode:
    def __init__(self, level: int, title: str):
        self.level = level       # e.g., 1 for Heading 1, 2 for Heading 2, etc.
        self.title = title       # text of the heading
        self.content = []        # paragraphs under this heading
        self.children = []       # subheadings

    def __repr__(self):
        return f"HeadingNode(level={self.level}, title={self.title!r})"

def get_heading_level(style_name: str) -> int:
    """
    Maps style name 'Heading 1', 'Heading 2', 'Heading 3' -> 1, 2, 3
    Returns 0 if not recognized as a heading style.
    """
    if style_name.startswith("Heading "):
        try:
            return int(style_name.split(" ")[1])
        except:
            return 1
    elif style_name == "Title":
        return 1
    return 0

def build_heading_hierarchy(docx_path: str) -> HeadingNode:
    """
    Builds a nested heading hierarchy from the Word document.
    Returns the 'root' HeadingNode (level=0) that contains top-level headings as children.
    """
    doc = docx.Document(docx_path)

    # We'll create a synthetic "root" node of level 0.
    root = HeadingNode(level=0, title="ROOT")
    stack = [root]  # stack top = current heading context

    for paragraph in doc.paragraphs:
        style_name = paragraph.style.name if paragraph.style else ""
        lvl = get_heading_level(style_name)
        text = paragraph.text.strip()

        if lvl > 0:
            # We have a heading paragraph
            new_heading = HeadingNode(level=lvl, title=text)

            # POP until we find a heading on stack that has a smaller level
            while stack and stack[-1].level >= lvl:
                stack.pop()

            # Now stack[-1] is the parent heading
            stack[-1].children.append(new_heading)
            # push new heading onto the stack
            stack.append(new_heading)
        else:
            # Not a heading -> add to content of current top-of-stack
            stack[-1].content.append(text)

    return root

def print_heading_tree(node: HeadingNode, indent=0):
    """
    Recursively prints the hierarchy of headings for demonstration.
    """
    prefix = "  " * indent
    if node.level > 0:
        print(f"{prefix}Heading L{node.level}: {node.title}")
        for c in node.content:
            print(f"{prefix}  - {c}")
    for child in node.children:
        print_heading_tree(child, indent+1)

if __name__ == "__main__":
    docx_file = "example.docx"
    root = build_heading_hierarchy(docx_file)
    print_heading_tree(root)

Explanation
root = HeadingNode(level=0, title="ROOT")

A dummy container node at level 0, so that real headings (level 1, 2, 3…) become its children.
stack = [root]

We keep a LIFO stack so stack[-1] is the current heading context.

Result

You end up with a tree of headings:
Level 1 headings are children of root.
Level 2 headings are children of whichever heading L1 preceded them.
Level 3 headings are children of the L2 heading, etc.
Each heading node has its own content (the paragraphs directly under it, up until the next heading of equal or higher level).

Identify Official Headings

If the paragraph style is exactly "Heading 1", map it to <h1>…</h1> in your output.
Style "Heading 2" → <h2>…</h2>, etc.
This immediately handles the “clean” portion of the document where the author used built-in heading styles.

Catches Manual “Heading-like” Text: Some authors never learned to use “Heading 1,” so they just enlarged and bolded text. 

Below is an illustrative example that combines:
- Official Word heading styles (Heading 1, 2, 3, etc.)
- Statistical detection of “unofficial headings” based on font sizes (z-scores from the mean).

** Statistical detection

# Here we get all our paragraphs in document
def gather_all_body_paragraphs(doc: docx.Document) -> List[Paragraph]:
    """
    Collects all top-level paragraphs (and table paragraphs) from the main document body.
    Skips headers, footers, shapes, etc. for brevity.
    """
    paragraphs = []
    body = doc.element.body

    for block in body:
        if block.tag.endswith("p"):  # paragraph
            paragraphs.append(Paragraph(block, doc))
        elif block.tag.endswith("tbl"):  # table
            table = Table(block, doc)
            # gather paragraphs in each cell
            for row in table.rows:
                for cell in row.cells:
                    paragraphs.extend(cell.paragraphs)
    return paragraphs

It would be nice to store the page number of the paragraph BUT: python-docx doesn’t provide a built-in way to retrieve actual page numbers for each paragraph, because the concept of “what page a paragraph is on” is not stored in the .docx file. Word dynamically calculates pagination (page layout) when rendering or printing. In other words, .docx is a reflowable format, so there’s no guaranteed or fixed “page number” embedded in the paragraph data.

def detect_if_paragraph_is_bullet(paragraph: Paragraph) -> bool:
    """
    Tries to detect bullet/numbered lists (very naive).
    If paragraph.style.name has "List" or there's paragraph numbering,
    treat it as a bullet item. You might refine with docx's numbering APIs.
    """
    style_name = paragraph.style.name if paragraph.style else ""
    if "List" in style_name:
        return True
    # you could also check paragraph.paragraph_format or paragraph._p for w:numPr
    return False

def get_runs_font_sizes(paragraph: Paragraph) -> List[float]:
    """
    Returns a list of float font sizes (in points) for each run
    that has an explicitly defined font size. Ignores runs with no size set.
    """
    sizes = []
    for run in paragraph.runs:
        if run.font.size is not None:
            sizes.append(run.font.size.pt)
    return sizes

In python-docx, a run is effectively a segment of text within a paragraph that shares the same styling and formatting (e.g., font size, bold, italic). Each paragraph can contain multiple runs, each potentially with different properties. For example, if a single sentence has one word in bold and the rest in normal text, that sentence might be split into two runs: one run for the bold word, and one run for the remaining normal text.

Here’s a hypothetical example of what get_runs_font_sizes(paragraph) might return if a paragraph has multiple runs with different font sizes:

[11.0, 11.0, 14.0, 12.0, 16.0]
- The first two runs might be normal text at 11.0 pt.
- Then a bold subheading part at 14.0 pt.
- A short italic part at 12.0 pt.
- And finally, a larger run at 16.0 pt (maybe to emphasize something).

Any runs that do not have an explicitly defined size (and rely on inherited or default formatting) would not appear in this list (or return no size) unless python-docx was able to detect a specific point size from the file.
Each has .runs with .font.size, .text, .bold, etc. Then you can apply your statistical logic across all runs.

Below is an example of how you might use quartiles (rather than z-scores) to detect “large font size outliers” in a relatively small dataset (≤30 data points). 

- Sort the data (font sizes).
- Find Q1, Q2 (median), and Q3.
- Compute the Interquartile Range (IQR) = Q3 − Q1.
- Treat anything above Q3 + k * IQR (or below Q1 - k * IQR) as an outlier.
- A typical choice is k = 1.5 for a “mild” outlier, or k = 3.0 for a more “extreme” outlier threshold.

Because you only have up to 30 data points, quartiles/IQR can be more robust than a normal-based approach, and it doesn’t assume a unimodal or symmetric distribution.

import numpy as np
from typing import List

def detect_font_size_by_interquartile(font_sizes: List[float], k: float = 1.5) -> float:
    """
    Returns a threshold above which we consider font sizes to be "large outliers".
    
    font_sizes: list of font sizes (e.g., [11.0, 11.0, 12.0, 16.0, 18.0, ...])
    k: multiplier for the Interquartile Range (IQR). Default 1.5 or 3.0 are common.
    
    Returns:
      A float cutoff. Any size > cutoff is considered an "outlier" (candidate for heading).
      If fewer than 2 data points, returns float('inf') so no outlier detection is possible.
    """
    if len(font_sizes) < 2:
        return float('inf')  # can't do quartiles with fewer than 2 points
    
    # sort the data
    data = sorted(font_sizes)
    
    # compute quartiles
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    # outlier threshold: Q3 + k * IQR
    cutoff = Q3 + k * IQR
    return cutoff

# EXAMPLE USAGE:
if __name__ == "__main__":
    # Suppose we have a small dataset of font sizes
    sample_sizes = [11.0, 11.0, 12.0, 14.0, 11.5, 16.0, 11.0, 18.0, 12.5, 11.0]
    threshold = detect_large_font_sizes(sample_sizes, k=1.5)
    print("Outlier threshold:", threshold)
    
    outliers = [fs for fs in sample_sizes if fs > threshold]
    print("Sizes above threshold:", outliers)

Explanation
Sorting the data

We first sort the list of font sizes, so we can find the 25th and 75th percentiles.
Quartiles

Q1 = the 25th percentile.
Q3 = the 75th percentile.
Interquartile Range (IQR)

IQR = Q3 - Q1.
This measures the middle 50% of the data.
Defining Outlier Threshold

Common default is Q3 + 1.5 * IQR.
Values above that are “moderate” outliers.
If you want a stricter definition of outliers, you could do Q3 + 3.0 * IQR.
Return the Cutoff

In the example, anything larger than cutoff is considered an “outlier.”
If you want to detect extremely small sizes as outliers, you might also check Q1 - k * IQR. But typically for heading detection, you’re only worried about large font sizes.
Handling Very Small n

If you have fewer than 2 data points with explicit font sizes, you can’t meaningfully compute quartiles. We return float('inf'), effectively meaning no outliers.



Now in the case of more than 30 sizes, which can potentially lead to a normally distributed data set:

To give some context to this statement below (the AI generated a hardcode value for z-score at 1.4):
I don't understand why we set K to hardcoded value here:
                    # Example threshold: 1.4 stdev above mean
                    # You can tweak K
                    K = 1.4
                    if dominant_size >= mu + K * sigma:    Shouldnt the z-score be defined by the confidance level and significant value based on the distribution itself in the event of a normal distribution. For example, each document is different and if they are all normally distributed, the mu will be different and the standard deviation will be different, therefore we specify our confidence level, and then determine the z-score to use for the outlier areas of the curve

Yes, in a purely statistical approach, you would indeed derive K from your desired confidence level (
𝛼
α) for a one-tailed normal distribution, rather than just hardcoding it to 1.4. The snippet uses K = 1.4 as a simple heuristic – effectively saying, “If the font size is more than 1.4 standard deviations above the mean, we’ll treat it as unusual.”

Suppose you want to pick a 5% false-positive risk (i.e., you only want 5% of normal data to be flagged as heading). That corresponds to a one-tailed critical z-value of about 1.645. You could do:

import math
from scipy.stats import norm

alpha = 0.05  # 5% one-tailed
K = norm.ppf(1 - alpha)  # ~1.645

Or for a 2.5% tail, you’d get 1.96. Then:

if dominant_size >= mu + K*sigma:
    # It's in the top 5% (if the data is roughly normal)
    is_stat_heading = True

This way, K is derived from the confidence/alpha level you want, rather than being an arbitrary 1.4.

If your document truly has a roughly normal distribution of font sizes, deriving K from an alpha or p-value can be conceptually clearer (e.g., “we want to catch the top 5% as headings”).  So rewrite the method as a method called detect_font_size_by_normally_distributed which handles finding the outlier font sizes using the confidence level, z-score, where the z-score exceeds 
𝐾, we treat the paragraph as a “heading.” using numpy, scipy and remember to conform to the section concept of: current_section = {
                "heading": p_data["text"],
                "level": lvl,
                "content": []
            }    that is the contract that should be returned, where a heading has a list of contents. 


Below is an illustrative function that:

Loads a Word document with python-docx.
Gathers the font sizes from all runs to compute a mean and standard deviation.
Derives a z-score cutoff (zcrit) from the desired confidence level (
𝛼
α) using SciPy (instead of hardcoding a multiplier like 1.4).
Segments the document into a list of “sections,” each with a heading (heading) and heading level (level), plus a list of content paragraphs.
Contract: Each item in the returned list looks like:

python
Copy
Edit
{
  "heading": str,      # text of the heading paragraph
  "level": int,        # heading level (1,2,3,...)
  "content": [str, ...]  # list of normal paragraphs under this heading
}

This example:

Treats official Word headings (Heading 1, Heading 2, Heading 3) as sections automatically.
If a paragraph is not an official heading, but its font size is in the top α-tail of a normal distribution (z-score > zcrit) and it meets a few mild heuristics (short and bold), we flag it as an unofficial heading (level=2).
Everything else is appended to the current section’s content.

import docx
import numpy as np
from scipy.stats import norm
from typing import List, Dict, Any

def detect_font_size_by_normally_distributed(docx_path: str,
                                             alpha: float = 0.05
                                             ) -> List[Dict[str, Any]]:
    """
    Loads a .docx file, finds paragraphs that are likely headings in two ways:
      1) Official Word headings (Heading 1, 2, 3).
      2) Large font outliers (based on normal distribution & chosen alpha).
    Returns a list of sections: each has {heading, level, content[]}.

    Example of returned structure:
    [
      {
        "heading": "Introduction",
        "level": 1,
        "content": ["Paragraph 1", "Paragraph 2", ...]
      },
      {
        "heading": "Sub-topic",
        "level": 2,
        "content": ["Paragraph 3", ...]
      },
      ...
    ]
    """
    doc = docx.Document(docx_path)

    # 1) Collect all run font sizes for global stats
    paragraphs = doc.paragraphs
    all_run_sizes = []
    
    def get_run_sizes(paragraph):
        """Return list of explicitly defined font sizes (in points) in each run."""
        sizes = []
        for run in paragraph.runs:
            if run.font.size is not None:
                sizes.append(run.font.size.pt)
        return sizes

    # Gather all sizes
    for p in paragraphs:
        all_run_sizes.extend(get_run_sizes(p))

    # If there's not enough data, we can't do a meaningful normal analysis
    use_stats = False
    mu = 12.0  # fallback mean
    sigma = 0.0

    if len(all_run_sizes) >= 2:
        size_array = np.array(all_run_sizes, dtype=float)
        mu = size_array.mean()
        sigma = size_array.std(ddof=1)  # sample std
        if sigma > 0:
            use_stats = True

    # Derive z-critical from alpha (one-tailed). e.g. alpha=0.05 => zcrit ~ 1.645
    zcrit = norm.ppf(1 - alpha)  # top alpha tail

    # 2) We'll build "sections" as a flat list: heading + content
    sections = []
    current_section = {
        "heading": "Untitled Document",
        "level": 1,
        "content": []
    }

    def get_heading_level(style_name: str) -> int:
        """
        Parse style name 'Heading 1' => 1, 'Heading 2' => 2, etc.
        Returns 0 if not recognized as an official heading style.
        """
        if style_name.startswith("Heading "):
            try:
                return int(style_name.split(" ")[1])
            except:
                return 1
        elif style_name == "Title":
            return 1
        return 0

    for p in paragraphs:
        text = p.text.strip()
        run_sizes = get_run_sizes(p)
        dominant_size = max(run_sizes) if run_sizes else 0.0

        style_name = p.style.name if p.style else ""
        official_level = get_heading_level(style_name)

        # Heuristic to detect "unofficial heading" if not official heading
        is_unofficial_heading = False
        if official_level == 0 and text:
            # Some extra signals to reduce false positives
            is_short = (len(text.split()) <= 8)
            is_bold = any(run.bold for run in p.runs)
            # We'll only consider it if we have stats
            if use_stats and dominant_size > 0:
                # compute z-score
                z = (dominant_size - mu) / sigma if sigma else 0.0
                # if it's above zcrit, we treat as heading
                if (z >= zcrit) and is_short and is_bold:
                    is_unofficial_heading = True

        # Decide heading level
        if official_level > 0:
            # Start new section
            if current_section["content"] or current_section["heading"]:
                sections.append(current_section)
            current_section = {
                "heading": text,
                "level": official_level,
                "content": []
            }
        elif is_unofficial_heading:
            # We'll label these as "Heading 2" for demonstration
            if current_section["content"] or current_section["heading"]:
                sections.append(current_section)
            current_section = {
                "heading": text,
                "level": 2,
                "content": []
            }
        else:
            # Normal text => add to current section's content
            if text:
                current_section["content"].append(text)

    # Final section
    if current_section["content"] or current_section["heading"]:
        sections.append(current_section)

    return sections

# EXAMPLE USAGE:
if __name__ == "__main__":
    docx_file = "example.docx"
    alpha_level = 0.05  # e.g., top 5% tail => z ~ 1.645
    sections = detect_font_size_by_normally_distributed(docx_file, alpha=alpha_level)
    
    for idx, sec in enumerate(sections, start=1):
        print(f"Section {idx}: level={sec['level']} heading={sec['heading']!r}")
        for content_par in sec["content"]:
            print("  -", content_par)

Key Points
We do not hardcode a multiplier like 1.4. Instead:

python
Copy
Edit
zcrit = norm.ppf(1 - alpha)
So for alpha = 0.05 (top 5% tail), zcrit ≈ 1.645. That means if a paragraph’s font size is 1.645 standard deviations above the mean, we consider it outlier large.

Official Headings

If the style is "Heading 1", "Heading 2", or "Heading 3", we take that as a genuine heading. No need for a z-score check.
Unofficial Headings

If the paragraph is short, bold, and the dominant_size is above our alpha-based z-score threshold, we label it a heading (level=2 for example).
Segmenting

Each heading starts a new section. The paragraphs that follow (until the next heading) go into that section’s "content". We store them in a list for easy iteration or further processing.
Document Variation

If a doc has a very different distribution or few paragraphs with explicit sizes, you might not get good results from a normal approach. Then you can consider IQR-based detection or simpler heuristics.

Short answer: The Central Limit Theorem (CLT) doesn’t guarantee that your raw data (e.g., individual font sizes) follows a normal distribution. It only guarantees that the distribution of the sample mean (when repeatedly sampled from the population) approaches normal as sample size grows. That’s different from saying “all data points themselves are normally distributed.” If your underlying data is highly multi-modal or skewed, applying a z-score outlier test (which assumes a roughly normal distribution of data points) can lead to misleading results.

So how should we handle the multi-modal or skewed situation?

Below is a short summary of three common approaches to handle a multi-modal or skewed data distribution—where simply using a z-score outlier test (which relies on a roughly normal distribution) might be inappropriate for detecting “large” font sizes or outliers:

-  Use a Percentile / Rank-Based Cutoff
- IQR (Interquartile Range) Method
-  Clustering (e.g., K-Means or Hierarchical)

1. Use a Percentile / Rank-Based Cutoff
Rather than using μ+kσ, you can define a cutoff based on the empirical distribution of your data. For example:

- Sort the font sizes
- Decide you only want the top 5% or 2.5% of font sizes to be considered “headings” (or “outliers”).
- Compute that percentile. For instance, the 95th percentile (for a top 5% cutoff) is s_{0.95 * n}.
- Any paragraph with a font size above that percentile is flagged.
This approach doesn’t assume normality. It just says “the top X% of sizes are significantly larger than the rest.”

A more robust approach than mean/standard deviation is to use quartiles:
- Compute Q1 (25th percentile) and Q3 (75th percentile).
- Compute IQR = Q3 − Q1.
- Define a cutoff (e.g., Q3 + 1.5 * IQR) or Q3 + 3.0 * IQR for more extreme outliers.
- Any font size above that cutoff is considered an outlier.
Again, this is distribution-free: no assumption that data points follow a bell curve. It’s purely rank-based.

Clustering (e.g., K-Means or Hierarchical)
- If the data is multi-modal, you can attempt to cluster your font sizes:
- Run a small K-Means with K=2 or K=3 on the list of font sizes.
- Identify which cluster is the “largest font” cluster.
- Flag paragraphs in that cluster as “heading-like.”
- Optionally also incorporate additional signals (bold, short text, etc.).
This explicitly handles multi-modal distributions by grouping data into clusters. You don’t rely on a single mean and std. dev.



3) Restricting User Access to Mongo

How can I create two collections in mongodb and restrict user access to each collection where User A has access to one collection and User B has access to another collection.

In MongoDB, you can restrict users to specific collections by creating custom roles that grant privileges only on the desired collections, and then assigning those roles to specific users. Below is an outline of how you would set this up.

Edit mongod.conf so MongoDB runs with authorization enabled. For example:

security:
  authorization: enabled

Restart MongoDB so changes to mongod.conf take effect.

If you are running a standalone instance with the default settings, then by default no authentication is required. You must enable authentication for user and role management to work properly.

2. Connect as the Admin User
Start the Mongo shell without authentication (if you have not yet created any admin user), or connect as an already-existing admin user.
If you do not have an admin user at all, create one in the admin database. For example:
js
Copy
Edit
use admin

db.createUser({
  user: "siteAdmin",
  pwd: "StrongPassword",
  roles: [ { role: "root", db: "admin" } ]
})
After creating this first admin user, restart MongoDB with authorization: enabled. Then connect using this admin user:
sh
Copy
Edit
mongo -u "siteAdmin" -p "StrongPassword" --authenticationDatabase "admin"
3. Create Your Database and Collections
Let’s assume you have a database named myDB with two collections: collectionA and collectionB. For demonstration, you can do:

js
Copy
Edit
use myDB

db.collectionA.insertOne({ sampleField: "A" })
db.collectionB.insertOne({ sampleField: "B" })
These inserts also implicitly create the collections.

4. Create Roles That Restrict Access
You will create two custom roles in myDB:

readCollectionA only has read privilege on collectionA.
readCollectionB only has read privilege on collectionB.
js
Copy
Edit
use myDB

db.createRole({
  role: "readCollectionA",
  privileges: [
    {
      resource: { db: "myDB", collection: "collectionA" },
      actions: [ "find" ]   // "find" is read-only
    }
  ],
  roles: []
})

db.createRole({
  role: "readCollectionB",
  privileges: [
    {
      resource: { db: "myDB", collection: "collectionB" },
      actions: [ "find" ]
    }
  ],
  roles: []
})
Explanation:

resource specifies which database and collection the privileges apply to.
actions specify what operations can be done. "find" is a read-only operation. If you want to let the user insert, update, or remove documents, you would add "insert", "update", "remove", etc.
5. Create Users and Assign Roles
Next, create two users, userA and userB, each having only the corresponding custom role:

js
Copy
Edit
use myDB

db.createUser({
  user: "userA",
  pwd: "passwordA",
  roles: [
    { role: "readCollectionA", db: "myDB" }
  ]
})

db.createUser({
  user: "userB",
  pwd: "passwordB",
  roles: [
    { role: "readCollectionB", db: "myDB" }
  ]
})
6. Verify the Access Restrictions
Connect as userA:

 
mongo -u "userA" -p "passwordA" --authenticationDatabase "myDB"
Attempt to read collectionA:

 
use myDB
db.collectionA.find()
This should succeed because userA has the role readCollectionA.

Attempt to read collectionB:

 
db.collectionB.find()
This should fail (throw an authorization error) because userA does not have privileges on collectionB.

Connect as userB:
 
mongo -u "userB" -p "passwordB" --authenticationDatabase "myDB"
Attempt to read collectionB:
 
db.collectionB.find()
This should succeed.

Attempt to read collectionA:
 
db.collectionA.find()
This should fail with an authorization error.

4) Redis Monitoring of vector data and LLM monitoring of runnables (e.g. calls made to the model, generation and response times) 

1) Excel and Table Extract

The way I am doing tool calling right now is all wrong. 

I should create a Tool called TableDistributionTool that extends BaseTool so I can use both the sync _run and async _arun methods. 

We define an args_schema which will be our schema using pydantic. This part is crucial. In the past I said have one pandas data expression. But then it only produces a single expression which does not capture the nuance and multi step ask of the user prompt. 

Hence I need to create a schema that has a field called operation which captures the type of operation the user wants to perform. It should be an enum with a predefined list. That way one value could be z-test and when the Params are determined by the LLM, then it calls the tool and it takes the z-test route. Then the tool internally performs z-test on the data. 

Of course we also need to know the fields the user is inquiring about. So we need to add to our schema the list of the names of fields the user is inquiring about from the actual table in question. 

So now we have a function that knows what the user wants and the fields they are asking about from the table. We do the calculation. We add it in the ToolMessage as part of our message history. Then we take the tool response and the original message and then send it back to the model using the lang graph state graph conditional edge. 

We ask the model does this answer the users question. Let the model use its massive trained parameters (e.g. positional embedding vectors, token embeddings, transformer layers), which are frozen at inference time, to do a feed forward pass where the model takes the input tokens and passes them through its layers to produce a probability distribution over the next token. So let's say based on the user's original input, the model chooses the "operation" z-score and the fields "column_a, column_b, column_c". Then I run the tool for z-score, use the specified columns as filters, and then send the response back to the model. This is accomplished using a separate node to evaluate whether the most recent assistant message actually answers the user’s question. Branch to either END (if “yes”) or a fallback node (if “no”), where you let the LLM propose how to fix it. So I invoke the LLM and it proposes a fix. But what exactly is this fix? Is this another tool call? How do I actually fix it? When I ask the model for proposed fix, I provide the model with again the schema and ask it to choose an operation associated with enum or to use another tool? Yes, it usually boils down to the LLM proposing a new or modified tool call (e.g., a new operation, a new set of columns if columns were wrong, etc.). Then you parse that proposal, and re-run the tool (or a different tool) with the new parameters.

Let's say the LLM retry step fails. Then I want to invoke Human-in-the-loop (HIL). Human-in-the-loop (HIL) interactions are crucial for agentic systems. A common pattern is to add some human in the loop step after certain tool calls. These tool calls often lead to either a function call or saving of some information. I ask the human to clarify their ask by providing a list of possible operations. Therefore, they can select the operation.

Yes, that’s a very common use of Human-in-the-Loop (HIL). In agentic systems, HIL often means:
- The system tries to solve the user’s request autonomously (e.g., calling a tool, generating an answer).
- If it fails or is uncertain, the system interrupts or pauses execution and explicitly asks a human for help.
- The human clarifies or corrects some aspect—like selecting an operation from a list, fixing a tool call’s parameters, providing extra instructions, etc.
- The system then resumes execution using the human’s inputs.

This is precisely what you described: if the model cannot figure out the correct operation or produces an unsatisfactory result, you allow a human to pick from a set of operations or otherwise clarify. The human’s selection or clarification then gets fed back into the system, allowing it to continue with improved guidance.

For each invocation of the model, we must use our tool schema using pydantic to validate the input.

This part is crucial: the user may ask multiple questions in a single prompt regarding the data. Therefore we must invoke separate tool calls for each! Here is an example: 

query = "What is 3 * 12? Also, what is 11 + 49?"

llm_with_tools.invoke(query).tool_calls
# [
#     {
#         'name': 'multiply',
#         'args': { 'a': 3, 'b': 12 },
#         'id': 'call_UL7E2232GfDHIQGOM4gJfEDD'
#     },
#     {
#         'name': 'add',
#         'args': {'a': 11, 'b': 49},
#         'id': 'call_VKw8t5tpAuzvbHgdAXe9mjUx'
#     }
# ]

My with_structured_output method should account for the underlying engine whether it is TGI or vLLN because TGI has that capability to handle json out of the box using that guidance capability.

Before I was incorrectly adding few shot prompts to tool calling prompt. The correct way to do it is using the ToolMessage class. Because ToolMessage represents an actual response to a tool calling prompt! Here is an example: 

Tools and Few-Shot Examples

For more complex tool use it's very useful to add few-shot examples to the prompt. We can do this by adding AIMessages with ToolCalls and corresponding ToolMessages to our prompt.

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

examples = [
    HumanMessage(
        "What's the product of 317253 and 128472 plus four", name="example_user"
    ),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[
            {"name": "multiply", "args": {"x": 317253, "y": 128472}, "id": "1"}
        ],
    ),
    ToolMessage("16505054784", tool_call_id="1"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[{"name": "add", "args": {"x": 16505054784, "y": 4}, "id": "2"}],
    ),
    ToolMessage("16505054788", tool_call_id="2"),
    AIMessage(
        "The product of 317253 and 128472 plus four is 16505054788",
        name="example_assistant",
    ),
]

system = """You are bad at math but are an expert at using a calculator. 

Use past tool usage as an example of how to correctly use the tools."""
few_shot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        *examples,
        ("human", "{query}"),
    ]
)

chain = {"query": RunnablePassthrough()} | few_shot_prompt | llm_with_tools
chain.invoke("Whats 119 times 8 minus 20").tool_calls
# [{'name': 'multiply',
#   'args': {'a': 119, 'b': 8},
#   'id': 'call_tWwpzWqqc8dQtN13CyKZCVMe'}]



A) Excel and Table Analysis from PDFs, Word documents, and PowerPoint presentations, as well as in user prompts (time estimated is 3 weeks). Task break down:

1. Tool for Table-Based Operations
Description: Develop a single, unified mechanism (a “TableDistributionTool”) for performing various operations on tabular data (e.g., z-scores, t-tests, summaries) using Pandas as the base framework.
Justification: Perform analytics on tabular data which is not possible with a SQL Tool or with no Tool at all
Estimated Effort: 2 days

2. Structured “Operation” Schema
Description: Define a structured schema (e.g., with pydantic) that includes:

An “operation” field (using a defined list of allowable operations).
A list of data fields/columns to act on.
Justification: Ensures each request is validated against known operations (e.g., summary, group by operations, aggregations) and clarifies which columns or fields to use.
Estimated Effort: 3 days

3. Multi-Step Data Queries
Description: Allow the system to handle multiple data questions in a single user request (e.g., “What’s 3×12 and also 11+49?”). Each sub-question triggers its own operation call.
Justification: Better user experience for complex prompts, enabling separate calculations or lookups in one conversation turn.
Estimated Effort: 1 week

4. Automated “Answer Satisfaction” Check
Description: After each operation’s result, run an automated check (via the language model) to determine if the user’s question was fully answered.

If “yes,” conclude.
If “no,” prompt the model for a revised or alternative approach (“fix”).
Justification: Creates a feedback loop for error correction and more complete answers before final output.
Estimated Effort: 2 days

5. Re-Try with Model-Proposed Fix
Description: When the model indicates “no” (unsatisfied answer), it proposes an updated or new operation (e.g., switching from z-test to summary). The system parses this proposal and re-runs the operation automatically.
Justification: Reduces dead ends in user queries by automating iterative improvements—no need for immediate human intervention if a simple fix can be found.
Estimated Effort: 2 days

6. Human-in-the-Loop (HIL) Escalation
Description: If the automated fix step also fails or the system remains uncertain, pause execution. Prompt a human reviewer to:

Approve, modify, or choose from a list of possible operations.
Provide extra clarification or instructions.
Justification: Ensures high-stakes or complex tasks don’t get stuck in an automated loop; a human can correct parameters or clarify goals.
Estimated Effort: 1 week

7. Structured Output Compatibility (TGI/vLLM)
Description: Make sure the system supports structured output (e.g., JSON using TGI's Guidance capability) regardless of the underlying model backend (TGI or similar).
Justification: Maintains a consistent, validated interface for responses—especially when the model is enumerating operations or columns.
Estimated Effort: 2 days

B) Document Ingestion Enhancements

1. Extended Document Coverage
Scope: Enhance ingestion to handle tables in PDF, Word, and PowerPoint files, as well as images in Word and PowerPoint files (right now image extract and analysis is only done in PDF)
Outcome: Users can reliably ingest tabular data and images from any supported file type.
Benefit: Broader format support improves data coverage, enabling more comprehensive searches and analysis.
Estimated Effort: 1 week

2. Structured Text Extraction & Processing
Scope: Currently, text is extracted from PDF, Word, and PowerPoint documents, including from paragraphs, tables, table cells, lists, section headers, all the various xml nodes in the document. But it is just preserved as pure text and so semantic meaning is potentially lost (e.g. this paragraph is associated with this header where the header has critical summary data). This enhancement means taking those xml nodes in the various documents and converting them to semantically related html nodes to produce hierchical well-defined text segments. This will improve chunk overlap capability when doing chunking operations, so each chunk preserves its parent context.
Outcome: Text is broken into hierarchical “chunks” (e.g., one or many paragraphs per heading) for downstream tasks like vectorization and similarity search.
Benefit: Ensures high-accuracy retrieval; allows advanced context-based queries.
Estimated Effort: 1 week

3. Heading Detection & Hierarchical Outlining
Scope: Use file-specific metadata (styles, font size, formatting) to identify headings vs. normal text—even when the document lacks explicit heading tags.
Outcome: Creates a logical content hierarchy (e.g., headings, subheadings) that aligns with the author’s intent.
Benefit: Better “section-aware” indexing and improved user navigation (e.g., searching within a heading or subheading).
Estimated Effort: 2 days

4. Statistical / Style-Based Outlier Detection
Scope: For documents with irregular or customized formatting, detect “unofficial headings” (using z-scores and confidence levels for normally distributed large data set of font sizes (probability density function), etc and interquartiles for smaller data sets - e.g. less than 35 font sizes or bimodal for non-normal data sets), thus identifying the significance level to find the probable headers in pure paragraph text.
Outcome: More consistent heading identification in user-created documents (even if authors did not use built-in heading styles such as taking a piece of text in word and styling it as 24 points where surrounding paragraphs are 12 points).
Benefit: Robust ingestion across highly varied document styles; fewer missed headings.
Estimated Effort: 3 days

5. Metadata Preservation (Optional)
Scope: Capture additional metadata in addition to what we already have with page numbers. THis includes headers, footers, style indicators as metadata if needed for advanced queries.
Outcome: More granular search and analysis (e.g., searching a bullet lists associated with a specific header).
Benefit: Allows future expansions for advanced content classification.
Estimated Effort: 2 days

6. Confidence & Fallback Handling
Scope: Provide fallbacks for documents that break standard formatting (e.g., unusual structures, missing style data).
Outcome: Graceful degradation—system extracts what it can and logs any segments it cannot parse with high confidence (e.g. failure to identify the significance level).
Benefit: Minimizes ingestion failures and ensures partial data is still captured rather than lost.
Estimated Effort: 2 days

C) Segmenting GuideWell Chat data for separate business units

1. Access Partitioning by Business Unit
Scope: Split the chat data storage so that each business unit (e.g., Business A vs. Business B) can only view and manage their own sets of collections.
Outcome: Guarantee that users from Business A see only collections “A, B, and C,” while users from Business B see only collections “D, E, and F,” ensuring no cross-business data leakage. This will require infrastructure changes
Benefit: Strict data isolation meets compliance and privacy requirements for multi-tenant systems.
Estimated Effort: 1 week

2. Identity & Routing Logic
Description: Update the application FastAPI routes, Mongo client, consuming RBAC (role-based permissions) to recognize the user’s “business unit” (via credentials or session tokens) and automatically route read/write operations to the relevant collections.
Goal: Ensure that, upon sign-in, the system determines the user’s business unit and allows access solely to that unit’s data.
Benefit: Simplifies user workflows—no risk of accidentally querying or writing to unauthorized data.
Estimated Effort: 3 days

3. CRUD Operations with Restricted Visibility
Description: For each Create, Read, Update, and Delete request (on conversations, messages), the application will confirm the user’s business unit and operate only on the matching collections.
Goal: Prevent the possibility of an update or retrieval affecting data outside the user’s scope.
Benefit: Maintains data integrity and trust in a multi-tenant environment.
Estimated Effort: 3 days

4. Security Enforcement & Auditing
Description: Integrate checks to log any unauthorized access attempts, and provide an audit trail of operations by business unit.
Goal: Detect and document security issues (e.g., attempts to retrieve or modify the wrong business unit’s data).
Benefit: Offers compliance transparency and ensures immediate awareness of potential breaches.
Estimated Effort: 2 days

5. Configuration & Deployment
Description: Roll out changes to MongoDB and the FastAPI service so that each environment (e.g., test, stage, production) has correct mappings of “business unit to allowed collections.”
Goal: Align environment configuration with business structures—any new business unit gets its own isolated set of collections.
Benefit: Streamlined onboarding of future business units, plus simpler maintenance due to clear, structured data segmentation.
Estimated Effort: 2 days

D) Redis Monitoring of vector data and LLM monitoring of runnables (e.g. calls made to the model, generation and response times for each request) 

TBD

