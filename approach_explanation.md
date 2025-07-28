# Approach Explanation

This document outlines the technical methodology for the PDF processing solution. The approach is engineered to meet the strict performance and resource constraints outlined in the challenge by avoiding large machine learning models. Instead, it employs a sophisticated, **heuristic-based system** that intelligently classifies documents and applies context-specific rules to extract their structure.

---

## Core Methodology: A Multi-Stage Heuristic Engine

The solution processes PDFs through a sequential pipeline orchestrated by the `IntelligentPDFExtractor` class. The core logic resides within the `IntelligentDocumentAnalyzer`, which adapts its strategy based on the document's content and style.

### 1. Document Classification

The first and most critical step is to determine the document's category. This classification dictates which set of extraction rules will be used in subsequent steps.

* **Keyword Scoring**: The `IntelligentDocumentAnalyzer` analyzes the entire document's text against predefined keyword lists for types such as **technical, resume, business, and educational**.
* **Weighted Heuristics**: A scoring system classifies the document based on keyword frequency and specific contextual clues, giving higher weight to more definitive terms (e.g., "cgpa" for resume detection).

### 2. Detailed Content & Style Extraction

The system goes beyond simple text extraction to capture rich formatting and positional data, which are crucial inputs for the heuristic engine.

* **Deep PDF Parsing**: Using the **PyMuPDF** library, the script iterates through every text block on each page.
* **Rich Metadata Capture**: For each block, it extracts not just the text but also crucial metadata, including **font size, font name, boldness, and position**. This data is stored in a structured `TextBlock` object for analysis.
* **Font Analysis**: The system calculates the median font size using **NumPy** to establish a baseline "body text" size. This is a key reference point for identifying headings, which are typically larger.

### 3. Context-Aware Structure Extraction

This is the core of the solution, where the system uses the document type and rich metadata to identify the title and hierarchical outline.

* **Dynamic Rule Application**: Based on the document type detected in Step 1, the system selects a specialized set of regular expressions and rules to find headings. For example, the patterns to find headings in a 'technical' document differ significantly from those for a 'resume'.
* **Multi-Factor Heading Identification**: A block of text is identified as a "true heading" only if it satisfies a combination of criteria, including passing document-specific regex patterns, having a font size significantly larger than the body text, and being **bold** or in **ALL CAPS**.
* **Hierarchical Level Assignment**: Once headings are identified, they are assigned a level (`H1`, `H2`, etc.) based on a final scoring system that considers font size, style, and capitalization, creating a structured document outline.

---

## Key Libraries and Rationale

The choice of libraries was driven by the need for performance and precision while adhering to the challenge constraints.

* **PyMuPDF**: Selected for its high speed and its ability to provide detailed, low-level information about text blocks, including font styles and exact coordinates, which are essential for the heuristic engine.
* **NumPy**: Used for efficient numerical calculations, specifically for determining the median font size across the document.

---

## Containerization and Execution

The solution is containerized using Docker for consistent and platform-independent execution.

* **Base Image**: The `Dockerfile` uses a standard `python:3.10` image built for the required `linux/amd64` architecture.
* **Execution**: The container's command is set to directly run the `process_pdfs.py` script, which automatically scans the `/app/input` directory and writes results to `/app/output`.