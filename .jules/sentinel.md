## 2024-05-24 - Predictable Temporary File Generation in Streamlit File Processors
**Vulnerability:** Deterministic filename generation using SHA256 of file contents for temporary uploads.
**Learning:** Using predictable paths (even hashes) in world-writable or shared directories like the default CWD allows for file overwriting, race conditions, and unauthorized local file access.
**Prevention:** Always use `tempfile.NamedTemporaryFile` for generating temporary files to guarantee atomicity and uniqueness.
