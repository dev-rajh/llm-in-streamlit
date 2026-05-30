## 2026-05-20 - Insecure Predictable Temporary Files
**Vulnerability:** The application was manually creating temporary files in the current working directory using a predictable naming convention (`temp_<sha256>.pdf`) based on file content.
**Learning:** This exposes the application to temporary file vulnerabilities such as local file inclusion, unintended overwrites, and pollutes the application directory if cleanup fails. The usage of `tempfile` should always be preferred over manual hashing for temp file management.
**Prevention:** Enforce the usage of the `tempfile` module (`NamedTemporaryFile`) for any temporary file operations, ensuring files are placed in the OS's dedicated temporary directory with appropriately restrictive permissions.

## 2024-05-20 - Unhandled Temporary File Resource Leaks
**Vulnerability:** The application was creating temporary files but not ensuring their cleanup in case of failures during file writing or parsing. Specifically, `filename` was assigned *after* writing the buffer, so if writing failed (e.g. disk full), the file would leak. Additionally, `PDFHelper.ask` didn't use a `try...finally` block for cleanup.
**Learning:** Even when using `tempfile.NamedTemporaryFile` with manual cleanup, strict exception handling guarantees are needed. Operations that write to disk or parse complex formats (like PDFs) can fail and abandon file handles/disk space.
**Prevention:** Always initialize the temporary filename variable *before* the `try` block or immediately after opening it, and wrap the entire lifecycle in a `try...finally` block to guarantee `os.unlink()` runs regardless of exceptions.
