## 2026-05-20 - Insecure Predictable Temporary Files
**Vulnerability:** The application was manually creating temporary files in the current working directory using a predictable naming convention (`temp_<sha256>.pdf`) based on file content.
**Learning:** This exposes the application to temporary file vulnerabilities such as local file inclusion, unintended overwrites, and pollutes the application directory if cleanup fails. The usage of `tempfile` should always be preferred over manual hashing for temp file management.
**Prevention:** Enforce the usage of the `tempfile` module (`NamedTemporaryFile`) for any temporary file operations, ensuring files are placed in the OS's dedicated temporary directory with appropriately restrictive permissions.

## 2024-05-24 - Temporary File Disk Leaks
**Vulnerability:** Temporary file creation in Streamlit PDF processing logic lacked proper `try...finally` resource cleanup, risking resource exhaustion/disk leaks on file processing errors.
**Learning:** Hard failures during file parsing (`pypdf.PdfReader` throwing errors, or `write` operations failing on full disk) bypassed standard `os.unlink` calls located at the end of the functions.
**Prevention:** Always initialize temp file path variables to `None` prior to a `try` block. Then assign the temp path _before_ attempting `write` operations. Place all processing logic in the `try` block, and use a `finally` block to explicitly `os.remove` the file.
