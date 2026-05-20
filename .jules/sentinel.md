## 2026-05-20 - Insecure Predictable Temporary Files
**Vulnerability:** The application was manually creating temporary files in the current working directory using a predictable naming convention (`temp_<sha256>.pdf`) based on file content.
**Learning:** This exposes the application to temporary file vulnerabilities such as local file inclusion, unintended overwrites, and pollutes the application directory if cleanup fails. The usage of `tempfile` should always be preferred over manual hashing for temp file management.
**Prevention:** Enforce the usage of the `tempfile` module (`NamedTemporaryFile`) for any temporary file operations, ensuring files are placed in the OS's dedicated temporary directory with appropriately restrictive permissions.
