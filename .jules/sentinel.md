## 2026-05-20 - Insecure Predictable Temporary Files
**Vulnerability:** The application was manually creating temporary files in the current working directory using a predictable naming convention (`temp_<sha256>.pdf`) based on file content.
**Learning:** This exposes the application to temporary file vulnerabilities such as local file inclusion, unintended overwrites, and pollutes the application directory if cleanup fails. The usage of `tempfile` should always be preferred over manual hashing for temp file management.
**Prevention:** Enforce the usage of the `tempfile` module (`NamedTemporaryFile`) for any temporary file operations, ensuring files are placed in the OS's dedicated temporary directory with appropriately restrictive permissions.

## 2026-05-20 - Temporary File Resource Exhaustion (DoS)
**Vulnerability:** The application was not guaranteeing the cleanup of temporary PDF files in `PDFHelper.ask` if an exception occurred during file writing or reading. Additionally, in both `process_pdf` and `PDFHelper.ask`, the temporary filename was extracted *after* writing the file content.
**Learning:** If `temp_file.write()` fails (e.g., due to a full disk or an interrupt), an exception is raised before the filename variable is assigned. This makes it impossible for a `finally` block to locate and delete the partially written file, leading to disk leaks and potential Denial of Service (DoS) via resource exhaustion over time.
**Prevention:** Always initialize the filename variable to `None` outside the `try` block, and assign `filename = temp_file.name` *before* executing write operations. Ensure all temporary file processing is wrapped in a `try...finally` block that verifies file existence before unlinking.

## 2026-06-04 - Unhandled Exceptions Leaking Stack Traces (Information Disclosure/DoS)
**Vulnerability:** The application was making external network calls to the Ollama LLM backend (`client.chat(...)`) without `try...except` wrappers.
**Learning:** In Streamlit applications, unhandled exceptions typically crash the current script execution and print the raw stack trace directly into the frontend UI. This leaks internal file paths, dependency versions, and environmental state to the user. Additionally, a persistent failure from the LLM backend acts as a localized DoS by repeatedly crashing user sessions.
**Prevention:** Always wrap external API calls and unreliable operations in `try...except` blocks. Catch specific or generic exceptions, log them securely on the server side, and return a sanitized, user-friendly error string to the frontend.
