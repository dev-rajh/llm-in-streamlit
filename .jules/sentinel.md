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

## 2026-06-04 - Streamlit Initialization Unhandled Exception (Information Disclosure)
**Vulnerability:** The application was calling `ollama.list()['models']` during the Streamlit sidebar initialization without a `try...except` block.
**Learning:** If the required backend service (e.g., Ollama) is not running when the Streamlit UI initializes, the application instantly crashes, dumping an unhandled exception stack trace to the frontend, revealing internal directory structures and application state to the user.
**Prevention:** All external API calls, even during Streamlit UI initialization or within sidebars, must be wrapped in `try...except` blocks. Use graceful error handling (e.g., `st.error` and `st.stop()`) to halt UI rendering securely rather than crashing. Additionally, wrap these initializers in `@st.cache_data` to prevent repetitive failing calls on UI reruns.

## 2026-06-15 - Unbounded Persistent Storage of Temporary Vector Stores (DoS)
**Vulnerability:** The application was persistently saving temporary, query-specific vector store objects (FAISS indices and chunks) to disk inside `PDFHelper.ask` using unique directory names (`uuid.uuid4()`), without any reliable cleanup mechanism.
**Learning:** This leads to unbounded disk usage, as every query creates a new directory. Over time, or under heavy usage/abuse, this will fill up the storage disk, resulting in a Denial of Service (DoS) due to resource exhaustion.
**Prevention:** Avoid persistent disk saves for temporary, ephemeral, or query-specific data structures. Use in-memory objects (e.g., passing `storing_path=None` to initializers) for interactions that do not require long-term persistence.

## 2026-06-25 - Infinite Loop in Text Splitting (DoS)
**Vulnerability:** The application was not validating input parameters `chunk_size` and `chunk_overlap` in `split_text_into_chunks()`.
**Learning:** If an attacker or user configures `chunk_overlap >= chunk_size`, the indexing loop (`start += chunk_size - chunk_overlap`) will never advance or will step backward, resulting in an infinite loop that crashes the process and exhausts CPU/Memory resources (Denial of Service).
**Prevention:** Always validate size parameters on chunking functions. Ensure `chunk_size > chunk_overlap` before starting any loops.

## 2026-06-25 - Synchronous File I/O UI Blocking (Localized DoS)
**Vulnerability:** The application was synchronously writing and reading the `chats.json` state file on the main thread during Streamlit user interactions.
**Learning:** In a Streamlit environment, any blocking synchronous operation (like heavy file I/O on large JSON objects) will stall the main execution thread. If `chats.json` grows large, or the disk is slow, saving on every message will cause severe UI freezing, resulting in a localized Denial of Service (DoS) for the user. Additionally, unhandled exceptions during file reading/writing can lead to application crashes.
**Prevention:** Always offload non-critical file I/O operations (like saving state) to a background thread using `concurrent.futures.ThreadPoolExecutor`. Ensure thread safety by deep copying state objects (like `st.session_state`) before passing them to the background worker. Wrap file loading operations in `try...except` blocks to handle malformed or locked files gracefully without crashing the UI.
