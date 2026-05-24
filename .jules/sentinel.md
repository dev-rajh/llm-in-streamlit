## 2024-05-18 - Fix custom hash-based temporary filenames
**Vulnerability:** The application was using custom hash-based temporary filenames (`hashlib.sha256(file.getvalue()).hexdigest()`) to save uploaded PDF files to disk.
**Learning:** This approach is vulnerable to file collisions and race condition attacks, as the custom hash-based temporary filenames are predictable.
**Prevention:** Always use secure, built-in libraries like `tempfile.NamedTemporaryFile` instead of manually rolling hash-based temporary filenames, to ensure temporary files are generated securely and uniquely.