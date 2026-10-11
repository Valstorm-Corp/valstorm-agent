use once_cell::sync::Lazy;
use regex::Regex;

struct SecretPattern {
    regex: Regex,
    replacement: &'static str,
}

static PATTERNS: Lazy<Vec<SecretPattern>> = Lazy::new(|| {
    vec![
        // OpenAI keys
        SecretPattern {
            regex: Regex::new(r"\b(?:sk-proj|sk-org|sk-admin)-[a-zA-Z0-9_\-]{20,}\b").unwrap(),
            replacement: "<REDACTED_OPENAI_KEY>",
        },
        SecretPattern {
            regex: Regex::new(r"\bsk-[a-zA-Z0-9]{20,}\b").unwrap(),
            replacement: "<REDACTED_OPENAI_KEY>",
        },
        // Anthropic keys
        SecretPattern {
            regex: Regex::new(r"\bsk-ant-[a-zA-Z0-9_\-]{20,}\b").unwrap(),
            replacement: "<REDACTED_ANTHROPIC_KEY>",
        },
        // Google Gemini keys
        SecretPattern {
            regex: Regex::new(r"\bAIzaSy[A-Za-z0-9_\-]{33}\b").unwrap(),
            replacement: "<REDACTED_GOOGLE_API_KEY>",
        },
        // GitHub tokens
        SecretPattern {
            regex: Regex::new(r"\bghp_[a-zA-Z0-9]{36,}\b").unwrap(),
            replacement: "<REDACTED_GITHUB_PAT>",
        },
        SecretPattern {
            regex: Regex::new(r"\bgithub_pat_[a-zA-Z0-9_]{60,}\b").unwrap(),
            replacement: "<REDACTED_GITHUB_FINE_GRAINED_PAT>",
        },
        SecretPattern {
            regex: Regex::new(r"\bgh[ousr]_[a-zA-Z0-9]{36,}\b").unwrap(),
            replacement: "<REDACTED_GITHUB_TOKEN>",
        },
        // Slack tokens
        SecretPattern {
            regex: Regex::new(r"\bxox[baprs]-[0-9]{10,14}-[0-9]{10,14}[a-zA-Z0-9_\-]*\b").unwrap(),
            replacement: "<REDACTED_SLACK_TOKEN>",
        },
        // Twilio tokens
        SecretPattern {
            regex: Regex::new(r"\bSK[0-9a-fA-F]{32}\b").unwrap(),
            replacement: "<REDACTED_TWILIO_API_KEY>",
        },
        // AWS keys
        SecretPattern {
            regex: Regex::new(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b").unwrap(),
            replacement: "<REDACTED_AWS_ACCESS_KEY_ID>",
        },
        // Bearer tokens
        SecretPattern {
            regex: Regex::new(r"(?i)\bBearer\s+[a-zA-Z0-9_\-\.]{20,}\b").unwrap(),
            replacement: "Bearer <REDACTED_BEARER_TOKEN>",
        },
        // Private keys
        SecretPattern {
            regex: Regex::new(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----").unwrap(),
            replacement: "<REDACTED_PRIVATE_KEY_BLOCK>",
        },
    ]
});

pub fn fast_sanitize_text_rs(text: &str) -> String {
    let mut current = text.to_string();
    for p in PATTERNS.iter() {
        if p.regex.is_match(&current) {
            current = p.regex.replace_all(&current, p.replacement).to_string();
        }
    }
    current
}
