use once_cell::sync::Lazy;
use regex::{Captures, Regex};
use std::borrow::Cow;

struct SecretPattern {
    regex: Regex,
    replacement: &'static str,
}

static PATTERNS: Lazy<Vec<SecretPattern>> = Lazy::new(|| {
    vec![
        // Private keys & certificates
        SecretPattern {
            regex: Regex::new(
                r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----",
            )
            .unwrap(),
            replacement: "<REDACTED_PRIVATE_KEY_BLOCK>",
        },
        SecretPattern {
            regex: Regex::new(
                r"-----BEGIN CERTIFICATE-----[\s\S]+?-----END CERTIFICATE-----",
            )
            .unwrap(),
            replacement: "<REDACTED_CERTIFICATE_BLOCK>",
        },
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
        // Google Gemini / Cloud API keys
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
        // Slack tokens & webhooks
        SecretPattern {
            regex: Regex::new(r"\bxox[baprs]-[0-9]{10,14}-[0-9]{10,14}[a-zA-Z0-9_\-]*\b").unwrap(),
            replacement: "<REDACTED_SLACK_TOKEN>",
        },
        SecretPattern {
            regex: Regex::new(
                r"https://hooks\.slack\.com/services/T[0-9A-Z]+/B[0-9A-Z]+/[0-9a-zA-Z]+",
            )
            .unwrap(),
            replacement: "<REDACTED_SLACK_WEBHOOK>",
        },
        // AWS Access Key IDs
        SecretPattern {
            regex: Regex::new(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b").unwrap(),
            replacement: "<REDACTED_AWS_ACCESS_KEY_ID>",
        },
        // Stripe keys
        SecretPattern {
            regex: Regex::new(r"\b(?:sk|pk|rk)_(?:live|test)_[0-9a-zA-Z]{24,}\b").unwrap(),
            replacement: "<REDACTED_STRIPE_KEY>",
        },
        // Twilio API key
        SecretPattern {
            regex: Regex::new(r"\bSK[0-9a-fA-F]{32}\b").unwrap(),
            replacement: "<REDACTED_TWILIO_API_KEY>",
        },
        // SendGrid API key
        SecretPattern {
            regex: Regex::new(r"\bSG\.[a-zA-Z0-9_\-\.]{60,}\b").unwrap(),
            replacement: "<REDACTED_SENDGRID_API_KEY>",
        },
        // E2B API key
        SecretPattern {
            regex: Regex::new(r"\be2b_[a-zA-Z0-9_\-]{20,}\b").unwrap(),
            replacement: "<REDACTED_E2B_KEY>",
        },
        // Database URIs with credentials (including password-only redis://:pass@host and SQLAlchemy drivers)
        SecretPattern {
            regex: Regex::new(
                r#"(?i)(\b(?:mongodb(?:\+srv)?|postgres(?:ql)?(?:\+[a-z0-9_]+)?|mysql(?:\+[a-z0-9_]+)?|rediss?|amqps?|mssql(?:\+[a-z0-9_]+)?|sqlserver)://[^:/\s@"']*:)(?:<REDACTED_DB_PASSWORD>|[^@\s"'<>]+)(@)"#,
            )
            .unwrap(),
            replacement: "${1}<REDACTED_DB_PASSWORD>${2}",
        },
        // Azure Storage AccountKey in connection strings
        SecretPattern {
            regex: Regex::new(r#"(?i)(\bAccountKey\s*=\s*)([^;\s"'<>]{16,})"#).unwrap(),
            replacement: "${1}<REDACTED_AZURE_KEY>",
        },
        // Azure Communication Services (ACS) connection string accesskey
        SecretPattern {
            regex: Regex::new(
                r#"(?i)(endpoint\s*=\s*https?://[^;\s"']+;\s*accesskey\s*=\s*)([^;\s"'<>]{16,})"#,
            )
            .unwrap(),
            replacement: "${1}<REDACTED_AZURE_KEY>",
        },
        SecretPattern {
            regex: Regex::new(r#"(?i)(;\s*accesskey\s*=\s*)([^;\s"'<>]{16,})"#).unwrap(),
            replacement: "${1}<REDACTED_AZURE_KEY>",
        },
        // Azure Service Bus / Event Hub SharedAccessKey
        SecretPattern {
            regex: Regex::new(r#"(?i)(\bSharedAccessKey\s*=\s*)([^;\s"'<>]{16,})"#).unwrap(),
            replacement: "${1}<REDACTED_AZURE_KEY>",
        },
        // Azure SAS URL signature parameter
        SecretPattern {
            regex: Regex::new(
                r#"(?i)(https?://[^\s"'<>]*(?:windows\.net|azure\.com)[^\s"'<>]*[?&]sig=)([^&\s"'<>]{16,})"#,
            )
            .unwrap(),
            replacement: "${1}<REDACTED_AZURE_KEY>",
        },
        // JSON Web Tokens (JWT)
        SecretPattern {
            regex: Regex::new(
                r"\beyJ[a-zA-Z0-9_\-]{10,}\.eyJ[a-zA-Z0-9_\-]{10,}\.[a-zA-Z0-9_\-]{10,}\b",
            )
            .unwrap(),
            replacement: "<REDACTED_JWT_TOKEN>",
        },
        // Bearer tokens
        SecretPattern {
            regex: Regex::new(r"(?i)\bBearer\s+[a-zA-Z0-9_\-\.]{20,}\b").unwrap(),
            replacement: "Bearer <REDACTED_BEARER_TOKEN>",
        },
        // HTTP API key headers (x-api-key, Ocp-Apim-Subscription-Key, api-key, x-auth-token, x-access-token)
        SecretPattern {
            regex: Regex::new(
                r#"(?i)(["']?(?:x-api-key|x-auth-token|x-access-token|ocp-apim-subscription-key|api-key)["']?\s*[:=]\s*["']?)([^\s"'`,;}\[\]<>]{8,})"#,
            )
            .unwrap(),
            replacement: "${1}<REDACTED_API_KEY>",
        },
        // Quoted key-value credential / password assignments
        SecretPattern {
            regex: Regex::new(
                r#"(?i)(["']?(?:api[_-]?key|api[_-]?secret|secret[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|client[_-]?secret|private[_-]?key|signing[_-]?secret|webhook[_-]?secret|encryption[_-]?key|master[_-]?key|service[_-]?role[_-]?key|db[_-]?password|smtp[_-]?password|password|passwd)["']?\s*[:=]\s*["'])([^"'\r\n<>]{6,})(["'])"#,
            )
            .unwrap(),
            replacement: "${1}<REDACTED_SECRET>${3}",
        },
    ]
});

static UNQUOTED_CRED_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?i)\b((?:password|passwd|pwd|db[_-]?password|smtp[_-]?password|client[_-]?secret|api[_-]?key|api[_-]?secret|secret[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|private[_-]?key|signing[_-]?secret|webhook[_-]?secret)\s*=\s*)([^\s&;,"'`<>()\[\]{}]{6,})"#,
    )
    .unwrap()
});

static PAN_CANDIDATE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"\d{4}(?:-\d{4}){3}-\d{3}|\d{4}(?: \d{4}){3} \d{3}|\d{4}(?:-\d{4}){2}-\d{1,4}|\d{4}(?: \d{4}){2} \d{1,4}|\d{4}-\d{6}-\d{4,5}|\d{4} \d{6} \d{4,5}|\d{4}-\d{4}-\d{5}|\d{4} \d{4} \d{5}|\d{13,19}",
    )
    .unwrap()
});

static CVV_CONTEXT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?i)\b((?:card[ _-]?)?(?:cvv2?|cvc2?|cvn|security[ _-]?code|card[ _-]?verification[ _-]?(?:value|code|number))["']?(?:\s+(?:code|number|is))?\s*[:=]?\s*["']?)(\d{3,4})\b"#,
    )
    .unwrap()
});

static EXPIRY_CONTEXT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?i)\b((?:card[ _-]?)?(?:exp(?:iry|iration)?(?:[ _-]?date)?|valid[ _-]?thru|good[ _-]?thru|expires)["']?(?:\s+(?:date|on|at|is))?\s*[:=]?\s*["']?)((?:0[1-9]|1[0-2])\s*[/-]\s*(?:20\d{2}|\d{2}))\b"#,
    )
    .unwrap()
});

static EXPIRY_MONTH_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?i)\b((?:card[ _-]?)?exp(?:iry|iration)?[ _-]?(?:month|mm)["']?\s*[:=]\s*["']?)(0?[1-9]|1[0-2])\b"#,
    )
    .unwrap()
});

static EXPIRY_YEAR_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?i)\b((?:card[ _-]?)?exp(?:iry|iration)?[ _-]?(?:year|yy|yyyy)["']?\s*[:=]\s*["']?)(20\d{2}|\d{2})\b"#,
    )
    .unwrap()
});

// Adjacent tabular/CSV/pipe/space expiry & CVV immediately following a redacted PAN on the same line
static PAN_ADJACENT_EXP_CVV_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(<REDACTED_PAYMENT_CARD>[ \t]*[,|\t/ ][ \t]*)(?:0[1-9]|1[0-2])[/-](?:20\d{2}|\d{2})([ \t]*[,|\t/ ][ \t]*)\d{3,4}\b",
    )
    .unwrap()
});

static PAN_ADJACENT_CVV_EXP_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(<REDACTED_PAYMENT_CARD>[ \t]*[,|\t/ ][ \t]*)\d{3,4}([ \t]*[,|\t/ ][ \t]*)(?:0[1-9]|1[0-2])[/-](?:20\d{2}|\d{2})\b",
    )
    .unwrap()
});

static PAN_ADJACENT_EXP_ONLY_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(<REDACTED_PAYMENT_CARD>[ \t]*[,|\t/ ][ \t]*)(?:0[1-9]|1[0-2])[/-](?:20\d{2}|\d{2})\b",
    )
    .unwrap()
});

static PAN_ADJACENT_CVV_ONLY_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(<REDACTED_PAYMENT_CARD>[ \t]*[,|\t][ \t]*)\d{3,4}\b").unwrap()
});

const IGNORED_UNQUOTED_VALUES: &[&str] = &[
    "true",
    "false",
    "none",
    "null",
    "undefined",
    "string",
    "optional",
    "boolean",
    "number",
    "localhost",
    "default",
    "required",
    "masked",
    "redacted",
    "bearer",
    "basic",
];

const CODE_EXPR_PREFIXES: &[&str] = &[
    "self.",
    "this.",
    "os.",
    "env.",
    "process.",
    "settings.",
    "config.",
    "props.",
    "args.",
    "req.",
    "ctx.",
    "params.",
    "data.",
    "user.",
    "payload.",
    "body.",
    "headers.",
];

fn is_code_expression_value(key_part: &str, val: &str) -> bool {
    let lower_val = val.to_ascii_lowercase();
    if IGNORED_UNQUOTED_VALUES.contains(&lower_val.as_str()) {
        return true;
    }
    if CODE_EXPR_PREFIXES
        .iter()
        .any(|prefix| lower_val.starts_with(prefix))
    {
        return true;
    }
    // Skip env variable references like $DB_PASSWORD
    if let Some(rest) = val.strip_prefix('$') {
        if !rest.is_empty()
            && rest
                .bytes()
                .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
        {
            return true;
        }
    }
    // Skip variable-to-keyword-arg passthroughs like `password=password` or `client_secret=client_secret`
    let key_name = key_part
        .trim_end_matches(|c: char| c == '=' || c.is_ascii_whitespace())
        .to_ascii_lowercase();
    if lower_val == key_name {
        return true;
    }
    false
}

fn passes_luhn(digits: &[u8]) -> bool {
    let mut sum: u32 = 0;
    let mut double = false;
    for &b in digits.iter().rev() {
        let mut d = (b - b'0') as u32;
        if double {
            d *= 2;
            if d > 9 {
                d -= 9;
            }
        }
        sum += d;
        double = !double;
    }
    sum % 10 == 0
}

fn parse_prefix(digits: &[u8], n: usize) -> u32 {
    let mut val: u32 = 0;
    for &b in &digits[..n] {
        val = val * 10 + (b - b'0') as u32;
    }
    val
}

fn is_valid_card_iin_and_length(digits: &[u8]) -> bool {
    let len = digits.len();
    if !(13..=19).contains(&len) {
        return false;
    }
    let p1 = (digits[0] - b'0') as u32;
    let p2 = parse_prefix(digits, 2);
    let p3 = parse_prefix(digits, 3);
    let p4 = parse_prefix(digits, 4);
    let p6 = parse_prefix(digits, 6);

    // Visa: 4, length 13, 16, 19
    if p1 == 4 && matches!(len, 13 | 16 | 19) {
        return true;
    }
    // Mastercard: 51-55 or 2221-2720, length 16
    if len == 16 && ((51..=55).contains(&p2) || (2221..=2720).contains(&p4)) {
        return true;
    }
    // American Express: 34 or 37, length 15
    if len == 15 && (p2 == 34 || p2 == 37) {
        return true;
    }
    // Discover: 6011, 65, 644-649, 622126-622925, length 16-19
    if (16..=19).contains(&len)
        && (p4 == 6011
            || p2 == 65
            || (644..=649).contains(&p3)
            || (622126..=622925).contains(&p6))
    {
        return true;
    }
    // Diners Club: 300-305, 36, 38-39, length 14-19
    if (14..=19).contains(&len)
        && ((300..=305).contains(&p3) || p2 == 36 || p2 == 38 || p2 == 39)
    {
        return true;
    }
    // JCB: 3528-3589, length 16-19
    if (16..=19).contains(&len) && (3528..=3589).contains(&p4) {
        return true;
    }
    // UnionPay: 62, length 16-19
    if (16..=19).contains(&len) && p2 == 62 {
        return true;
    }
    // Maestro: 50, 56-58, 6304, 6759, 6761-6763, length 13-19
    if (13..=19).contains(&len)
        && (p2 == 50
            || (56..=58).contains(&p2)
            || p4 == 6304
            || p4 == 6759
            || (6761..=6763).contains(&p4))
    {
        return true;
    }
    false
}

fn has_valid_pan_left_boundary(bytes: &[u8], start: usize) -> bool {
    if start == 0 {
        return true;
    }
    let prev = bytes[start - 1];
    if prev.is_ascii_alphanumeric() || prev == b'_' || prev == b'+' || prev == b'.' {
        return false;
    }
    // Reject prefixed record IDs like `clm-4111111111111111` or `pat-4111-1111-1111-1111`
    if prev == b'-' && start >= 2 {
        let before_dash = bytes[start - 2];
        if before_dash.is_ascii_alphanumeric() || before_dash == b'_' {
            return false;
        }
    }
    true
}

fn has_valid_pan_right_boundary(bytes: &[u8], end: usize) -> bool {
    if end >= bytes.len() {
        return true;
    }
    let next = bytes[end];
    if next.is_ascii_alphanumeric() || next == b'_' {
        return false;
    }
    // Reject decimal numbers like `4111111111111111.5`, allow sentence-ending `4111111111111111.`
    if next == b'.' && end + 1 < bytes.len() && bytes[end + 1].is_ascii_digit() {
        return false;
    }
    // Reject longer hyphenated identifiers like `4111-1111-1111-1111-ABCD` or `-2026`
    if next == b'-' && end + 1 < bytes.len() && bytes[end + 1].is_ascii_alphanumeric() {
        return false;
    }
    true
}

fn is_valid_pan_slice(slice: &str) -> bool {
    let mut digits: [u8; 24] = [0; 24];
    let mut count = 0usize;
    for b in slice.bytes() {
        if b.is_ascii_digit() {
            if count >= digits.len() {
                return false;
            }
            digits[count] = b;
            count += 1;
        }
    }
    let digit_slice = &digits[..count];
    is_valid_card_iin_and_length(digit_slice) && passes_luhn(digit_slice)
}

fn redact_payment_cards_rs(text: &str) -> String {
    if !text.bytes().any(|b| b.is_ascii_digit()) {
        return text.to_string();
    }
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut last_end = 0usize;
    let mut search_pos = 0usize;

    while let Some(m) = PAN_CANDIDATE_RE.find_at(text, search_pos) {
        let start = m.start();
        let end = m.end();
        let matched = m.as_str();

        if has_valid_pan_left_boundary(bytes, start) {
            if has_valid_pan_right_boundary(bytes, end) && is_valid_pan_slice(matched) {
                out.push_str(&text[last_end..start]);
                out.push_str("<REDACTED_PAYMENT_CARD>");
                last_end = end;
                search_pos = end;
                continue;
            }
            // Fallback for 16-digit space-separated PAN followed by a 3-digit CVV or 4-digit year
            // e.g. `4111 1111 1111 1111 742` matched 19 digits, try the 16-digit `4 4 4 4` prefix
            if matched.len() == 23
                && matched.as_bytes()[4] == b' '
                && matched.as_bytes()[9] == b' '
                && matched.as_bytes()[14] == b' '
                && matched.as_bytes()[19] == b' '
            {
                let prefix_end = start + 19;
                if has_valid_pan_right_boundary(bytes, prefix_end)
                    && is_valid_pan_slice(&text[start..prefix_end])
                {
                    out.push_str(&text[last_end..start]);
                    out.push_str("<REDACTED_PAYMENT_CARD>");
                    last_end = prefix_end;
                    search_pos = prefix_end;
                    continue;
                }
            }
        }
        // Advance past this match without skipping valid later spans
        search_pos = start + 1;
        while search_pos < text.len() && !text.is_char_boundary(search_pos) {
            search_pos += 1;
        }
    }

    if last_end == 0 {
        return text.to_string();
    }
    out.push_str(&text[last_end..]);
    out
}

fn redact_unquoted_credentials_rs(text: &str) -> String {
    if !UNQUOTED_CRED_RE.is_match(text) {
        return text.to_string();
    }
    UNQUOTED_CRED_RE
        .replace_all(text, |caps: &Captures| -> Cow<'_, str> {
            let prefix = &caps[1];
            let val = &caps[2];
            if is_code_expression_value(prefix, val) {
                Cow::Owned(caps[0].to_string())
            } else {
                Cow::Owned(format!("{}<REDACTED_SECRET>", prefix))
            }
        })
        .into_owned()
}

fn redact_cvv_and_expiry_rs(text: &str) -> String {
    let mut current = text.to_string();

    // Contextual CVV/CVC/security_code (avoid matching phone fragments like `cvv 123-4567`)
    if CVV_CONTEXT_RE.is_match(&current) {
        let bytes = current.as_bytes();
        let mut out = String::with_capacity(current.len());
        let mut last_end = 0usize;
        for caps in CVV_CONTEXT_RE.captures_iter(&current) {
            let full = caps.get(0).unwrap();
            let end = full.end();
            let followed_by_dash_or_dot_digit = end + 1 < bytes.len()
                && (bytes[end] == b'-' || bytes[end] == b'.')
                && bytes[end + 1].is_ascii_digit();
            if !followed_by_dash_or_dot_digit {
                out.push_str(&current[last_end..full.start()]);
                out.push_str(&caps[1]);
                out.push_str("<REDACTED_CVV>");
                last_end = end;
            }
        }
        if last_end > 0 {
            out.push_str(&current[last_end..]);
            current = out;
        }
    }

    // Contextual Card Expiry MM/YY or MM/YYYY (avoid truncating full dates like `05/15/2027`)
    if EXPIRY_CONTEXT_RE.is_match(&current) {
        let bytes = current.as_bytes();
        let mut out = String::with_capacity(current.len());
        let mut last_end = 0usize;
        for caps in EXPIRY_CONTEXT_RE.captures_iter(&current) {
            let full = caps.get(0).unwrap();
            let end = full.end();
            let followed_by_date_part = end + 1 < bytes.len()
                && (bytes[end] == b'/' || bytes[end] == b'-')
                && bytes[end + 1].is_ascii_digit();
            if !followed_by_date_part {
                out.push_str(&current[last_end..full.start()]);
                out.push_str(&caps[1]);
                out.push_str("<REDACTED_CARD_EXPIRY>");
                last_end = end;
            }
        }
        if last_end > 0 {
            out.push_str(&current[last_end..]);
            current = out;
        }
    }

    if EXPIRY_MONTH_RE.is_match(&current) {
        current = EXPIRY_MONTH_RE
            .replace_all(&current, "${1}<REDACTED_CARD_EXPIRY>")
            .into_owned();
    }
    if EXPIRY_YEAR_RE.is_match(&current) {
        current = EXPIRY_YEAR_RE
            .replace_all(&current, "${1}<REDACTED_CARD_EXPIRY>")
            .into_owned();
    }

    // PAN-adjacent tabular/CSV/pipe/space expiry & CVV
    if current.contains("<REDACTED_PAYMENT_CARD>") {
        if PAN_ADJACENT_EXP_CVV_RE.is_match(&current) {
            current = PAN_ADJACENT_EXP_CVV_RE
                .replace_all(&current, "${1}<REDACTED_CARD_EXPIRY>${2}<REDACTED_CVV>")
                .into_owned();
        }
        if PAN_ADJACENT_CVV_EXP_RE.is_match(&current) {
            current = PAN_ADJACENT_CVV_EXP_RE
                .replace_all(&current, "${1}<REDACTED_CVV>${2}<REDACTED_CARD_EXPIRY>")
                .into_owned();
        }
        if PAN_ADJACENT_EXP_ONLY_RE.is_match(&current) {
            current = PAN_ADJACENT_EXP_ONLY_RE
                .replace_all(&current, "${1}<REDACTED_CARD_EXPIRY>")
                .into_owned();
        }
        if PAN_ADJACENT_CVV_ONLY_RE.is_match(&current) {
            current = PAN_ADJACENT_CVV_ONLY_RE
                .replace_all(&current, "${1}<REDACTED_CVV>")
                .into_owned();
        }
    }

    current
}

pub fn fast_sanitize_text_rs(text: &str) -> String {
    let mut current = text.to_string();
    for p in PATTERNS.iter() {
        if p.regex.is_match(&current) {
            current = p.regex.replace_all(&current, p.replacement).to_string();
        }
    }
    current = redact_unquoted_credentials_rs(&current);
    current = redact_payment_cards_rs(&current);
    current = redact_cvv_and_expiry_rs(&current);
    current
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_visa_mastercard_amex_discover_pans() {
        let visa_plain = concat!("4111", "1111", "1111", "1111");
        let visa_dash = concat!("4242-", "4242-", "4242-", "4242");
        let visa_space = concat!("4012 ", "8888 ", "8888 ", "1881");
        let mc_5x = concat!("5555-", "5555-", "5555-", "4444");
        let mc_2x = concat!("2221 ", "0000 ", "0000 ", "0009");
        let amex_plain = concat!("3782", "822463", "10005");
        let amex_dash = concat!("3714-", "496353-", "98431");
        let amex_space = concat!("3400 ", "000000 ", "00009");
        let discover = concat!("6011-", "1111-", "1111-", "1117");
        let discover_65 = concat!("6500 ", "0000 ", "0000 ", "0002");
        let diners = concat!("3056 ", "930902 ", "5904");
        let jcb = concat!("3528-", "0000-", "0000-", "0007");
        let visa_13 = concat!("4111", "1111", "1111", "9");

        for pan in [
            visa_plain, visa_dash, visa_space, visa_13, mc_5x, mc_2x, amex_plain, amex_dash,
            amex_space, discover, discover_65, diners, jcb,
        ] {
            let input = format!("Charge card {} now.", pan);
            let out = fast_sanitize_text_rs(&input);
            assert!(
                !out.contains(pan),
                "Expected PAN {} to be redacted, got: {}",
                pan,
                out
            );
            assert_eq!(out, "Charge card <REDACTED_PAYMENT_CARD> now.");
        }
    }

    #[test]
    fn zero_false_positives_on_clinical_and_record_ids() {
        let safe_samples = [
            // 10-digit NPIs
            "Provider NPI: 1234567893 and secondary NPI 1245319599",
            // US and international phone numbers
            "Call +1-415-555-2671 or (555) 234-5678 or +49 151 2345 6789",
            // Prefixed Valstorm & clinical record IDs
            "Ticket tick_Z5NdByhscBapsiME patient pat_4111111111111111 claim clm-4111111111111111 MRN4111111111111111",
            // 16-digit numbers failing Luhn
            "Invalid card 4111-1111-1111-1112 and 5555-5555-5555-4445 and 1234-5678-9012-3456",
            // 20-digit tracking / snowflake IDs
            "Tracking 41111111111111119999 and snowflake 17325735370001234567",
            // Timestamps & full dates
            "Created 2026-11-25T22:25:37Z stamp 20261125143059 license expires 05/15/2027",
            // High-precision floats
            "Ratio 3.1415926535897932 and metric 4.1111111111111111",
            // UUID and clinical codes
            "UUID 550e8400-e29b-41d4-a716-446655440000 ICD-10 E11.9 CPT 99213 NDC 00093-3109-05",
            // Source code expressions (must not break developer code inspection)
            "password = self.password\nclient_secret = settings.client_secret\npassword = os.getenv('PASS')\npassword=${DB_PASSWORD}",
        ];

        for sample in safe_samples {
            let out = fast_sanitize_text_rs(sample);
            assert_eq!(
                out, sample,
                "Unexpected false-positive redaction!\nInput:  {}\nOutput: {}",
                sample, out
            );
        }
    }

    #[test]
    fn redacts_cvv_cvc_security_code_and_card_expiry() {
        let input = r#"{"cvv": "742", "cvc": 8431, "security_code": "019", "exp_date": "09/28", "exp_month": "09", "exp_year": "2028"}"#;
        let out = fast_sanitize_text_rs(input);
        assert!(!out.contains("742"), "out={}", out);
        assert!(!out.contains("8431"), "out={}", out);
        assert!(!out.contains("019"), "out={}", out);
        assert!(!out.contains("09/28"), "out={}", out);
        assert!(out.contains(r#""cvv": "<REDACTED_CVV>""#), "out={}", out);
        assert!(out.contains(r#""cvc": <REDACTED_CVV>"#), "out={}", out);
        assert!(out.contains(r#""security_code": "<REDACTED_CVV>""#), "out={}", out);
        assert!(out.contains(r#""exp_date": "<REDACTED_CARD_EXPIRY>""#), "out={}", out);
        assert!(out.contains(r#""exp_month": "<REDACTED_CARD_EXPIRY>""#), "out={}", out);
        assert!(out.contains(r#""exp_year": "<REDACTED_CARD_EXPIRY>""#), "out={}", out);

        let prose = "Card Security Code: 991, CVV 312, Valid Thru: 11/2029, expires 04/27";
        let out_prose = fast_sanitize_text_rs(prose);
        assert!(!out_prose.contains("991"));
        assert!(!out_prose.contains("312"));
        assert!(!out_prose.contains("11/2029"));
        assert!(!out_prose.contains("04/27"));
        assert!(out_prose.contains("<REDACTED_CVV>"));
        assert!(out_prose.contains("<REDACTED_CARD_EXPIRY>"));
    }

    #[test]
    fn redacts_csv_and_table_pan_expiry_cvv_rows() {
        let pan = concat!("4111-", "1111-", "1111-", "1111");
        let csv_row = format!("Jane Doe,{},09/28,742,90210", pan);
        let out = fast_sanitize_text_rs(&csv_row);
        assert_eq!(
            out,
            "Jane Doe,<REDACTED_PAYMENT_CARD>,<REDACTED_CARD_EXPIRY>,<REDACTED_CVV>,90210"
        );

        let spaced = format!("{} 742", concat!("4111 ", "1111 ", "1111 ", "1111"));
        let out_spaced = fast_sanitize_text_rs(&spaced);
        assert!(!out_spaced.contains("4111"));
        assert!(out_spaced.contains("<REDACTED_PAYMENT_CARD>"));
    }

    #[test]
    fn redacts_database_uris_azure_conn_strings_headers_and_passwords() {
        let pg = concat!("postgres://admin:", "PgSecret99!", "@db.internal:5432/valstorm");
        let pg_async = concat!("postgresql+asyncpg://app:", "AsyncPass123", "@pg.local:5432/app");
        let redis_user = concat!("redis://default:", "RedisSecret88!", "@redis.local:6379/0");
        let redis_nouser = concat!("rediss://:", "AzureRedisKey999=", "@cache.redis.cache.windows.net:6380/0");
        let mongo = concat!("mongodb+srv://root:", "MongoPass777!", "@cluster0.mongodb.net/prod");
        let az_storage = concat!(
            "DefaultEndpointsProtocol=https;AccountName=valstormprod;AccountKey=",
            "MockAzureStorageKeyBase64Value1234567890==",
            ";EndpointSuffix=core.windows.net"
        );
        let az_acs = concat!(
            "AZURE_ACS_EMAIL_CONNECTION_STRING=endpoint=https://valstorm-acs.communication.azure.com/;accesskey=",
            "MockAcsAccessKeyBase64Value9876543210=="
        );
        let az_sb = concat!(
            "Endpoint=sb://valstorm.servicebus.windows.net/;SharedAccessKeyName=Root;SharedAccessKey=",
            "MockServiceBusSharedAccessKey1234567890="
        );
        let x_api = concat!("x-api-key: ", "val_prod_api_key_9988776655");
        let ocp = concat!("Ocp-Apim-Subscription-Key: ", "0123456789abcdef0123456789abcdef");
        let pwd_kv = concat!("Server=sql.local;User Id=sa;Password=", "SqlAdminPass2026!", ";");
        let cs_kv = concat!("https://auth.example.com/token?client_id=abc&client_secret=", "oauth_secret_val_998877");

        let combined = [
            pg, pg_async, redis_user, redis_nouser, mongo, az_storage, az_acs, az_sb, x_api, ocp,
            pwd_kv, cs_kv,
        ]
        .join("\n");

        let out = fast_sanitize_text_rs(&combined);

        for secret in [
            "PgSecret99!",
            "AsyncPass123",
            "RedisSecret88!",
            "AzureRedisKey999=",
            "MongoPass777!",
            "MockAzureStorageKeyBase64Value1234567890==",
            "MockAcsAccessKeyBase64Value9876543210==",
            "MockServiceBusSharedAccessKey1234567890=",
            "val_prod_api_key_9988776655",
            "0123456789abcdef0123456789abcdef",
            "SqlAdminPass2026!",
            "oauth_secret_val_998877",
        ] {
            assert!(
                !out.contains(secret),
                "Secret '{}' leaked in output:\n{}",
                secret,
                out
            );
        }

        assert!(out.contains("postgres://admin:<REDACTED_DB_PASSWORD>@db.internal:5432/valstorm"));
        assert!(out.contains("rediss://:<REDACTED_DB_PASSWORD>@cache.redis.cache.windows.net:6380/0"));
        assert!(out.contains("mongodb+srv://root:<REDACTED_DB_PASSWORD>@cluster0.mongodb.net/prod"));
        assert!(out.contains("AccountKey=<REDACTED_AZURE_KEY>"));
        assert!(out.contains("<REDACTED_AZURE_KEY>"));
        assert!(out.contains("x-api-key: <REDACTED_API_KEY>"));
        assert!(out.contains("Password=<REDACTED_SECRET>;"));
        assert!(out.contains("client_secret=<REDACTED_SECRET>"));

        // Verify idempotency
        assert_eq!(fast_sanitize_text_rs(&out), out);
    }

    #[test]
    fn redacts_provider_tokens_azure_sas_json_passwords_and_pan_contexts() {
        let openai = concat!("sk-proj-", "abcdef123456789012345678");
        let anthropic = concat!("sk-ant-", "api03-abcdef1234567890123456");
        let google = concat!("AIzaSy", "abcdef123456789012345678901234567");
        let ghp = concat!("ghp_", "1234567890abcdefghijklmnopqrstuvwxyz");
        let stripe = concat!("sk_live_", "51Abcdefghijklmnopqrstuvwxyz12345");
        let e2b = concat!("e2b_", "1234567890abcdefghijklmn");
        let sas = concat!(
            "https://valstorm.blob.core.windows.net/files/inv.pdf?sv=2022-11-02&sig=",
            "MockSasSigTokenValue1234567890abcd"
        );
        let json_creds = concat!(
            r#"{"password": ""#,
            "JsonPassWord99!",
            r#"", "client_secret": ""#,
            "JsonClientSecret88!",
            r#"", "pan": ""#,
            "4111", "1111", "1111", "1111",
            r#""}"#
        );
        let sentence = format!("Card {}.", concat!("4242-", "4242-", "4242-", "4242"));

        let combined = [
            openai, anthropic, google, ghp, stripe, e2b, sas, json_creds, &sentence,
        ]
        .join("\n");
        let out = fast_sanitize_text_rs(&combined);

        for secret in [
            "abcdef123456789012345678",
            "api03-abcdef1234567890123456",
            "abcdef123456789012345678901234567",
            "1234567890abcdefghijklmnopqrstuvwxyz",
            "51Abcdefghijklmnopqrstuvwxyz12345",
            "1234567890abcdefghijklmn",
            "MockSasSigTokenValue1234567890abcd",
            "JsonPassWord99!",
            "JsonClientSecret88!",
            "4111",
            "4242",
        ] {
            assert!(!out.contains(secret), "Leaked '{}':\n{}", secret, out);
        }

        assert!(out.contains("<REDACTED_OPENAI_KEY>"));
        assert!(out.contains("<REDACTED_ANTHROPIC_KEY>"));
        assert!(out.contains("<REDACTED_GOOGLE_API_KEY>"));
        assert!(out.contains("<REDACTED_GITHUB_PAT>"));
        assert!(out.contains("<REDACTED_STRIPE_KEY>"));
        assert!(out.contains("<REDACTED_E2B_KEY>"));
        assert!(out.contains("<REDACTED_AZURE_KEY>"));
        assert!(out.contains(r#""password": "<REDACTED_SECRET>""#));
        assert!(out.contains(r#""client_secret": "<REDACTED_SECRET>""#));
        assert!(out.contains(r#""pan": "<REDACTED_PAYMENT_CARD>""#));
        assert!(out.contains("Card <REDACTED_PAYMENT_CARD>."));
    }
}
