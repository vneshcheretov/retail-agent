"""Tests for the PII masking node.

The contract: no email, phone, or PII-named column value may survive masking,
no matter how the SQL produced it.
"""

from __future__ import annotations

from src.nodes.mask_pii import EMAIL_RE, PHONE_RE, mask_row, mask_rows


def test_masks_pii_named_columns():
    row = {"user_id": 7, "email": "vadim@example.com", "first_name": "Vadim", "spend": 120.5}
    masked, redactions = mask_row(row)

    assert masked["email"] == "[redacted]"
    assert masked["first_name"] == "[redacted]"
    assert masked["user_id"] == 7  # non-PII kept as-is
    assert masked["spend"] == 120.5
    assert redactions == 2


def test_masks_email_in_free_text():
    row = {"note": "contact bob at bob.smith@shop.io please"}
    masked, redactions = mask_row(row)

    assert "bob.smith@shop.io" not in masked["note"]
    assert redactions == 1
    assert not EMAIL_RE.search(masked["note"])


def test_masks_phone_in_free_text():
    row = {"note": "call +1 415-555-2671 today"}
    masked, _ = mask_row(row)

    assert not PHONE_RE.search(masked["note"])
    assert "555" not in masked["note"]


def test_no_pii_leaves_rows_unchanged():
    rows = [{"category": "Jeans", "revenue": 1000}, {"category": "Tops", "revenue": 800}]
    masked, redactions = mask_rows(rows)

    assert masked == rows
    assert redactions == 0


def test_handles_none_values():
    row = {"email": None, "user_id": 3}
    masked, redactions = mask_row(row)

    assert masked["email"] is None  # nothing to redact
    assert redactions == 0


def test_masks_pii_name_variants():
    row = {
        "e-mail": "a@b.com",
        "e_mail": "c@d.com",
        "customer_email": "e@f.com",
        "contact_phone": "12345678",
        "street_address": "1 Main St",
        "full_name": "Jane Roe",
        "product_name": "Blue Jeans",   # not PII despite "name"
        "category": "Jeans",
    }
    masked, _ = mask_row(row)

    for col in ("e-mail", "e_mail", "customer_email", "contact_phone", "street_address", "full_name"):
        assert masked[col] == "[redacted]", col
    assert masked["product_name"] == "Blue Jeans"  # kept
    assert masked["category"] == "Jeans"


def test_numeric_aggregate_with_pii_word_in_name_is_kept():
    # An aggregate aliased with a PII word but holding a number is not PII.
    row = {"email_revenue": 1234.5, "phone_count": 7, "first_name_count": 3}
    masked, redactions = mask_row(row)

    assert masked == row
    assert redactions == 0


def test_masks_llm_flagged_alias():
    # SQL did 'SELECT first_name AS x'; the model flagged 'x' as PII.
    row = {"x": "Vadim", "amount": 5}
    masked, redactions = mask_row(row, extra_pii=["x"])

    assert masked["x"] == "[redacted]"
    assert masked["amount"] == 5
    assert redactions == 1
