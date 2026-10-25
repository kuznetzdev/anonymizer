from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException

try:
    from app.main import CategoryInput, CategoryState, add_category, categories, set_category
except ModuleNotFoundError as exc:
    if exc.name not in {"psycopg", "psycopg_pool"}:
        raise
    raise unittest.SkipTest("backend API dependencies are not installed") from exc


class LegacyCategoryTests(unittest.TestCase):
    @patch("app.main.require_auth")
    @patch("app.main.require_origin")
    def test_legacy_addr_cannot_be_added_or_restored(self, _origin: Mock, _auth: Mock) -> None:
        request = Mock()
        with self.assertRaises(HTTPException) as add_error:
            add_category(CategoryInput(label="addr"), request)
        with self.assertRaises(HTTPException) as restore_error:
            set_category("ADDR", CategoryState(active=True), request)

        self.assertEqual(add_error.exception.status_code, 409)
        self.assertEqual(restore_error.exception.status_code, 409)

    @patch("app.main._category_descriptions", {"ADDR": "Для населенных пунктов применяется LOCALITY."})
    @patch("app.main.active_categories", return_value=[{"label": "ADDR", "active": False, "built_in": True}])
    @patch("app.main.require_auth")
    def test_legacy_addr_is_exposed_as_deprecated(self, _auth: Mock, _active_categories: Mock) -> None:
        response = categories(Mock())

        self.assertTrue(response["categories"][0]["deprecated"])
        self.assertIn("LOCALITY", response["categories"][0]["description"])

    @patch("app.main._category_descriptions", {})
    @patch("app.main.active_categories", return_value=[
        {"label": "PASSPORT", "active": True, "built_in": True},
        {"label": "OMS_POLICY_ID", "active": True, "built_in": False},
        {"label": "OTHER", "active": True, "built_in": False},
    ])
    @patch("app.main.require_auth")
    def test_document_leaf_categories_expose_parent_routing(self, _auth: Mock, _active_categories: Mock) -> None:
        response = categories(Mock())
        by_label = {row["label"]: row for row in response["categories"]}

        self.assertEqual(by_label["PASSPORT"]["parent_class"], "DOCUMENT")
        self.assertEqual(by_label["OMS_POLICY_ID"]["parent_class"], "DOCUMENT")
        self.assertEqual(by_label["OMS_POLICY_ID"]["description"], "Номер полиса обязательного медицинского страхования")
        self.assertIsNone(by_label["OTHER"]["parent_class"])

    @patch("app.main.current_model", return_value=({"id": 1}, b"artifact"))
    @patch("app.main.predict", return_value=[{"start": 6, "end": 16, "label": "OMS_POLICY_ID", "confidence": 0.9}])
    @patch("app.main.active_categories", return_value=[{"label": "OMS_POLICY_ID", "active": True}])
    @patch("app.main.require_auth")
    def test_analysis_entity_preserves_leaf_and_exposes_parent(self, _auth: Mock, _active: Mock, _predict: Mock, _model: Mock) -> None:
        from app.main import TextInput, analyze

        response = analyze(TextInput(text="Полис ОМС-123456"), Mock())

        self.assertEqual(response["entities"][0]["label"], "OMS_POLICY_ID")
        self.assertEqual(response["entities"][0]["parent_class"], "DOCUMENT")
        self.assertEqual(response["masked_text"], "Полис <OMS_POLICY_ID>")


if __name__ == "__main__":
    unittest.main()
