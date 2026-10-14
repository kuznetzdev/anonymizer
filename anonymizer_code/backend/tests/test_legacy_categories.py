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


if __name__ == "__main__":
    unittest.main()
