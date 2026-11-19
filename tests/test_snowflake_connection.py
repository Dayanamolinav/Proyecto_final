"""Pruebas de configuración; no abren conexiones externas."""
import os
import unittest
from unittest.mock import patch

from src.data_layer.snowflake_connection import connection_parameters


class SnowflakeConnectionConfigTests(unittest.TestCase):
    def test_requires_account_and_user(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                connection_parameters()

    def test_password_auth_uses_environment_without_logging_secret(self):
        env = {
            "SNOWFLAKE_ACCOUNT": "org-account",
            "SNOWFLAKE_USER": "user",
            "SNOWFLAKE_PASSWORD": "secret-test-value",
            "SNOWFLAKE_AUTHENTICATOR": "snowflake",
            "SNOWFLAKE_ROLE": "ACCOUNTADMIN",
        }
        with patch.dict(os.environ, env, clear=True):
            params = connection_parameters()
        self.assertEqual(params["authenticator"], "snowflake")
        self.assertEqual(params["password"], "secret-test-value")
        self.assertEqual(params["role"], "ACCOUNTADMIN")

    def test_pat_uses_token_parameter(self):
        env = {
            "SNOWFLAKE_ACCOUNT": "org-account",
            "SNOWFLAKE_USER": "user",
            "SNOWFLAKE_AUTHENTICATOR": "PROGRAMMATIC_ACCESS_TOKEN",
            "SNOWFLAKE_TOKEN": "pat-test-value",
        }
        with patch.dict(os.environ, env, clear=True):
            params = connection_parameters()
        self.assertEqual(params["authenticator"], "PROGRAMMATIC_ACCESS_TOKEN")
        self.assertEqual(params["token"], "pat-test-value")
        self.assertNotIn("password", params)

    def test_external_browser_does_not_require_password(self):
        env = {
            "SNOWFLAKE_ACCOUNT": "org-account",
            "SNOWFLAKE_USER": "user",
            "SNOWFLAKE_AUTHENTICATOR": "externalbrowser",
        }
        with patch.dict(os.environ, env, clear=True):
            params = connection_parameters()
        self.assertNotIn("password", params)


if __name__ == "__main__":
    unittest.main()
