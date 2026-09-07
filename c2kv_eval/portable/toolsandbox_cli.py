"""Run the official ToolSandbox CLI against an OpenAI-compatible local server.

ToolSandbox intentionally hard-codes ``https://api.openai.com/v1`` in its OpenAI
role constructors and ignores ``OPENAI_BASE_URL``.  This wrapper keeps the
official checkout unmodified while redirecting only the constructed client.
The constructor patch is applied at module import so multiprocessing ``spawn``
workers inherit it when they re-import this main module.
"""
from __future__ import annotations

import os
import sys

from openai import OpenAI
from tool_sandbox.roles.openai_api_agent import OpenAIAPIAgent
from tool_sandbox.roles.openai_api_user import OpenAIAPIUser


def _redirect_client(role_class):
    original_init = role_class.__init__

    def redirected_init(self):
        original_init(self)
        self.openai_client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            base_url=os.environ["OPENAI_BASE_URL"],
            timeout=600.0,
        )

    role_class.__init__ = redirected_init


_redirect_client(OpenAIAPIAgent)
_redirect_client(OpenAIAPIUser)


def main() -> None:
    sys.argv[0] = "tool_sandbox"
    from tool_sandbox.cli import main as tool_sandbox_main

    tool_sandbox_main()


if __name__ == "__main__":
    main()
