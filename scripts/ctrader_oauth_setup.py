"""One-time setup: obtain cTrader OAuth tokens, resolve the account_id for a
given trader login, and seed the local token store so the main application
can start without further interaction.

Prerequisites:
  1. Register an application at https://openapi.ctrader.com/ to get a
     client ID and client secret.
  2. Install the OAuth helper and run it to get your first access/refresh
     token pair:
         uvx ctrader-oauth-fetcher --client-id YOUR_ID --client-secret YOUR_SECRET
     This opens a browser, asks you to log in to cTrader and authorize the
     app, then prints an access_token, refresh_token, and token lifetime in seconds. Have
     those three values ready before running this script.

Run from the project root:
    python scripts/ctrader_oauth_setup.py
"""

from __future__ import annotations

import asyncio
import time

from ctrader_api_client import AccountCredentials, ClientConfig, CTraderClient

from config.ctrader_token_store import SQLiteTokenStore
from config.settings import settings


async def main() -> None:
    print("cTrader Open API -- one-time account setup\n")

    client_id = settings.CTRADER_CLIENT_ID or input("Client ID: ").strip()
    client_secret = settings.CTRADER_CLIENT_SECRET or input("Client secret: ").strip()
    trader_login = int(
        input("Trader login (numeric, from cTrader account number): ").strip()
    )
    access_token = input("Access token (from ctrader-oauth-fetcher): ").strip()
    refresh_token = input("Refresh token (from ctrader-oauth-fetcher): ").strip()
    expires_in = int(
        input("Token lifetime in seconds (from ctrader-oauth-fetcher): ").strip()
    )
    if expires_in <= 0:
        raise ValueError("Token lifetime must be a positive number of seconds.")
    expires_at = int(time.time()) + expires_in
    environment = (
        settings.CTRADER_ENVIRONMENT
        or input("Environment [demo/live] (default demo): ").strip()
        or "demo"
    )

    host = (
        "live.ctraderapi.com"
        if environment.lower() == "live"
        else "demo.ctraderapi.com"
    )
    config = ClientConfig(
        client_id=client_id,
        client_secret=client_secret,
        host=host,
        port=5035,
        use_ssl=True,
    )

    token_store = SQLiteTokenStore(settings.DB_PATH)
    client = CTraderClient(config, token_store=token_store)

    async with client:
        account_id = await client.accounts.resolve_account_id(
            access_token, trader_login=trader_login
        )
        print(f"\nResolved account_id = {account_id} for trader_login = {trader_login}")

        creds = AccountCredentials(
            account_id=account_id,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
        )
        await client.auth.authenticate_trader(creds)
        authenticated_creds = client.auth.get_credentials(account_id)
        if authenticated_creds is None:
            raise RuntimeError(
                f"cTrader authentication completed without credentials for account {account_id}"
            )
        await token_store.save(authenticated_creds)

        account = await client.accounts.get_trader(account_id)
        print(
            f"Authenticated OK. Broker: {account.broker_name}  Balance: {account.balance}"
        )

    print("\nTokens saved to the local token store (same SQLite file as DB_PATH).")
    print("Add the following to your .env so the main app can start:\n")
    print(f"CTRADER_CLIENT_ID={client_id}")
    print(f"CTRADER_CLIENT_SECRET={client_secret}")
    print(f"CTRADER_ACCOUNT_ID={account_id}")
    print(f"CTRADER_ENVIRONMENT={environment}")
    print(
        "\nCTRADER_ACCESS_TOKEN / CTRADER_REFRESH_TOKEN are not needed in .env going "
        "forward -- they are now persisted in the token store and the client rotates "
        "them automatically on every refresh. Keep today's values somewhere safe only "
        "as a one-time fallback in case the token store file is ever deleted."
    )


if __name__ == "__main__":
    asyncio.run(main())
