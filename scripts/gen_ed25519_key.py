#!/usr/bin/env python3
"""Generate an Ed25519 keypair for audit-log signing.

Produces a hex-encoded private key and public key suitable for use with
the --ed25519-private-key / --ed25519-public-key CLI flags (or the
RFSN_ED25519_PRIVATE_KEY / RFSN_ED25519_PUBLIC_KEY env vars).

Usage:
    python3 scripts/gen_ed25519_key.py
    python3 scripts/gen_ed25519_key.py --shell-export  # print export commands

The private key MUST be kept secret. Anyone with the private key can
sign audit-log entries as this node. The public key can be shared
freely — it can only verify, not forge.

Requires the optional 'cryptography' package:
    pip install cryptography
"""

import argparse
import os
import sys

# Add the project root to sys.path so `signers` is importable when run
# from anywhere (e.g. python3 scripts/gen_ed25519_key.py).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate an Ed25519 keypair for audit-log signing."
    )
    parser.add_argument(
        "--shell-export", action="store_true",
        help="Print export commands for shell sourcing (RFSN_ED25519_* env vars).",
    )
    args = parser.parse_args()

    try:
        from signers import Ed25519Signer
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Install with: pip install cryptography", file=sys.stderr)
        return 1

    signer = Ed25519Signer.generate()
    priv = signer.private_key_hex
    pub = signer.public_key_hex

    if args.shell_export:
        print(f'export RFSN_ED25519_PRIVATE_KEY="{priv}"')
        print(f'export RFSN_ED25519_PUBLIC_KEY="{pub}"')
    else:
        print("Ed25519 keypair generated for audit-log signing.")
        print()
        print(f"Private key (SECRET — keep this safe):")
        print(f"  {priv}")
        print()
        print(f"Public key (share freely — verify-only):")
        print(f"  {pub}")
        print()
        print("Usage:")
        print(f"  python rfsn_cli.py --ed25519-private-key {priv}")
        print(f"  # Or via env:")
        print(f'  export RFSN_ED25519_PRIVATE_KEY="{priv}"')
        print(f"  python rfsn_cli.py")
        print()
        print("Verify-only node (reads but doesn't write the log):")
        print(f"  python rfsn_cli.py --ed25519-public-key {pub}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
