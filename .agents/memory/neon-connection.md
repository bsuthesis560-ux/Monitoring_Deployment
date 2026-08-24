---
name: Neon connection behavior
description: Connection details needed when this project uses Neon through a pooled PostgreSQL endpoint.
---

Neon pooled endpoints may reject `search_path` as a startup option and may not provide a usable default schema search path.

**Why:** The application uses unqualified PostgreSQL table names, so the API can report that existing `public` tables do not exist even after a successful restore.

**How to apply:** Prefer the Neon connection secret at runtime and set the schema after each pool connection rather than passing `search_path` in the startup options.