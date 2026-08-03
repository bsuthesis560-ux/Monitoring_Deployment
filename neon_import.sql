-- ============================================================
-- BSU Personnel Monitoring System — Neon Import Script
-- Clean version: no Replit-specific commands, full schema
-- ============================================================

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

-- ============================================================
-- TABLES
-- ============================================================

CREATE TABLE IF NOT EXISTS public.personnel (
    id          SERIAL PRIMARY KEY,
    last_name   TEXT NOT NULL,
    first_name  TEXT NOT NULL,
    middle_initial TEXT,
    employee_id TEXT NOT NULL UNIQUE,
    department  TEXT NOT NULL,
    position    TEXT NOT NULL,
    photo_url   TEXT,
    created_at  TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL
);

CREATE TABLE IF NOT EXISTS public.personnel_photos (
    id           SERIAL PRIMARY KEY,
    personnel_id INTEGER NOT NULL REFERENCES public.personnel(id) ON DELETE CASCADE,
    view_type    TEXT NOT NULL,
    photo_base64 TEXT NOT NULL,
    created_at   TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL
);

CREATE TABLE IF NOT EXISTS public.accounts (
    id                     SERIAL PRIMARY KEY,
    username               TEXT NOT NULL UNIQUE,
    password_hash          TEXT NOT NULL,
    role                   TEXT NOT NULL DEFAULT 'user',
    personnel_id           INTEGER REFERENCES public.personnel(id) ON DELETE CASCADE,
    failed_login_attempts  INTEGER NOT NULL DEFAULT 0,
    locked_until           TIMESTAMP WITHOUT TIME ZONE,
    created_at             TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL
);

CREATE TABLE IF NOT EXISTS public.sessions (
    id            SERIAL PRIMARY KEY,
    session_token TEXT NOT NULL UNIQUE,
    account_id    INTEGER NOT NULL REFERENCES public.accounts(id) ON DELETE CASCADE,
    expires_at    TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    created_at    TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL
);

CREATE TABLE IF NOT EXISTS public.attendance_logs (
    id           SERIAL PRIMARY KEY,
    personnel_id INTEGER REFERENCES public.personnel(id) ON DELETE SET NULL,
    employee_id  TEXT NOT NULL,
    name         TEXT NOT NULL,
    department   TEXT NOT NULL,
    position     TEXT NOT NULL DEFAULT '',
    log_type     TEXT NOT NULL DEFAULT 'entry',
    timestamp    TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL
);

CREATE TABLE IF NOT EXISTS public.audit_logs (
    id           SERIAL PRIMARY KEY,
    performed_by INTEGER,
    action       TEXT NOT NULL,
    target_type  TEXT NOT NULL,
    target_id    INTEGER,
    detail       TEXT,
    ip           TEXT,
    timestamp    TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL
);

-- ============================================================
-- SEED DATA — default admin account
-- username: admin
-- password: admin  (change this immediately after first login)
-- ============================================================

INSERT INTO public.accounts (id, username, password_hash, role, personnel_id, failed_login_attempts, created_at)
VALUES (1, 'admin', '$2b$10$y8OAkxzfdfGLtyJz2Y8K4etzZyk1.wN1nRqkekPzI5bW6Yjd8TBcC', 'admin', NULL, 0, '2026-05-08 14:10:33.028388')
ON CONFLICT (username) DO NOTHING;

-- Keep sequences in sync
SELECT pg_catalog.setval('public.accounts_id_seq', 1, true);
SELECT pg_catalog.setval('public.attendance_logs_id_seq', 1, false);
SELECT pg_catalog.setval('public.personnel_id_seq', 1, false);
SELECT pg_catalog.setval('public.personnel_photos_id_seq', 1, false);
SELECT pg_catalog.setval('public.sessions_id_seq', 1, false);

-- ============================================================
-- DONE
-- ============================================================
