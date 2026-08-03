--
-- PostgreSQL database dump
--

\restrict UgkhztsJSlgLbHv4cNouphRYKFgujNPOgEvq3qK0a5saSdvLWFYYyLOjLZsfxHe

-- Dumped from database version 16.10
-- Dumped by pg_dump version 16.10

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

--
-- Name: public; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA public;


--
-- Name: SCHEMA public; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON SCHEMA public IS 'standard public schema';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: accounts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.accounts (
    id integer NOT NULL,
    username text NOT NULL,
    password_hash text NOT NULL,
    role text DEFAULT 'user'::text NOT NULL,
    personnel_id integer,
    created_at timestamp without time zone DEFAULT now() NOT NULL
);


--
-- Name: accounts_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.accounts_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: accounts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.accounts_id_seq OWNED BY public.accounts.id;


--
-- Name: attendance_logs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.attendance_logs (
    id integer NOT NULL,
    personnel_id integer,
    employee_id text NOT NULL,
    name text NOT NULL,
    department text NOT NULL,
    log_type text DEFAULT 'entry'::text NOT NULL,
    "timestamp" timestamp without time zone DEFAULT now() NOT NULL,
    "position" text DEFAULT ''::text NOT NULL
);


--
-- Name: attendance_logs_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.attendance_logs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: attendance_logs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.attendance_logs_id_seq OWNED BY public.attendance_logs.id;


--
-- Name: personnel; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.personnel (
    id integer NOT NULL,
    last_name text NOT NULL,
    first_name text NOT NULL,
    middle_initial text,
    employee_id text NOT NULL,
    department text NOT NULL,
    "position" text NOT NULL,
    photo_url text,
    created_at timestamp without time zone DEFAULT now() NOT NULL
);


--
-- Name: personnel_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.personnel_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: personnel_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.personnel_id_seq OWNED BY public.personnel.id;


--
-- Name: personnel_photos; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.personnel_photos (
    id integer NOT NULL,
    personnel_id integer NOT NULL,
    view_type text NOT NULL,
    photo_base64 text NOT NULL,
    created_at timestamp without time zone DEFAULT now() NOT NULL
);


--
-- Name: personnel_photos_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.personnel_photos_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: personnel_photos_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.personnel_photos_id_seq OWNED BY public.personnel_photos.id;


--
-- Name: sessions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sessions (
    id integer NOT NULL,
    session_token text NOT NULL,
    account_id integer NOT NULL,
    expires_at timestamp without time zone NOT NULL,
    created_at timestamp without time zone DEFAULT now() NOT NULL
);


--
-- Name: sessions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.sessions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: sessions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.sessions_id_seq OWNED BY public.sessions.id;


--
-- Name: accounts id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.accounts ALTER COLUMN id SET DEFAULT nextval('public.accounts_id_seq'::regclass);


--
-- Name: attendance_logs id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.attendance_logs ALTER COLUMN id SET DEFAULT nextval('public.attendance_logs_id_seq'::regclass);


--
-- Name: personnel id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.personnel ALTER COLUMN id SET DEFAULT nextval('public.personnel_id_seq'::regclass);


--
-- Name: personnel_photos id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.personnel_photos ALTER COLUMN id SET DEFAULT nextval('public.personnel_photos_id_seq'::regclass);


--
-- Name: sessions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sessions ALTER COLUMN id SET DEFAULT nextval('public.sessions_id_seq'::regclass);


--
-- Data for Name: accounts; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.accounts (id, username, password_hash, role, personnel_id, created_at) FROM stdin;
1	admin	$2b$10$y8OAkxzfdfGLtyJz2Y8K4etzZyk1.wN1nRqkekPzI5bW6Yjd8TBcC	admin	\N	2026-05-08 14:10:33.028388
\.


--
-- Data for Name: attendance_logs; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.attendance_logs (id, personnel_id, employee_id, name, department, log_type, "timestamp", "position") FROM stdin;
\.


--
-- Data for Name: personnel; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.personnel (id, last_name, first_name, middle_initial, employee_id, department, "position", photo_url, created_at) FROM stdin;
\.


--
-- Data for Name: personnel_photos; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.personnel_photos (id, personnel_id, view_type, photo_base64, created_at) FROM stdin;
\.


--
-- Data for Name: sessions; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.sessions (id, session_token, account_id, expires_at, created_at) FROM stdin;
1	f9622d2f4d971819a294723415e74edd4b0296001f6c04630b55c76bc1a7d8d6	1	2026-06-23 07:22:28.068	2026-06-16 07:22:28.06985
2	653e2ca2132705ecd140e07da60c0bc119d98944630c6fef4bf48aaa11f6baeb	1	2026-06-23 07:28:37.047	2026-06-16 07:28:37.073602
\.


--
-- Name: accounts_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.accounts_id_seq', 1, true);


--
-- Name: attendance_logs_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.attendance_logs_id_seq', 1, false);


--
-- Name: personnel_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.personnel_id_seq', 1, false);


--
-- Name: personnel_photos_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.personnel_photos_id_seq', 1, false);


--
-- Name: sessions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.sessions_id_seq', 2, true);


--
-- Name: accounts accounts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.accounts
    ADD CONSTRAINT accounts_pkey PRIMARY KEY (id);


--
-- Name: accounts accounts_username_unique; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.accounts
    ADD CONSTRAINT accounts_username_unique UNIQUE (username);


--
-- Name: attendance_logs attendance_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.attendance_logs
    ADD CONSTRAINT attendance_logs_pkey PRIMARY KEY (id);


--
-- Name: personnel personnel_employee_id_unique; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.personnel
    ADD CONSTRAINT personnel_employee_id_unique UNIQUE (employee_id);


--
-- Name: personnel_photos personnel_photos_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.personnel_photos
    ADD CONSTRAINT personnel_photos_pkey PRIMARY KEY (id);


--
-- Name: personnel personnel_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.personnel
    ADD CONSTRAINT personnel_pkey PRIMARY KEY (id);


--
-- Name: sessions sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_pkey PRIMARY KEY (id);


--
-- Name: sessions sessions_session_token_unique; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_session_token_unique UNIQUE (session_token);


--
-- Name: accounts accounts_personnel_id_personnel_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.accounts
    ADD CONSTRAINT accounts_personnel_id_personnel_id_fk FOREIGN KEY (personnel_id) REFERENCES public.personnel(id) ON DELETE CASCADE;


--
-- Name: attendance_logs attendance_logs_personnel_id_personnel_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.attendance_logs
    ADD CONSTRAINT attendance_logs_personnel_id_personnel_id_fk FOREIGN KEY (personnel_id) REFERENCES public.personnel(id) ON DELETE SET NULL;


--
-- Name: personnel_photos personnel_photos_personnel_id_personnel_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.personnel_photos
    ADD CONSTRAINT personnel_photos_personnel_id_personnel_id_fk FOREIGN KEY (personnel_id) REFERENCES public.personnel(id) ON DELETE CASCADE;


--
-- Name: sessions sessions_account_id_accounts_id_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_account_id_accounts_id_fk FOREIGN KEY (account_id) REFERENCES public.accounts(id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

\unrestrict UgkhztsJSlgLbHv4cNouphRYKFgujNPOgEvq3qK0a5saSdvLWFYYyLOjLZsfxHe

