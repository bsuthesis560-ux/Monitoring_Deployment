# Workspace

## Overview

Personnel Monitoring System for Batangas State University - Lipa Campus (CET Department). Full-stack web application with admin and user roles, personnel management, and account management.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Frontend**: React + Vite + Tailwind CSS + shadcn/ui
- **Auth**: Session-based with cookies (bcryptjs for password hashing)
- **Build**: esbuild (CJS bundle)

## Structure

```text
artifacts-monorepo/
├── artifacts/
│   ├── api-server/         # Express API server
│   └── personnel-monitoring/ # React + Vite frontend
├── lib/                    # Shared libraries
│   ├── api-spec/           # OpenAPI spec + Orval codegen config
│   ├── api-client-react/   # Generated React Query hooks
│   ├── api-zod/            # Generated Zod schemas from OpenAPI
│   └── db/                 # Drizzle ORM schema + DB connection
├── scripts/                # Utility scripts
├── pnpm-workspace.yaml
├── tsconfig.base.json
├── tsconfig.json
└── package.json
```

## Database Schema

- **personnel** — Personnel records (name, employee_id, department, position, photo, vehicle plate)
- **accounts** — User accounts (username=employee_id, hashed password, role: admin|user, links to personnel)
- **sessions** — Session tokens for cookie-based auth
- **attendance_logs** — Real-time facial recognition logs (employee_id, name, department, log_type, timestamp)

## Default Admin Account

- Username: `admin`
- Password: `password`

## Replit Setup

- **Frontend** runs on port 5000 (Vite dev server, `webview` workflow)
- **Backend** runs on port 8080 (Express, `console` workflow)
- Vite proxies `/api/*` requests to `http://localhost:8080`
- Database: Replit PostgreSQL (provisioned, `DATABASE_URL` auto-set)
- Schema pushed via `pnpm --filter @workspace/db run push`
- `SESSION_SECRET` stored in Replit secrets

## Environment Variables

- `SESSION_SECRET` — Secret for signed session cookies (stored in Replit secrets)
- `DATABASE_URL` — Replit PostgreSQL connection string (auto-provisioned)
- `FACIAL_RECOGNITION_API_KEY` — API key used by the local Python facial recognition service to authenticate log submissions

## System Flow

- Login → admin goes to `/dashboard`, user goes to `/staff-monitoring`
- Dashboard has: department boxes, sidebar with Staff Monitoring + Register Personnel + Manage Accounts
- Register Personnel (`/register`) — creates personnel + optionally creates account at the same time
- Manage Accounts (`/accounts`) — add account for existing personnel, update password, delete account
- Staff Monitoring (`/staff-monitoring`) — live attendance log table (polls every 5s) + personnel roster tab; department-filtered for user role

## Facial Recognition Service (Local)

- Located at `services/facial-recognition/facial_recognition_service.py`
- Runs **locally** on the machine with camera access (not on Replit cloud)
- On startup, downloads registered personnel photos from the API
- Streams RTSP from Hikvision camera (192.168.1.64), runs DeepFace recognition
- POSTs recognized employee IDs to `/api/logs` with the `FACIAL_RECOGNITION_API_KEY`
- 60-second cooldown per person enforced by API

## API Endpoints

- `POST /api/auth/login` — Login
- `POST /api/auth/logout` — Logout
- `GET /api/auth/me` — Get current session
- `GET /api/personnel` — List all personnel
- `POST /api/personnel` — Create personnel (+ optional account)
- `GET/PUT/DELETE /api/personnel/:id`
- `GET /api/accounts` — List all accounts
- `POST /api/accounts` — Create account for existing personnel
- `PUT/DELETE /api/accounts/:id`
- `GET /api/logs` — Get attendance logs (dept-filtered for user role; requires session cookie)
- `POST /api/logs` — Submit a recognition log (requires `x-api-key` header)
- `GET /api/logs/personnel-photos` — Download all registered personnel photos as base64 (requires `x-api-key`)
