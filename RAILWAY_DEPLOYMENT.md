# Railway deployment

This repository is configured as one Railway web service. The Express API serves
the compiled React frontend from `artifacts/personnel-monitoring/dist/public`.
Do not create a separate frontend service.

## Railway service settings

Use the repository root as the service root directory. Railway will read
`railway.json` automatically.

If entering settings manually, use:

- Build command:

  ```text
  corepack enable && corepack prepare pnpm@10.26.1 --activate && pnpm install --frozen-lockfile && pnpm --filter @workspace/personnel-monitoring run build && pnpm --filter @workspace/api-server run build
  ```

- Start command:

  ```text
  pnpm --filter @workspace/api-server run start
  ```

- Health check path: `/api/healthz`

Do not set `PORT`; Railway supplies it automatically.

## Required Railway variables

Add these under the Railway service's Variables tab:

```text
NODE_ENV=production
NEON_DATABASE_URL=<the existing Neon connection string>
SESSION_SECRET=<a long random value>
FACIAL_RECOGNITION_API_KEY=<a long random value>
```

The application prefers `NEON_DATABASE_URL` and accepts `DATABASE_URL` as a
fallback. The Neon URL must be added as a Railway variable; do not commit it to
the repository.

The existing Neon database already contains the application schema and data.
The Railway build does not run a schema push or reset command.

## Deploy

1. Push the repository, including `railway.json`, to GitHub.
2. Create a Railway project from that repository.
3. Keep the service root directory at the repository root.
4. Add the four variables above.
5. Deploy the service.
6. Generate a Railway domain under **Networking**.
7. Open `https://YOUR-RAILWAY-DOMAIN/api/healthz`; it should return:

   ```json
   {"status":"ok"}
   ```

8. Open the domain root and sign in.

## Local facial-recognition service

The Python camera service should continue running on the computer connected to
the RTSP camera. After Railway is deployed, configure its local `.env` with:

```env
API_URL=https://YOUR-RAILWAY-DOMAIN
API_KEY=<the same value as FACIAL_RECOGNITION_API_KEY>
```

Do not commit that `.env` file.