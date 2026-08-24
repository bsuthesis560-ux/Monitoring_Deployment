import { drizzle } from "drizzle-orm/node-postgres";
import pg from "pg";
import * as schema from "./schema";

const { Pool } = pg;

const databaseUrl = process.env.NEON_DATABASE_URL || process.env.DATABASE_URL;

if (!databaseUrl) {
  throw new Error(
    "NEON_DATABASE_URL or DATABASE_URL must be set. Did you forget to configure a database?",
  );
}

const sslRequired = databaseUrl.includes("sslmode=require") ||
                    databaseUrl.includes("neon.tech");

export const pool = new Pool({
  connectionString: databaseUrl,
  ...(sslRequired && { ssl: { rejectUnauthorized: false } }),
});
// Neon pooled connections may reject search_path as a startup parameter.
// Set it after each connection is established instead.
pool.on("connect", (client) => {
  void client.query("SET search_path TO public");
});
export const db = drizzle(pool, { schema });

export * from "./schema";
