import { defineConfig } from "drizzle-kit";
import path from "path";

const databaseUrl = process.env.NEON_DATABASE_URL || process.env.DATABASE_URL;

if (!databaseUrl) {
  throw new Error("NEON_DATABASE_URL or DATABASE_URL must be set");
}

export default defineConfig({
  schema: path.join(__dirname, "./src/schema/index.ts"),
  dialect: "postgresql",
  dbCredentials: {
    url: databaseUrl,
  },
});
