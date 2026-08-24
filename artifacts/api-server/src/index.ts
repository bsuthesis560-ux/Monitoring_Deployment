import app from "./app";
import { logger } from "./lib/logger";
import { db, accountsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import bcrypt from "bcryptjs";
import { startBackupScheduler } from "./lib/backup";

const rawPort = process.env["PORT"];

if (!rawPort) {
  throw new Error(
    "PORT environment variable is required but was not provided.",
  );
}

const port = Number(rawPort);

if (Number.isNaN(port) || port <= 0) {
  throw new Error(`Invalid PORT value: "${rawPort}"`);
}

async function seedAdmin() {
  try {
    const [existing] = await db
      .select()
      .from(accountsTable)
      .where(eq(accountsTable.username, "admin"))
      .limit(1);

    if (!existing) {
      const passwordHash = await bcrypt.hash("password", 10);
      await db.insert(accountsTable).values({
        username: "admin",
        passwordHash,
        role: "admin",
        personnelId: null,
      });
      logger.info("Admin account created (username: admin, password: password)");
    }
  } catch (err) {
    logger.error({ err }, "Failed to seed admin account");
  }
}

app.listen(port, async (err) => {
  if (err) {
    logger.error({ err }, "Error listening on port");
    process.exit(1);
  }

  logger.info({ port }, "Server listening");
  await seedAdmin();
  startBackupScheduler(); // D6 — nightly attendance + personnel backup
});
