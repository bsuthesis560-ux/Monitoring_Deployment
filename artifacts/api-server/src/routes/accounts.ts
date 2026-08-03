import { Router, type IRouter } from "express";
import { db } from "@workspace/db";
import { accountsTable, personnelTable, sessionsTable, auditLogsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import { securityLog } from "../lib/security-log";
import bcrypt from "bcryptjs";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Account backup goes to <workspace-root>/account_backup/users_backup.csv
// __dirname is artifacts/api-server/dist/ at runtime, so go up 3 levels
const BACKUP_DIR = path.resolve(__dirname, "../../../account_backup");
const BACKUP_FILE = path.join(BACKUP_DIR, "users_backup.csv");

async function writeAccountBackup(): Promise<void> {
  try {
    const rows = await db
      .select({
        id: accountsTable.id,
        username: accountsTable.username,
        role: accountsTable.role,
        personnelId: accountsTable.personnelId,
        createdAt: accountsTable.createdAt,
        firstName: personnelTable.firstName,
        lastName: personnelTable.lastName,
        middleInitial: personnelTable.middleInitial,
        department: personnelTable.department,
        position: personnelTable.position,
      })
      .from(accountsTable)
      .leftJoin(personnelTable, eq(personnelTable.id, accountsTable.personnelId))
      .orderBy(accountsTable.id);

    const header = [
      "ID",
      "Username (Employee ID)",
      "Full Name",
      "Department",
      "Position",
      "Role",
      "Personnel ID",
      "Created At",
      "NOTE",
    ].join(",");

    const lines = rows.map((r) => {
      const fullName =
        r.firstName && r.lastName
          ? `"${r.lastName}, ${r.firstName}${r.middleInitial ? ` ${r.middleInitial}.` : ""}"`
          : `"—"`;
      return [
        r.id,
        r.username,
        fullName,
        `"${r.department ?? ""}"`,
        `"${r.position ?? ""}"`,
        r.role,
        r.personnelId ?? "",
        new Date(r.createdAt).toISOString(),
        '"Passwords are hashed and cannot be recovered. Reset via admin panel."',
      ].join(",");
    });

    const csvContent = [
      "# BatStateU Personnel Monitoring System — Account Backup",
      `# Generated: ${new Date().toISOString()}`,
      `# Total accounts: ${rows.length}`,
      "#",
      header,
      ...lines,
    ].join("\n");

    if (!fs.existsSync(BACKUP_DIR)) {
      fs.mkdirSync(BACKUP_DIR, { recursive: true });
    }
    fs.writeFileSync(BACKUP_FILE, csvContent, "utf-8");
  } catch (err) {
    console.error("[AccountBackup] Failed to write backup:", err);
  }
}

const router: IRouter = Router();

/** Returns the admin's account ID on success, or null (after sending a response) on failure. */
async function requireAdmin(req: any, res: any): Promise<number | null> {
  const token = req.cookies?.session_token;
  if (!token) {
    res.status(401).json({ error: "Not authenticated" });
    return null;
  }
  const [session] = await db
    .select()
    .from(sessionsTable)
    .where(eq(sessionsTable.sessionToken, token))
    .limit(1);
  if (!session || session.expiresAt < new Date()) {
    res.status(401).json({ error: "Session expired" });
    return null;
  }
  const [account] = await db
    .select()
    .from(accountsTable)
    .where(eq(accountsTable.id, session.accountId))
    .limit(1);
  if (!account || account.role !== "admin") {
    res.status(403).json({ error: "Admin access required" });
    return null;
  }
  return account.id;
}

router.get("/", async (req, res) => {
  const adminId = await requireAdmin(req, res);
  if (adminId === null) return;

  const rows = await db
    .select({
      id: accountsTable.id,
      username: accountsTable.username,
      role: accountsTable.role,
      personnelId: accountsTable.personnelId,
      createdAt: accountsTable.createdAt,
      firstName: personnelTable.firstName,
      lastName: personnelTable.lastName,
    })
    .from(accountsTable)
    .leftJoin(personnelTable, eq(personnelTable.id, accountsTable.personnelId));

  const result = rows.map((r) => ({
    id: r.id,
    username: r.username,
    role: r.role as "admin" | "user",
    personnelId: r.personnelId ?? null,
    personnelName:
      r.firstName && r.lastName ? `${r.firstName} ${r.lastName}` : null,
    createdAt: r.createdAt.toISOString(),
  }));

  res.json(result);
});

router.post("/", async (req, res) => {
  const adminId = await requireAdmin(req, res);
  if (adminId === null) return;

  const { personnelId, password } = req.body;

  if (!personnelId || !password) {
    res.status(400).json({ error: "personnelId and password are required" });
    return;
  }

  const [personnel] = await db
    .select()
    .from(personnelTable)
    .where(eq(personnelTable.id, personnelId))
    .limit(1);

  if (!personnel) {
    res.status(404).json({ error: "Personnel not found" });
    return;
  }

  const existingByPersonnel = await db
    .select()
    .from(accountsTable)
    .where(eq(accountsTable.personnelId, personnelId))
    .limit(1);

  if (existingByPersonnel.length > 0) {
    res.status(400).json({ error: "This personnel already has an account" });
    return;
  }

  const existingByUsername = await db
    .select()
    .from(accountsTable)
    .where(eq(accountsTable.username, personnel.employeeId))
    .limit(1);

  if (existingByUsername.length > 0) {
    res.status(400).json({ error: "Username already taken" });
    return;
  }

  const passwordHash = await bcrypt.hash(password, 10);
  const [account] = await db
    .insert(accountsTable)
    .values({
      username: personnel.employeeId,
      passwordHash,
      role: "user",
      personnelId: personnel.id,
    })
    .returning();

  void writeAccountBackup();

  res.status(201).json({
    id: account.id,
    username: account.username,
    role: account.role as "admin" | "user",
    personnelId: account.personnelId ?? null,
    personnelName: `${personnel.firstName} ${personnel.lastName}`,
    createdAt: account.createdAt.toISOString(),
  });
  // D3 + D4
  securityLog("ACCOUNT_CREATED", { targetId: account.id, performedBy: adminId, ip: req.ip });
  void db.insert(auditLogsTable).values({
    performedBy: adminId,
    action: "CREATE_ACCOUNT",
    targetType: "account",
    targetId: account.id,
    detail: JSON.stringify({ username: account.username, role: account.role }),
    ip: req.ip ?? null,
  });
});

router.put("/:id", async (req, res) => {
  const adminId = await requireAdmin(req, res);
  if (adminId === null) return;

  const id = parseInt(req.params.id);
  if (isNaN(id)) {
    res.status(400).json({ error: "Invalid ID" });
    return;
  }

  const { password } = req.body;
  if (!password) {
    res.status(400).json({ error: "Password is required" });
    return;
  }

  const [existing] = await db
    .select()
    .from(accountsTable)
    .where(eq(accountsTable.id, id))
    .limit(1);

  if (!existing) {
    res.status(404).json({ error: "Account not found" });
    return;
  }

  const passwordHash = await bcrypt.hash(password, 10);
  const [updated] = await db
    .update(accountsTable)
    .set({ passwordHash })
    .where(eq(accountsTable.id, id))
    .returning();

  let personnelName: string | null = null;
  if (updated.personnelId) {
    const [p] = await db
      .select()
      .from(personnelTable)
      .where(eq(personnelTable.id, updated.personnelId))
      .limit(1);
    if (p) personnelName = `${p.firstName} ${p.lastName}`;
  }

  void writeAccountBackup();

  res.json({
    id: updated.id,
    username: updated.username,
    role: updated.role as "admin" | "user",
    personnelId: updated.personnelId ?? null,
    personnelName,
    createdAt: updated.createdAt.toISOString(),
  });
  // D3 + D4
  securityLog("ACCOUNT_UPDATED", { targetId: id, performedBy: adminId, ip: req.ip });
  void db.insert(auditLogsTable).values({
    performedBy: adminId,
    action: "UPDATE_ACCOUNT",
    targetType: "account",
    targetId: id,
    detail: JSON.stringify({ username: updated.username }),
    ip: req.ip ?? null,
  });
});

router.delete("/:id", async (req, res) => {
  const adminId = await requireAdmin(req, res);
  if (adminId === null) return;

  const id = parseInt(req.params.id);
  if (isNaN(id)) {
    res.status(400).json({ error: "Invalid ID" });
    return;
  }

  const [existing] = await db
    .select()
    .from(accountsTable)
    .where(eq(accountsTable.id, id))
    .limit(1);

  if (!existing) {
    res.status(404).json({ error: "Account not found" });
    return;
  }

  await db.delete(accountsTable).where(eq(accountsTable.id, id));

  void writeAccountBackup();

  res.json({ message: "Account deleted" });
  // D3 + D4
  securityLog("ACCOUNT_DELETED", { targetId: id, performedBy: adminId, ip: req.ip });
  void db.insert(auditLogsTable).values({
    performedBy: adminId,
    action: "DELETE_ACCOUNT",
    targetType: "account",
    targetId: id,
    detail: JSON.stringify({ username: existing.username }),
    ip: req.ip ?? null,
  });
});

export default router;
