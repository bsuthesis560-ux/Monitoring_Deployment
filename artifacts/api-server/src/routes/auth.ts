import { Router, type IRouter } from "express";
import { z } from "zod";
import { db } from "@workspace/db";
import { accountsTable, personnelTable, sessionsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import bcrypt from "bcryptjs";
import crypto from "crypto";
import { validate } from "../middlewares/validate";
import { securityLog } from "../lib/security-log";

const router: IRouter = Router();

function generateToken(): string {
  return crypto.randomBytes(32).toString("hex");
}

const loginSchema = z.object({
  username: z.string().min(1).max(100).trim(),
  password: z.string().min(1).max(128),
});

// D2 — lockout constants
const MAX_ATTEMPTS    = 5;
const LOCKOUT_MINUTES = 30;

router.post("/login", validate(loginSchema), async (req, res) => {
  const { username, password } = req.body;
  const ip = req.ip ?? "unknown";

  const [account] = await db
    .select()
    .from(accountsTable)
    .where(eq(accountsTable.username, username))
    .limit(1);

  if (!account) {
    // D3 — log failed attempt even when account doesn't exist
    securityLog("LOGIN_FAILED", { username, ip });
    res.status(401).json({ error: "Invalid username or password" });
    return;
  }

  // D2 — check account lockout
  if (account.lockedUntil && account.lockedUntil > new Date()) {
    const minutesLeft = Math.ceil((account.lockedUntil.getTime() - Date.now()) / 60_000);
    securityLog("LOGIN_LOCKED", { username, ip });
    res.status(423).json({
      error: `Account is locked. Try again in ${minutesLeft} minute${minutesLeft === 1 ? "" : "s"}.`,
    });
    return;
  }

  const valid = await bcrypt.compare(password, account.passwordHash);

  if (!valid) {
    // D2 — increment failed attempts and lock if threshold reached
    const newAttempts = account.failedLoginAttempts + 1;
    const shouldLock  = newAttempts >= MAX_ATTEMPTS;
    await db
      .update(accountsTable)
      .set({
        failedLoginAttempts: newAttempts,
        lockedUntil: shouldLock
          ? new Date(Date.now() + LOCKOUT_MINUTES * 60_000)
          : account.lockedUntil,
      })
      .where(eq(accountsTable.id, account.id));

    // D3 — log failure
    securityLog("LOGIN_FAILED", {
      username,
      ip,
      detail: shouldLock ? `Account locked after ${newAttempts} failed attempts` : undefined,
    });
    res.status(401).json({ error: "Invalid username or password" });
    return;
  }

  // D2 — reset counter on successful login
  await db
    .update(accountsTable)
    .set({ failedLoginAttempts: 0, lockedUntil: null })
    .where(eq(accountsTable.id, account.id));

  const token     = generateToken();
  const expiresAt = new Date(Date.now() + 1000 * 60 * 60 * 24 * 7);

  await db.insert(sessionsTable).values({
    sessionToken: token,
    accountId: account.id,
    expiresAt,
  });

  res.cookie("session_token", token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    expires: expiresAt,
    path: "/",
  });

  // D3 — log success
  securityLog("LOGIN_SUCCESS", { username: account.username, ip });

  let name: string | null = null;
  if (account.personnelId) {
    const [personnel] = await db
      .select()
      .from(personnelTable)
      .where(eq(personnelTable.id, account.personnelId))
      .limit(1);
    if (personnel) {
      name = `${personnel.firstName} ${personnel.lastName}`;
    }
  }

  res.json({
    id: account.id,
    username: account.username,
    role: account.role,
    personnelId: account.personnelId ?? null,
    name,
  });
});

router.post("/logout", async (req, res) => {
  const token = req.cookies?.session_token;
  if (token) {
    await db.delete(sessionsTable).where(eq(sessionsTable.sessionToken, token));
    res.clearCookie("session_token", { path: "/" });
    // D3 — log logout
    securityLog("LOGOUT", { ip: req.ip ?? "unknown" });
  }
  res.json({ message: "Logged out" });
});

router.get("/me", async (req, res) => {
  const token = req.cookies?.session_token;
  if (!token) {
    res.status(401).json({ error: "Not authenticated" });
    return;
  }

  const [session] = await db
    .select()
    .from(sessionsTable)
    .where(eq(sessionsTable.sessionToken, token))
    .limit(1);

  if (!session || session.expiresAt < new Date()) {
    if (session) {
      await db.delete(sessionsTable).where(eq(sessionsTable.sessionToken, token));
    }
    res.clearCookie("session_token", { path: "/" });
    res.status(401).json({ error: "Session expired" });
    return;
  }

  const [account] = await db
    .select()
    .from(accountsTable)
    .where(eq(accountsTable.id, session.accountId))
    .limit(1);

  if (!account) {
    res.status(401).json({ error: "Account not found" });
    return;
  }

  let name: string | null = null;
  if (account.personnelId) {
    const [personnel] = await db
      .select()
      .from(personnelTable)
      .where(eq(personnelTable.id, account.personnelId))
      .limit(1);
    if (personnel) {
      name = `${personnel.firstName} ${personnel.lastName}`;
    }
  }

  res.json({
    id: account.id,
    username: account.username,
    role: account.role,
    personnelId: account.personnelId ?? null,
    name,
  });
});

export default router;
