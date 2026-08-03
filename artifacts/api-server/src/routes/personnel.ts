import { Router, type IRouter } from "express";
import { z } from "zod";
import { db } from "@workspace/db";
import { personnelTable, accountsTable, sessionsTable, personnelPhotosTable, auditLogsTable } from "@workspace/db";
import { eq, and, sql } from "drizzle-orm";
import bcrypt from "bcryptjs";
import { validate } from "../middlewares/validate";
import { validatePhoto } from "../lib/validate-photo";
import { securityLog } from "../lib/security-log";
import { writePersonnelBackup } from "../lib/backup";

// ── Zod schemas (P3) ─────────────────────────────────────────────────────────

// Only letters (incl. accented), spaces, apostrophes, hyphens, periods
const NAME_REGEX = /^[a-zA-ZÀ-ÿ\s.''\-]+$/;
const NAME_MSG   = "Only letters, spaces, hyphens, and apostrophes are allowed.";

function nameField() {
  return z.string().trim().min(1).max(100).regex(NAME_REGEX, NAME_MSG);
}

const photosSchema = z.object({
  front: z.string().optional(),
  left:  z.string().optional(),
  right: z.string().optional(),
  top:   z.string().optional(),
}).optional();

const createPersonnelSchema = z.object({
  lastName:      nameField(),
  firstName:     nameField(),
  middleInitial: z.string().trim().max(100).regex(NAME_REGEX, NAME_MSG).optional().or(z.literal("")),
  employeeId:    z.string().length(10, "Employee ID must be exactly 10 characters").regex(/^[A-Z0-9\-]+$/i, "Employee ID must contain only letters, numbers, or hyphens"),
  department:    z.string().min(1).max(100).trim(),
  position:      z.string().min(1).max(100).trim(),
  photoUrl:      z.string().optional(),
  photos:        photosSchema,
  createAccount: z.boolean().optional(),
  password:      z.string().min(4).max(128).optional(),
});

const updatePersonnelSchema = z.object({
  lastName:      nameField().optional(),
  firstName:     nameField().optional(),
  middleInitial: z.string().trim().max(100).regex(NAME_REGEX, NAME_MSG).optional().or(z.literal("")),
  department:    z.string().min(1).max(100).trim().optional(),
  position:      z.string().min(1).max(100).trim().optional(),
  photoUrl:      z.string().optional(),
  photos:        photosSchema,
});

const router: IRouter = Router();

async function getSessionAccount(req: any): Promise<{ id: number; role: string; personnelId: number | null } | null> {
  const token = req.cookies?.session_token;
  if (!token) return null;
  const [session] = await db.select().from(sessionsTable).where(eq(sessionsTable.sessionToken, token)).limit(1);
  if (!session || session.expiresAt < new Date()) return null;
  const [account] = await db.select().from(accountsTable).where(eq(accountsTable.id, session.accountId)).limit(1);
  if (!account) return null;
  return { id: account.id, role: account.role, personnelId: account.personnelId ?? null };
}

/** Returns the admin's account ID on success, or null (after sending a response) on failure. */
async function requireAdmin(req: any, res: any): Promise<number | null> {
  const account = await getSessionAccount(req);
  if (!account) { res.status(401).json({ error: "Not authenticated" }); return null; }
  if (account.role !== "admin") { res.status(403).json({ error: "Admin access required" }); return null; }
  return account.id;
}

const VIEW_TYPES = ["front", "left", "right", "top"] as const;

async function getPhotosForPersonnel(personnelId: number) {
  const rows = await db
    .select()
    .from(personnelPhotosTable)
    .where(eq(personnelPhotosTable.personnelId, personnelId));
  const photos: Record<string, string> = {};
  for (const row of rows) { photos[row.viewType] = row.photoBase64; }
  return photos;
}

/** Normalize a full name for duplicate detection (lowercase, trim, collapse spaces). */
function normalizeName(firstName: string, middleInitial: string | null | undefined, lastName: string): string {
  return [firstName, middleInitial, lastName]
    .map(s => s?.trim() ?? "")
    .filter(Boolean)
    .join(" ")
    .toLowerCase()
    .replace(/\s+/g, " ");
}

router.get("/", async (req, res) => {
  const account = await getSessionAccount(req);
  if (!account) { res.status(401).json({ error: "Not authenticated" }); return; }

  let departmentFilter: string | null = null;

  if (account.role === "user") {
    if (account.personnelId) {
      const [p] = await db.select().from(personnelTable).where(eq(personnelTable.id, account.personnelId)).limit(1);
      if (p) departmentFilter = p.department;
    }
  } else if (account.role === "admin" && req.query.department) {
    departmentFilter = req.query.department as string;
  }

  const query = db
    .select({
      id:          personnelTable.id,
      lastName:    personnelTable.lastName,
      firstName:   personnelTable.firstName,
      middleInitial: personnelTable.middleInitial,
      employeeId:  personnelTable.employeeId,
      department:  personnelTable.department,
      position:    personnelTable.position,
      photoUrl:    personnelTable.photoUrl,
      createdAt:   personnelTable.createdAt,
      accountId:   accountsTable.id,
    })
    .from(personnelTable)
    .leftJoin(accountsTable, eq(accountsTable.personnelId, personnelTable.id));

  const rows = departmentFilter
    ? await query.where(eq(personnelTable.department, departmentFilter))
    : await query;

  res.json(rows.map((r) => ({
    id:           r.id,
    lastName:     r.lastName,
    firstName:    r.firstName,
    middleInitial: r.middleInitial ?? null,
    employeeId:   r.employeeId,
    department:   r.department,
    position:     r.position,
    photoUrl:     r.photoUrl ?? null,
    hasAccount:   r.accountId !== null,
    createdAt:    r.createdAt.toISOString(),
  })));
});

router.post("/", validate(createPersonnelSchema), async (req, res) => {
  const adminId = await requireAdmin(req, res);
  if (adminId === null) return;

  const { lastName, firstName, middleInitial, employeeId, department, position, photoUrl, photos, createAccount, password } = req.body;

  if (!photos?.front && !photoUrl) {
    res.status(400).json({ error: "At least a front-facing photo is required" });
    return;
  }

  // P5/P6 — validate each submitted photo (MIME type + size)
  const allPhotos: [string, string][] = [
    ...Object.entries(photos ?? {}),
    ...(photoUrl ? [["photoUrl", photoUrl] as [string, string]] : []),
  ];
  for (const [field, dataUri] of allPhotos) {
    if (!dataUri) continue;
    const check = validatePhoto(dataUri as string);
    if (!check.valid) {
      res.status(400).json({ error: `Invalid photo (${field}): ${check.error}` });
      return;
    }
  }

  // Employee ID uniqueness
  const [existingId] = await db.select().from(personnelTable).where(eq(personnelTable.employeeId, employeeId)).limit(1);
  if (existingId) { res.status(400).json({ error: "Employee ID already exists" }); return; }

  // Duplicate full name check (case-insensitive, includes middle initial)
  const normalizedFull = normalizeName(firstName, middleInitial, lastName);
  const [dupName] = await db
    .select({ id: personnelTable.id })
    .from(personnelTable)
    .where(
      sql`LOWER(TRIM(COALESCE(${personnelTable.firstName},'')) || CASE WHEN TRIM(COALESCE(${personnelTable.middleInitial},'')) != '' THEN ' ' || TRIM(${personnelTable.middleInitial}) ELSE '' END || ' ' || TRIM(COALESCE(${personnelTable.lastName},''))) = ${normalizedFull}`
    )
    .limit(1);
  if (dupName) {
    res.status(400).json({ error: "A personnel record with this name already exists. Check for different capitalisation." });
    return;
  }

  const frontPhoto = photos?.front || photoUrl || null;

  const [personnel] = await db
    .insert(personnelTable)
    .values({ lastName, firstName, middleInitial: middleInitial || null, employeeId, department, position, photoUrl: frontPhoto })
    .returning();

  if (photos && typeof photos === "object") {
    const photoInserts = VIEW_TYPES
      .filter((vt) => photos[vt] && typeof photos[vt] === "string")
      .map((vt) => ({ personnelId: personnel.id, viewType: vt, photoBase64: photos[vt] as string }));
    if (photoInserts.length > 0) {
      await db.insert(personnelPhotosTable).values(photoInserts);
    }
  }

  if (createAccount && password) {
    const [existingAccount] = await db.select().from(accountsTable).where(eq(accountsTable.username, employeeId)).limit(1);
    if (!existingAccount) {
      const passwordHash = await bcrypt.hash(password, 10);
      await db.insert(accountsTable).values({ username: employeeId, passwordHash, role: "user", personnelId: personnel.id });
    }
  }

  res.status(201).json({
    id:           personnel.id,
    lastName:     personnel.lastName,
    firstName:    personnel.firstName,
    middleInitial: personnel.middleInitial ?? null,
    employeeId:   personnel.employeeId,
    department:   personnel.department,
    position:     personnel.position,
    photoUrl:     personnel.photoUrl ?? null,
    hasAccount:   !!(createAccount && password),
    createdAt:    personnel.createdAt.toISOString(),
  });
  // D3 + D4 + D5
  securityLog("PERSONNEL_CREATED", { targetId: personnel.id, performedBy: adminId, ip: req.ip });
  void db.insert(auditLogsTable).values({
    performedBy: adminId,
    action: "CREATE_PERSONNEL",
    targetType: "personnel",
    targetId: personnel.id,
    detail: JSON.stringify({ employeeId: personnel.employeeId, name: `${personnel.firstName} ${personnel.lastName}` }),
    ip: req.ip ?? null,
  });
  void writePersonnelBackup();
});

router.get("/:id", async (req, res) => {
  const account = await getSessionAccount(req);
  if (!account) { res.status(401).json({ error: "Not authenticated" }); return; }

  const id = parseInt(req.params.id);
  if (isNaN(id)) { res.status(400).json({ error: "Invalid ID" }); return; }

  const rows = await db
    .select({
      id:          personnelTable.id,
      lastName:    personnelTable.lastName,
      firstName:   personnelTable.firstName,
      middleInitial: personnelTable.middleInitial,
      employeeId:  personnelTable.employeeId,
      department:  personnelTable.department,
      position:    personnelTable.position,
      photoUrl:    personnelTable.photoUrl,
      createdAt:   personnelTable.createdAt,
      accountId:   accountsTable.id,
    })
    .from(personnelTable)
    .leftJoin(accountsTable, eq(accountsTable.personnelId, personnelTable.id))
    .where(eq(personnelTable.id, id))
    .limit(1);

  if (rows.length === 0) { res.status(404).json({ error: "Personnel not found" }); return; }
  const r = rows[0];
  const photos = await getPhotosForPersonnel(r.id);
  res.json({
    id: r.id, lastName: r.lastName, firstName: r.firstName,
    middleInitial: r.middleInitial ?? null, employeeId: r.employeeId,
    department: r.department, position: r.position,
    photoUrl: r.photoUrl ?? null, photos,
    hasAccount: r.accountId !== null, createdAt: r.createdAt.toISOString(),
  });
});

router.put("/:id", validate(updatePersonnelSchema), async (req, res) => {
  const adminId = await requireAdmin(req, res);
  if (adminId === null) return;

  const id = parseInt(req.params.id);
  if (isNaN(id)) { res.status(400).json({ error: "Invalid ID" }); return; }

  const { lastName, firstName, middleInitial, department, position, photoUrl, photos } = req.body;

  // P5/P6 — validate each submitted photo (MIME type + size)
  const allPhotos: [string, string][] = [
    ...Object.entries(photos ?? {}),
    ...(photoUrl ? [["photoUrl", photoUrl] as [string, string]] : []),
  ];
  for (const [field, dataUri] of allPhotos) {
    if (!dataUri) continue;
    const check = validatePhoto(dataUri as string);
    if (!check.valid) {
      res.status(400).json({ error: `Invalid photo (${field}): ${check.error}` });
      return;
    }
  }
  const [existing] = await db.select().from(personnelTable).where(eq(personnelTable.id, id)).limit(1);
  if (!existing) { res.status(404).json({ error: "Personnel not found" }); return; }

  // Duplicate full name check (case-insensitive, includes middle initial), excluding the current record
  const newFirst  = firstName      ?? existing.firstName;
  const newMid    = middleInitial  !== undefined ? middleInitial : existing.middleInitial;
  const newLast   = lastName       ?? existing.lastName;
  const normalizedFull = normalizeName(newFirst, newMid, newLast);
  const [dupName] = await db
    .select({ id: personnelTable.id })
    .from(personnelTable)
    .where(
      and(
        sql`LOWER(TRIM(COALESCE(${personnelTable.firstName},'')) || CASE WHEN TRIM(COALESCE(${personnelTable.middleInitial},'')) != '' THEN ' ' || TRIM(${personnelTable.middleInitial}) ELSE '' END || ' ' || TRIM(COALESCE(${personnelTable.lastName},''))) = ${normalizedFull}`,
        sql`${personnelTable.id} != ${id}`
      )
    )
    .limit(1);
  if (dupName) {
    res.status(400).json({ error: "Another personnel record with this name already exists." });
    return;
  }

  const frontPhoto = photos?.front !== undefined ? photos.front : (photoUrl !== undefined ? photoUrl : existing.photoUrl);

  const [updated] = await db
    .update(personnelTable)
    .set({
      lastName:      newLast,
      firstName:     newFirst,
      middleInitial: middleInitial !== undefined ? middleInitial : existing.middleInitial,
      department:    department    ?? existing.department,
      position:      position      ?? existing.position,
      photoUrl:      frontPhoto,
    })
    .where(eq(personnelTable.id, id))
    .returning();

  if (photos && typeof photos === "object") {
    for (const vt of VIEW_TYPES) {
      if (photos[vt] !== undefined && photos[vt] !== null) {
        const [existingPhoto] = await db.select().from(personnelPhotosTable)
          .where(and(eq(personnelPhotosTable.personnelId, id), eq(personnelPhotosTable.viewType, vt)))
          .limit(1);
        if (existingPhoto) {
          await db.update(personnelPhotosTable)
            .set({ photoBase64: photos[vt] })
            .where(and(eq(personnelPhotosTable.personnelId, id), eq(personnelPhotosTable.viewType, vt)));
        } else {
          await db.insert(personnelPhotosTable).values({ personnelId: id, viewType: vt, photoBase64: photos[vt] });
        }
      }
    }
  }

  const [accRow] = await db.select().from(accountsTable).where(eq(accountsTable.personnelId, id)).limit(1);
  const updatedPhotos = await getPhotosForPersonnel(id);
  res.json({
    id: updated.id, lastName: updated.lastName, firstName: updated.firstName,
    middleInitial: updated.middleInitial ?? null, employeeId: updated.employeeId,
    department: updated.department, position: updated.position,
    photoUrl: updated.photoUrl ?? null, photos: updatedPhotos,
    hasAccount: !!accRow, createdAt: updated.createdAt.toISOString(),
  });
  // D3 + D4 + D5
  securityLog("PERSONNEL_UPDATED", { targetId: id, performedBy: adminId, ip: req.ip });
  void db.insert(auditLogsTable).values({
    performedBy: adminId,
    action: "UPDATE_PERSONNEL",
    targetType: "personnel",
    targetId: id,
    detail: JSON.stringify({ employeeId: updated.employeeId, name: `${updated.firstName} ${updated.lastName}` }),
    ip: req.ip ?? null,
  });
  void writePersonnelBackup();
});

router.delete("/:id", async (req, res) => {
  const adminId = await requireAdmin(req, res);
  if (adminId === null) return;

  const id = parseInt(req.params.id);
  if (isNaN(id)) { res.status(400).json({ error: "Invalid ID" }); return; }

  const [existing] = await db.select().from(personnelTable).where(eq(personnelTable.id, id)).limit(1);
  if (!existing) { res.status(404).json({ error: "Personnel not found" }); return; }

  await db.delete(personnelTable).where(eq(personnelTable.id, id));
  res.json({ message: "Personnel deleted" });
  // D3 + D4 + D5
  securityLog("PERSONNEL_DELETED", { targetId: id, performedBy: adminId, ip: req.ip });
  void db.insert(auditLogsTable).values({
    performedBy: adminId,
    action: "DELETE_PERSONNEL",
    targetType: "personnel",
    targetId: id,
    detail: JSON.stringify({ employeeId: existing.employeeId, name: `${existing.firstName} ${existing.lastName}` }),
    ip: req.ip ?? null,
  });
  void writePersonnelBackup();
});

export default router;
