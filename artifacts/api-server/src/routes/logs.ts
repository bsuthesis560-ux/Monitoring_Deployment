import { Router, type IRouter } from "express";
import { db } from "@workspace/db";
import { attendanceLogsTable, accountsTable, personnelTable, sessionsTable, personnelPhotosTable } from "@workspace/db";
import { isValidPhoto } from "../lib/validate-photo";
import { eq, desc, and, gte } from "drizzle-orm";

const router: IRouter = Router();

async function getSessionUser(req: any): Promise<{ accountId: number; role: string; personnelId: number | null } | null> {
  const token = req.cookies?.session_token;
  if (!token) return null;

  const [session] = await db
    .select()
    .from(sessionsTable)
    .where(eq(sessionsTable.sessionToken, token))
    .limit(1);

  if (!session || session.expiresAt < new Date()) return null;

  const [account] = await db
    .select()
    .from(accountsTable)
    .where(eq(accountsTable.id, session.accountId))
    .limit(1);

  if (!account) return null;
  return { accountId: account.id, role: account.role, personnelId: account.personnelId ?? null };
}

router.get("/", async (req, res) => {
  const user = await getSessionUser(req);
  if (!user) {
    res.status(401).json({ error: "Not authenticated" });
    return;
  }

  let department: string | null = null;

  if (user.role === "user" && user.personnelId) {
    const [personnel] = await db
      .select()
      .from(personnelTable)
      .where(eq(personnelTable.id, user.personnelId))
      .limit(1);
    if (personnel) {
      department = personnel.department;
    }
  } else if (user.role === "admin" && req.query.department) {
    department = req.query.department as string;
  }

  const requestedLimit = req.query.limit ? parseInt(req.query.limit as string) : 500;
  const limit = Math.min(Math.max(1, requestedLimit), 2000);

  const logs = department
    ? await db
        .select()
        .from(attendanceLogsTable)
        .where(eq(attendanceLogsTable.department, department))
        .orderBy(desc(attendanceLogsTable.timestamp))
        .limit(limit)
    : await db
        .select()
        .from(attendanceLogsTable)
        .orderBy(desc(attendanceLogsTable.timestamp))
        .limit(limit);

  res.json(
    logs.map((l) => ({
      id: l.id,
      employeeId: l.employeeId,
      name: l.name,
      department: l.department,
      position: l.position,
      logType: l.logType,
      timestamp: l.timestamp.toISOString(),
    }))
  );
});

router.post("/", async (req, res) => {
  const apiKey = req.headers["x-api-key"];
  const expectedKey = process.env.FACIAL_RECOGNITION_API_KEY;

  if (!expectedKey || apiKey !== expectedKey) {
    res.status(403).json({ error: "Invalid API key" });
    return;
  }

  const { employeeId, logType } = req.body;

  if (!employeeId) {
    res.status(400).json({ error: "employeeId is required" });
    return;
  }

  const [personnel] = await db
    .select()
    .from(personnelTable)
    .where(eq(personnelTable.employeeId, employeeId))
    .limit(1);

  if (!personnel) {
    res.status(404).json({ error: "Personnel not found for employeeId: " + employeeId });
    return;
  }

  const cooldownSeconds = 8;
  const cooldownTime = new Date(Date.now() - cooldownSeconds * 1000);

  const recentLog = await db
    .select()
    .from(attendanceLogsTable)
    .where(
      and(
        eq(attendanceLogsTable.employeeId, employeeId),
        gte(attendanceLogsTable.timestamp, cooldownTime)
      )
    )
    .limit(1);

  if (recentLog.length > 0) {
    res.status(200).json({ message: "Duplicate suppressed (cooldown active)", duplicate: true });
    return;
  }

  const todayStart = new Date();
  todayStart.setHours(0, 0, 0, 0);

  const [lastLogToday] = await db
    .select()
    .from(attendanceLogsTable)
    .where(
      and(
        eq(attendanceLogsTable.employeeId, employeeId),
        gte(attendanceLogsTable.timestamp, todayStart)
      )
    )
    .orderBy(desc(attendanceLogsTable.timestamp))
    .limit(1);

  let nextLogType: string;
  if (!lastLogToday) {
    nextLogType = "TIME_IN";
  } else if (lastLogToday.logType === "TIME_IN") {
    nextLogType = "TIME_OUT";
  } else {
    nextLogType = "TIME_IN";
  }

  const [log] = await db
    .insert(attendanceLogsTable)
    .values({
      personnelId: personnel.id,
      employeeId: personnel.employeeId,
      name: `${personnel.firstName} ${personnel.lastName}`,
      department: personnel.department,
      position: personnel.position,
      logType: nextLogType,
    })
    .returning();

  res.status(201).json({
    id: log.id,
    employeeId: log.employeeId,
    name: log.name,
    department: log.department,
    position: log.position,
    logType: log.logType,
    timestamp: log.timestamp.toISOString(),
  });
});

router.get("/personnel-photos", async (req, res) => {
  const apiKey = req.headers["x-api-key"];
  const expectedKey = process.env.FACIAL_RECOGNITION_API_KEY;

  if (!expectedKey || apiKey !== expectedKey) {
    res.status(403).json({ error: "Invalid API key" });
    return;
  }

  const allPersonnel = await db
    .select({
      id: personnelTable.id,
      employeeId: personnelTable.employeeId,
      firstName: personnelTable.firstName,
      lastName: personnelTable.lastName,
      department: personnelTable.department,
      photoUrl: personnelTable.photoUrl,
    })
    .from(personnelTable);

  const allPhotos = await db.select().from(personnelPhotosTable);

  const photosByPersonnelId: Record<number, { viewType: string; photoBase64: string }[]> = {};
  for (const photo of allPhotos) {
    if (!photosByPersonnelId[photo.personnelId]) {
      photosByPersonnelId[photo.personnelId] = [];
    }
    photosByPersonnelId[photo.personnelId].push({ viewType: photo.viewType, photoBase64: photo.photoBase64 });
  }

  const result: any[] = [];

  for (const p of allPersonnel) {
    const multiPhotos = photosByPersonnelId[p.id] ?? [];

    if (multiPhotos.length > 0) {
      for (const photo of multiPhotos) {
        if (photo.photoBase64 && isValidPhoto(photo.photoBase64)) {
          result.push({
            employeeId: p.employeeId,
            name: `${p.firstName} ${p.lastName}`,
            department: p.department,
            viewType: photo.viewType,
            photoBase64: photo.photoBase64,
          });
        }
      }
    } else if (p.photoUrl && isValidPhoto(p.photoUrl)) {
      result.push({
        employeeId: p.employeeId,
        name: `${p.firstName} ${p.lastName}`,
        department: p.department,
        viewType: "front",
        photoBase64: p.photoUrl,
      });
    }
  }

  res.json(result);
});

export default router;
