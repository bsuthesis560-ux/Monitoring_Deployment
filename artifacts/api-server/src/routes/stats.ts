import { Router, type IRouter } from "express";
import { db } from "@workspace/db";
import { personnelTable, monitoringLogsTable, accountsTable, sessionsTable } from "@workspace/db";
import { eq, and, gte, lte } from "drizzle-orm";

const router: IRouter = Router();

const DEPARTMENTS = ["CET Faculty", "ICT", "ADMIN", "HR"];

async function requireAuth(req: any, res: any): Promise<{ role: string } | null> {
  const token = req.cookies?.session_token;
  if (!token) { res.status(401).json({ error: "Not authenticated" }); return null; }
  const [session] = await db.select().from(sessionsTable).where(eq(sessionsTable.sessionToken, token)).limit(1);
  if (!session || session.expiresAt < new Date()) { res.status(401).json({ error: "Session expired" }); return null; }
  const [account] = await db.select().from(accountsTable).where(eq(accountsTable.id, session.accountId)).limit(1);
  if (!account) { res.status(401).json({ error: "Not authenticated" }); return null; }
  return { role: account.role };
}

router.get("/", async (req, res) => {
  const auth = await requireAuth(req, res);
  if (!auth) return;

  const startOfDay = new Date();
  startOfDay.setHours(0, 0, 0, 0);
  const endOfDay = new Date();
  endOfDay.setHours(23, 59, 59, 999);

  const allPersonnel = await db.select({
    id: personnelTable.id,
    department: personnelTable.department,
    employeeId: personnelTable.employeeId,
  }).from(personnelTable);

  const todayLogs = await db.select({
    employeeId: monitoringLogsTable.employeeId,
    department: monitoringLogsTable.department,
  })
    .from(monitoringLogsTable)
    .where(and(
      gte(monitoringLogsTable.timestamp, startOfDay),
      lte(monitoringLogsTable.timestamp, endOfDay)
    ));

  const presentToday = new Set(todayLogs.map(l => l.employeeId));

  const stats = DEPARTMENTS.map((dept) => {
    const deptPersonnel = allPersonnel.filter(p => p.department === dept);
    const total = deptPersonnel.length;
    const present = deptPersonnel.filter(p => presentToday.has(p.employeeId)).length;
    return {
      department: dept,
      total,
      presentToday: present,
      absentToday: total - present,
    };
  });

  res.json(stats);
});

export default router;
