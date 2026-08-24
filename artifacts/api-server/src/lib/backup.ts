/**
 * D5 — Personnel backup (triggered on every create/update/delete)
 * D6 — Attendance log backup (nightly cron at midnight)
 */
import { db } from "@workspace/db";
import { personnelTable, attendanceLogsTable } from "@workspace/db";
import fs from "fs";
import path from "path";
import cron from "node-cron";

// Resolve from process.cwd() so paths work regardless of __dirname in bundled output
const PERSONNEL_BACKUP_DIR  = path.resolve(process.cwd(), "personnel_backup");
const ATTENDANCE_BACKUP_DIR = path.resolve(process.cwd(), "attendance_backup");

// ── D5: Personnel backup ──────────────────────────────────────────────────────

export async function writePersonnelBackup(): Promise<void> {
  try {
    const rows = await db.select().from(personnelTable).orderBy(personnelTable.id);

    const header = [
      "ID", "Employee ID", "Last Name", "First Name",
      "Middle Initial", "Department", "Position", "Created At",
    ].join(",");

    const lines = rows.map((r) =>
      [
        r.id,
        r.employeeId,
        `"${r.lastName}"`,
        `"${r.firstName}"`,
        r.middleInitial ?? "",
        `"${r.department}"`,
        `"${r.position}"`,
        r.createdAt.toISOString(),
      ].join(",")
    );

    const csv = [
      `# BatStateU Personnel Monitoring — Personnel Backup`,
      `# Generated: ${new Date().toISOString()}`,
      `# Total records: ${rows.length}`,
      header,
      ...lines,
    ].join("\n");

    if (!fs.existsSync(PERSONNEL_BACKUP_DIR)) {
      fs.mkdirSync(PERSONNEL_BACKUP_DIR, { recursive: true });
    }
    fs.writeFileSync(path.join(PERSONNEL_BACKUP_DIR, "personnel_backup.csv"), csv, "utf-8");
  } catch (err) {
    console.error("[PersonnelBackup] Failed:", err);
  }
}

// ── D6: Attendance log backup ─────────────────────────────────────────────────

export async function writeAttendanceBackup(): Promise<void> {
  try {
    const logs = await db.select().from(attendanceLogsTable).orderBy(attendanceLogsTable.timestamp);

    const header = [
      "ID", "Employee ID", "Name", "Department", "Position", "Log Type", "Timestamp",
    ].join(",");

    const lines = logs.map((l) =>
      [
        l.id,
        l.employeeId,
        `"${l.name}"`,
        `"${l.department}"`,
        `"${l.position}"`,
        l.logType,
        l.timestamp.toISOString(),
      ].join(",")
    );

    const csv = [
      `# BatStateU Personnel Monitoring — Attendance Log Backup`,
      `# Generated: ${new Date().toISOString()}`,
      `# Total records: ${logs.length}`,
      header,
      ...lines,
    ].join("\n");

    if (!fs.existsSync(ATTENDANCE_BACKUP_DIR)) {
      fs.mkdirSync(ATTENDANCE_BACKUP_DIR, { recursive: true });
    }
    const filename = `attendance_${new Date().toISOString().split("T")[0]}.csv`;
    fs.writeFileSync(path.join(ATTENDANCE_BACKUP_DIR, filename), csv, "utf-8");
  } catch (err) {
    console.error("[AttendanceBackup] Failed:", err);
  }
}

// ── D6: Scheduler — runs nightly at midnight ──────────────────────────────────

export function startBackupScheduler(): void {
  cron.schedule("0 0 * * *", () => {
    void writeAttendanceBackup();
    void writePersonnelBackup();
  });
}
