import { pgTable, text, serial, integer, timestamp, boolean } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

export const personnelTable = pgTable("personnel", {
  id: serial("id").primaryKey(),
  lastName: text("last_name").notNull(),
  firstName: text("first_name").notNull(),
  middleInitial: text("middle_initial"),
  employeeId: text("employee_id").notNull().unique(),
  department: text("department").notNull(),
  position: text("position").notNull(),
  photoUrl: text("photo_url"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertPersonnelSchema = createInsertSchema(personnelTable).omit({ id: true, createdAt: true });
export type InsertPersonnel = z.infer<typeof insertPersonnelSchema>;
export type Personnel = typeof personnelTable.$inferSelect;

export const personnelPhotosTable = pgTable("personnel_photos", {
  id: serial("id").primaryKey(),
  personnelId: integer("personnel_id").notNull().references(() => personnelTable.id, { onDelete: "cascade" }),
  viewType: text("view_type").notNull(),
  photoBase64: text("photo_base64").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export type PersonnelPhoto = typeof personnelPhotosTable.$inferSelect;

export const accountsTable = pgTable("accounts", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  passwordHash: text("password_hash").notNull(),
  role: text("role").notNull().default("user"),
  personnelId: integer("personnel_id").references(() => personnelTable.id, { onDelete: "cascade" }),
  // D2 — account lockout
  failedLoginAttempts: integer("failed_login_attempts").notNull().default(0),
  lockedUntil: timestamp("locked_until"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertAccountSchema = createInsertSchema(accountsTable).omit({ id: true, createdAt: true });
export type InsertAccount = z.infer<typeof insertAccountSchema>;
export type Account = typeof accountsTable.$inferSelect;

export const sessionsTable = pgTable("sessions", {
  id: serial("id").primaryKey(),
  sessionToken: text("session_token").notNull().unique(),
  accountId: integer("account_id").notNull().references(() => accountsTable.id, { onDelete: "cascade" }),
  expiresAt: timestamp("expires_at").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export type Session = typeof sessionsTable.$inferSelect;

export const monitoringLogsTable = pgTable("attendance_logs", {
  id: serial("id").primaryKey(),
  personnelId: integer("personnel_id").references(() => personnelTable.id, { onDelete: "set null" }),
  employeeId: text("employee_id").notNull(),
  name: text("name").notNull(),
  department: text("department").notNull(),
  position: text("position").notNull().default(""),
  logType: text("log_type").notNull().default("entry"),
  timestamp: timestamp("timestamp").defaultNow().notNull(),
});

export type MonitoringLog = typeof monitoringLogsTable.$inferSelect;

// D4 — Audit trail for every admin action
export const auditLogsTable = pgTable("audit_logs", {
  id:          serial("id").primaryKey(),
  performedBy: integer("performed_by"),           // account ID of the admin
  action:      text("action").notNull(),           // e.g. "DELETE_PERSONNEL"
  targetType:  text("target_type").notNull(),      // e.g. "personnel", "account"
  targetId:    integer("target_id"),
  detail:      text("detail"),                     // JSON snapshot of before/after
  ip:          text("ip"),
  timestamp:   timestamp("timestamp").defaultNow().notNull(),
});

export type AuditLog = typeof auditLogsTable.$inferSelect;
