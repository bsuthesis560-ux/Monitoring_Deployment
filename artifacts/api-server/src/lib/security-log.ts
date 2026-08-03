/**
 * D3 — Structured security event logger.
 * Writes to the existing Pino logger with a `security: true` marker so
 * events can be filtered / alerted on separately from normal HTTP traffic.
 */
import { logger } from "./logger";

export type SecurityEvent =
  | "LOGIN_SUCCESS"
  | "LOGIN_FAILED"
  | "LOGIN_LOCKED"
  | "LOGOUT"
  | "ACCOUNT_CREATED"
  | "ACCOUNT_UPDATED"
  | "ACCOUNT_DELETED"
  | "PERSONNEL_CREATED"
  | "PERSONNEL_UPDATED"
  | "PERSONNEL_DELETED"
  | "INVALID_API_KEY";

export interface SecurityContext {
  username?: string;
  ip?: string;
  targetId?: number;
  performedBy?: number;
  detail?: string;
}

export function securityLog(event: SecurityEvent, context: SecurityContext = {}): void {
  logger.warn({ security: true, event, ...context }, `[SECURITY] ${event}`);
}
