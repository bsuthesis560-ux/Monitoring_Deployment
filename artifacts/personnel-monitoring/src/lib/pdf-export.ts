import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";
import {
  format,
  startOfWeek, endOfWeek,
  startOfMonth, endOfMonth,
  getWeek, addDays,
} from "date-fns";

// ── Interfaces ────────────────────────────────────────────────────────────────

export interface AttendanceLog {
  id:           number;
  employeeId:   string;
  name:         string;
  department:   string;
  position:     string;
  logType:      string;
  timestamp:    string;
}

export type SummaryPeriod = "daily" | "weekly" | "monthly" | "custom";

/** Daily: one row per employee (morning + afternoon sessions). */
export interface EmployeeTimeRow {
  name:     string;
  employeeId?: string;
  timeIn1:  string;
  timeOut1: string;
  timeIn2:  string;
  timeOut2: string;
}

/** Weekly / Monthly: one row per calendar day. */
export interface DayRow {
  dateLabel: string;   // e.g. "Mon, May 5"
  timeIn:    string;   // e.g. "7:02 AM"  or "—"
  timeOut:   string;   // e.g. "5:15 PM"  or "—"
}

/** Weekly: one record per employee containing their daily rows for the week. */
export interface WeeklyEmployeeRecord {
  name:       string;
  employeeId: string;
  days:       DayRow[];  // up to 6 days (Mon–Sat)
}

/** Monthly: one record per employee, broken into week blocks. */
export interface MonthlyWeekBlock {
  weekLabel: string;   // "WEEK 1" … "WEEK 4"
  days:      DayRow[]; // up to 6 days (Mon–Sat) — blank if outside month
}

export interface MonthlyEmployeeRecord {
  name:       string;
  employeeId: string;
  weeks:      MonthlyWeekBlock[];
}

// ── Colors (shared) ───────────────────────────────────────────────────────────

const BLUE_DARK   = [10,  36,  99]  as [number,number,number];
const BLUE_MED    = [37,  99,  235] as [number,number,number];
const WHITE       = [255, 255, 255] as [number,number,number];
const GREEN_BG    = [220, 252, 231] as [number,number,number];
const GREEN_TEXT  = [22,  101, 52]  as [number,number,number];
const RED_BG      = [254, 226, 226] as [number,number,number];
const RED_TEXT    = [153, 27,  27]  as [number,number,number];
const ALT_ROW     = [245, 248, 255] as [number,number,number];
const WEEK_HEADER_BG   = [239, 246, 255] as [number,number,number];
const WEEK_HEADER_TEXT = [10,  36,  99]  as [number,number,number];

// ── Layout constants ──────────────────────────────────────────────────────────

const MARGIN   = 14;
const PAGE_W   = 210;
const PAGE_H   = 297;
const TOTAL_PG = "{totalPages}";  // placeholder for jsPDF putTotalPages

// Height in mm taken by the header block (logo + text + subheader + generated line)
// Used as margin.top on continuation pages so autoTable leaves room for the header.
const HEADER_BLOCK_H = 58;

// ── Period label helpers ──────────────────────────────────────────────────────

export function periodLabel(
  period: SummaryPeriod,
  dateStr: string,
  customStart?: string,
  customEnd?: string,
): string {
  if (period === "custom") {
    if (customStart && customEnd) {
      const s = new Date(customStart + "T00:00:00");
      const e = new Date(customEnd   + "T00:00:00");
      return `${format(s, "MMM d, yyyy")} \u2013 ${format(e, "MMM d, yyyy")}`;
    }
    return customStart
      ? format(new Date(customStart + "T00:00:00"), "MMM d, yyyy")
      : "Custom Range";
  }
  const d = new Date(dateStr + "T00:00:00");
  if (period === "daily")   return format(d, "MMMM d, yyyy");
  if (period === "weekly") {
    const ws = startOfWeek(d, { weekStartsOn: 1 });
    const we = endOfWeek(d,   { weekStartsOn: 1 });
    return `${format(ws, "MMM d")} \u2013 ${format(we, "MMM d, yyyy")}`;
  }
  return format(d, "MMMM yyyy");
}

export function periodSummaryTitle(period: SummaryPeriod): string {
  if (period === "daily")   return "Daily Attendance Summary";
  if (period === "weekly")  return "Weekly Attendance Summary";
  if (period === "monthly") return "Monthly Attendance Summary";
  return "Custom Attendance Summary";
}

function filenamePeriod(period: SummaryPeriod, dateStr: string, customStart?: string): string {
  const d = new Date((customStart || dateStr) + "T00:00:00");
  if (period === "daily")   return `${format(d,"MMM")}${format(d,"d")}_${format(d,"yyyy")}`;
  if (period === "weekly")  return `Week${getWeek(d,{weekStartsOn:1})}_${format(d,"MMMyyyy")}`;
  if (period === "monthly") return format(d, "MMMyyyy");
  return format(d, "yyyyMMdd");
}

function periodSuffix(period: SummaryPeriod): string {
  if (period === "daily")   return "DailySummary";
  if (period === "weekly")  return "WeeklySummary";
  if (period === "monthly") return "MonthlySummary";
  return "CustomSummary";
}

function sanitizeDeptForFilename(dept: string): string {
  return dept.replace(/[^a-zA-Z0-9]/g, "_").replace(/_+/g, "_").replace(/^_|_$/g, "");
}

// ── Logo loader ───────────────────────────────────────────────────────────────

async function loadLogo(): Promise<{ url: string; w: number; h: number } | null> {
  try {
    const resp = await fetch("/images/bsu-logo-bg.png");
    const blob = await resp.blob();
    const url  = await new Promise<string>((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.readAsDataURL(blob);
    });
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const i = new Image();
      i.onload  = () => resolve(i);
      i.onerror = reject;
      i.src = url;
    });
    const logoH = 18;
    const logoW = (img.naturalWidth / img.naturalHeight) * logoH;
    return { url, w: logoW, h: logoH };
  } catch {
    return null;
  }
}

// ── Header renderer  (UNCHANGED — same colors, fonts, layout as original) ────
// Returns the Y coordinate where body content should start.

function renderPageHeader(
  doc:         jsPDF,
  logo:        { url: string; w: number; h: number } | null,
  titleLabel:  string,
  periodStr:   string,
  dept:        string,
  empCount:    number,
  generatedAt: Date,
): number {
  const headerY = 12;

  // Logo
  if (logo) doc.addImage(logo.url, "PNG", MARGIN, headerY - 6, logo.w, logo.h);
  const textX = logo ? MARGIN + logo.w + 5 : MARGIN;

  // University name block (colors/fonts unchanged)
  doc.setFont("helvetica", "bold");
  doc.setFontSize(13);
  doc.setTextColor(...BLUE_DARK);
  doc.text("BATANGAS STATE UNIVERSITY", textX, headerY);

  doc.setFontSize(8);
  doc.setTextColor(180, 20, 20);
  doc.text("THE NATIONAL ENGINEERING UNIVERSITY \u2014 LIPA CAMPUS", textX, headerY + 5);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(7);
  doc.setTextColor(80, 80, 80);
  doc.text("College of Engineering and Technology (CET)", textX, headerY + 10);

  doc.setFontSize(6.5);
  doc.setTextColor(140, 140, 140);
  doc.text("BatStateU-REC-ATT-11", PAGE_W - MARGIN, headerY, { align: "right" });

  // Horizontal rule
  doc.setDrawColor(...BLUE_DARK);
  doc.setLineWidth(0.6);
  doc.line(MARGIN, headerY + 14, PAGE_W - MARGIN, headerY + 14);

  // Sub-header block
  const subY = headerY + 21;

  doc.setFont("helvetica", "bold");
  doc.setFontSize(11);
  doc.setTextColor(15, 15, 15);
  doc.text(titleLabel.toUpperCase(), PAGE_W / 2, subY, { align: "center" });

  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(...BLUE_DARK);
  doc.text(periodStr, PAGE_W / 2, subY + 6, { align: "center" });

  doc.setFontSize(8);
  doc.setTextColor(70, 70, 70);
  doc.text(`Department / Sub-unit:  ${dept}`, PAGE_W / 2, subY + 12, { align: "center" });

  doc.setFontSize(7);
  doc.setTextColor(120, 120, 120);
  doc.text(
    `Generated: ${format(generatedAt, "MMMM d, yyyy  hh:mm aa")}   |   ${empCount} employee(s)`,
    PAGE_W / 2, subY + 18, { align: "center" },
  );

  return subY + 24; // tableStartY
}

// ── Footer renderer (called after putTotalPages is applied) ──────────────────

function renderFooterOnPage(doc: jsPDF, pageNum: number) {
  const y = PAGE_H - 8;
  doc.setPage(pageNum);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(7);
  doc.setTextColor(150, 150, 150);
  doc.text(
    `This is a system-generated document from the BatStateU Personnel Monitoring System.  |  Confidential  |  Page ${pageNum} of ${TOTAL_PG}`,
    PAGE_W / 2, y, { align: "center" },
  );
}

// ── Data builders ─────────────────────────────────────────────────────────────

/** Daily: group logs by employee, pick first two TIME_IN and TIME_OUT events. */
function buildDailyRows(logs: AttendanceLog[]): EmployeeTimeRow[] {
  const sorted = [...logs].sort(
    (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime(),
  );
  const map = new Map<string, { ins: Date[]; outs: Date[]; employeeId: string }>();
  for (const log of sorted) {
    const key = log.name.trim();
    if (!map.has(key)) map.set(key, { ins: [], outs: [], employeeId: log.employeeId });
    const entry = map.get(key)!;
    const dt = new Date(log.timestamp);
    if (log.logType === "TIME_IN")  entry.ins.push(dt);
    if (log.logType === "TIME_OUT") entry.outs.push(dt);
  }
  const result: EmployeeTimeRow[] = [];
  for (const [name, entry] of map) {
    result.push({
      name,
      employeeId: entry.employeeId,
      timeIn1:  entry.ins.length  > 0 ? format(entry.ins[0],  "hh:mm aa") : "\u2014",
      timeOut1: entry.outs.length > 0 ? format(entry.outs[0], "hh:mm aa") : "\u2014",
      timeIn2:  entry.ins.length  > 1 ? format(entry.ins[1],  "hh:mm aa") : "\u2014",
      timeOut2: entry.outs.length > 1 ? format(entry.outs[entry.outs.length - 1], "hh:mm aa") : "\u2014",
    });
  }
  return result;
}

/** Weekly: group logs by employee and by calendar day. Generates Mon–Sat grid. */
function buildWeeklyData(logs: AttendanceLog[], dateStr: string): WeeklyEmployeeRecord[] {
  const d        = new Date(dateStr + "T00:00:00");
  const weekStart = startOfWeek(d, { weekStartsOn: 1 }); // Monday

  // Generate Mon–Sat (6 days)
  const days: Date[] = Array.from({ length: 6 }, (_, i) => addDays(weekStart, i));

  // Sort for deterministic order
  const sorted = [...logs].sort(
    (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime(),
  );

  // Map: "employeeId||name" → { ins per day, outs per day }
  type DayMap = Map<string, { ins: Date[]; outs: Date[] }>;
  const empMap = new Map<string, { name: string; employeeId: string; dayMap: DayMap }>();

  for (const log of sorted) {
    const key = `${log.employeeId}||${log.name.trim()}`;
    if (!empMap.has(key)) empMap.set(key, { name: log.name.trim(), employeeId: log.employeeId, dayMap: new Map() });
    const emp     = empMap.get(key)!;
    const dayKey  = format(new Date(log.timestamp), "yyyy-MM-dd");
    if (!emp.dayMap.has(dayKey)) emp.dayMap.set(dayKey, { ins: [], outs: [] });
    const dl = emp.dayMap.get(dayKey)!;
    if (log.logType === "TIME_IN")  dl.ins.push(new Date(log.timestamp));
    if (log.logType === "TIME_OUT") dl.outs.push(new Date(log.timestamp));
  }

  return Array.from(empMap.values()).map(({ name, employeeId, dayMap }) => ({
    name,
    employeeId,
    days: days.map((day) => {
      const dk  = format(day, "yyyy-MM-dd");
      const dl  = dayMap.get(dk);
      return {
        dateLabel: format(day, "EEE, MMM d"),
        timeIn:    dl && dl.ins.length  > 0 ? format(dl.ins[0],  "hh:mm aa") : "\u2014",
        timeOut:   dl && dl.outs.length > 0 ? format(dl.outs[dl.outs.length - 1], "hh:mm aa") : "\u2014",
      };
    }),
  }));
}

/** Monthly: group logs by employee and by calendar week (up to 5 weeks). */
function buildMonthlyData(logs: AttendanceLog[], dateStr: string): MonthlyEmployeeRecord[] {
  const d          = new Date(dateStr + "T00:00:00");
  const mStart     = startOfMonth(d);
  const mEnd       = endOfMonth(d);

  // Build week blocks: each block is Mon–Sat that falls within (or touching) the month.
  let cursor = startOfWeek(mStart, { weekStartsOn: 1 }); // first Monday on or before mStart
  const weekBlocks: { label: string; days: Date[] }[] = [];
  let wNum = 1;
  while (cursor <= mEnd && wNum <= 5) {
    const days = Array.from({ length: 6 }, (_, i) => addDays(cursor, i));
    // Include this block only if at least one day falls within the month
    if (days.some(day => day >= mStart && day <= mEnd)) {
      weekBlocks.push({ label: `WEEK ${wNum}`, days });
      wNum++;
    }
    cursor = addDays(cursor, 7);
  }

  const sorted = [...logs].sort(
    (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime(),
  );

  type DayMap = Map<string, { ins: Date[]; outs: Date[] }>;
  const empMap = new Map<string, { name: string; employeeId: string; dayMap: DayMap }>();

  for (const log of sorted) {
    const key = `${log.employeeId}||${log.name.trim()}`;
    if (!empMap.has(key)) empMap.set(key, { name: log.name.trim(), employeeId: log.employeeId, dayMap: new Map() });
    const emp    = empMap.get(key)!;
    const dayKey = format(new Date(log.timestamp), "yyyy-MM-dd");
    if (!emp.dayMap.has(dayKey)) emp.dayMap.set(dayKey, { ins: [], outs: [] });
    const dl = emp.dayMap.get(dayKey)!;
    if (log.logType === "TIME_IN")  dl.ins.push(new Date(log.timestamp));
    if (log.logType === "TIME_OUT") dl.outs.push(new Date(log.timestamp));
  }

  return Array.from(empMap.values()).map(({ name, employeeId, dayMap }) => ({
    name,
    employeeId,
    weeks: weekBlocks.map(({ label, days }) => ({
      weekLabel: label,
      days: days.map((day) => {
        const dk  = format(day, "yyyy-MM-dd");
        const dl  = dayMap.get(dk);
        const inMonth = day >= mStart && day <= mEnd;
        return {
          dateLabel: inMonth ? format(day, "EEE, MMM d") : `(${format(day, "MMM d")})`,
          timeIn:    inMonth && dl && dl.ins.length  > 0 ? format(dl.ins[0],  "hh:mm aa") : "\u2014",
          timeOut:   inMonth && dl && dl.outs.length > 0 ? format(dl.outs[dl.outs.length - 1], "hh:mm aa") : "\u2014",
        };
      }),
    })),
  }));
}

// ── Cell style helpers ────────────────────────────────────────────────────────

function timeInCell(value: string) {
  if (value === "\u2014") return { content: value, styles: { halign: "center" as const, textColor: [160,160,160] as [number,number,number] } };
  return { content: value, styles: { fillColor: GREEN_BG, textColor: GREEN_TEXT, fontStyle: "bold" as const, halign: "center" as const } };
}

function timeOutCell(value: string) {
  if (value === "\u2014") return { content: value, styles: { halign: "center" as const, textColor: [160,160,160] as [number,number,number] } };
  return { content: value, styles: { fillColor: RED_BG, textColor: RED_TEXT, fontStyle: "bold" as const, halign: "center" as const } };
}

function empHeaderCell(name: string, employeeId: string, colSpan: number) {
  return [{
    content: `NAME: ${name}      EMPLOYEE ID: ${employeeId}`,
    colSpan,
    styles: {
      fillColor:   BLUE_DARK,
      textColor:   WHITE,
      fontStyle:   "bold" as const,
      fontSize:    9,
      halign:      "left" as const,
      cellPadding: { top: 3.5, bottom: 3.5, left: 6, right: 4 },
    },
  }];
}

function weekHeaderCell(label: string, colSpan: number) {
  return [{
    content: label,
    colSpan,
    styles: {
      fillColor:   WEEK_HEADER_BG,
      textColor:   WEEK_HEADER_TEXT,
      fontStyle:   "bold" as const,
      fontSize:    8,
      halign:      "left" as const,
      cellPadding: { top: 2.5, bottom: 2.5, left: 10, right: 4 },
    },
  }];
}

// ── Daily body renderer ───────────────────────────────────────────────────────

function renderDailyBody(
  doc: jsPDF,
  rows: EmployeeTimeRow[],
  startY: number,
  logo: { url: string; w: number; h: number } | null,
  titleLabel: string,
  periodStr: string,
  dept: string,
  generatedAt: Date,
) {
  const tableRows = rows.map((r) => [r.name, r.timeIn1, r.timeOut1, r.timeIn2, r.timeOut2]);

  autoTable(doc, {
    startY,
    head: [[
      { content: "NAME",     styles: { halign: "left",   fillColor: BLUE_DARK, textColor: WHITE, fontStyle: "bold", fontSize: 8 } },
      { content: "TIME IN",  styles: { halign: "center", fillColor: BLUE_DARK, textColor: WHITE, fontStyle: "bold", fontSize: 8 } },
      { content: "TIME OUT", styles: { halign: "center", fillColor: BLUE_DARK, textColor: WHITE, fontStyle: "bold", fontSize: 8 } },
      { content: "TIME IN",  styles: { halign: "center", fillColor: BLUE_DARK, textColor: WHITE, fontStyle: "bold", fontSize: 8 } },
      { content: "TIME OUT", styles: { halign: "center", fillColor: BLUE_DARK, textColor: WHITE, fontStyle: "bold", fontSize: 8 } },
    ]],
    body: tableRows,
    columnStyles: {
      0: { cellWidth: 74,  fontStyle: "bold", fontSize: 9,   textColor: [20, 20, 20] },
      1: { cellWidth: 27,  halign: "center",  fontSize: 8.5, textColor: GREEN_TEXT   },
      2: { cellWidth: 27,  halign: "center",  fontSize: 8.5, textColor: RED_TEXT     },
      3: { cellWidth: 27,  halign: "center",  fontSize: 8.5, textColor: GREEN_TEXT   },
      4: { cellWidth: 27,  halign: "center",  fontSize: 8.5, textColor: RED_TEXT     },
    },
    styles: {
      cellPadding: { top: 2.5, bottom: 2.5, left: 4, right: 4 },
      overflow:    "ellipsize",
      lineColor:   [210, 215, 230],
      lineWidth:   0.2,
      fontSize:    8,
    },
    alternateRowStyles: { fillColor: ALT_ROW },
    margin: { left: MARGIN, right: MARGIN, top: HEADER_BLOCK_H },
    didDrawPage: (data) => {
      if (data.pageNumber > 1) {
        renderPageHeader(doc, logo, titleLabel, periodStr, dept, rows.length, generatedAt);
      }
    },
    didDrawCell: (data) => {
      if (data.section !== "body") return;
      const raw = String(data.cell.raw ?? "");
      if (raw === "\u2014" || raw === "") return;
      if (data.column.index === 1 || data.column.index === 3) {
        doc.setFillColor(...GREEN_BG);
        doc.rect(data.cell.x, data.cell.y, data.cell.width, data.cell.height, "F");
        doc.setFont("helvetica", "bold"); doc.setFontSize(8); doc.setTextColor(...GREEN_TEXT);
        doc.text(raw, data.cell.x + data.cell.width / 2, data.cell.y + data.cell.height / 2 + 1.2, { align: "center" });
      }
      if (data.column.index === 2 || data.column.index === 4) {
        doc.setFillColor(...RED_BG);
        doc.rect(data.cell.x, data.cell.y, data.cell.width, data.cell.height, "F");
        doc.setFont("helvetica", "bold"); doc.setFontSize(8); doc.setTextColor(...RED_TEXT);
        doc.text(raw, data.cell.x + data.cell.width / 2, data.cell.y + data.cell.height / 2 + 1.2, { align: "center" });
      }
    },
  });
}

// ── Weekly body renderer ──────────────────────────────────────────────────────
// Layout: NAME/ID header row (full-width, dark blue) then 6 day rows per employee.
// Columns: DATE & DAY | TIME IN | TIME OUT

function renderWeeklyBody(
  doc: jsPDF,
  employees: WeeklyEmployeeRecord[],
  startY: number,
  logo: { url: string; w: number; h: number } | null,
  titleLabel: string,
  periodStr: string,
  dept: string,
  generatedAt: Date,
) {
  // Build one combined body for all employees
  const body: any[][] = [];
  for (const emp of employees) {
    body.push(empHeaderCell(emp.name, emp.employeeId, 3));
    for (const day of emp.days) {
      body.push([
        { content: day.dateLabel, styles: { fontStyle: "bold" as const, textColor: [30, 30, 30] as [number,number,number] } },
        timeInCell(day.timeIn),
        timeOutCell(day.timeOut),
      ]);
    }
  }

  autoTable(doc, {
    startY,
    head: [[
      { content: "DATE & DAY", styles: { halign: "left",   fillColor: BLUE_DARK, textColor: WHITE, fontStyle: "bold", fontSize: 8 } },
      { content: "TIME IN",   styles: { halign: "center", fillColor: BLUE_DARK, textColor: WHITE, fontStyle: "bold", fontSize: 8 } },
      { content: "TIME OUT",  styles: { halign: "center", fillColor: BLUE_DARK, textColor: WHITE, fontStyle: "bold", fontSize: 8 } },
    ]],
    body,
    columnStyles: {
      0: { cellWidth: 72 },
      1: { cellWidth: 55, halign: "center" },
      2: { cellWidth: 55, halign: "center" },
    },
    styles: {
      cellPadding: { top: 2.5, bottom: 2.5, left: 4, right: 4 },
      overflow:    "ellipsize",
      lineColor:   [210, 215, 230],
      lineWidth:   0.2,
      fontSize:    8.5,
    },
    alternateRowStyles: { fillColor: ALT_ROW },
    margin: { left: MARGIN, right: MARGIN, top: HEADER_BLOCK_H },
    didDrawPage: (data) => {
      if (data.pageNumber > 1) {
        renderPageHeader(doc, logo, titleLabel, periodStr, dept, employees.length, generatedAt);
      }
    },
  });
}

// ── Monthly body renderer ─────────────────────────────────────────────────────
// Layout: NAME/ID header row, then WEEK n sub-header rows, then 6 day rows each.
// Columns: DATE & DAY | TIME IN | TIME OUT

function renderMonthlyBody(
  doc: jsPDF,
  employees: MonthlyEmployeeRecord[],
  startY: number,
  logo: { url: string; w: number; h: number } | null,
  titleLabel: string,
  periodStr: string,
  dept: string,
  generatedAt: Date,
) {
  const body: any[][] = [];
  for (const emp of employees) {
    body.push(empHeaderCell(emp.name, emp.employeeId, 3));
    for (const week of emp.weeks) {
      body.push(weekHeaderCell(week.weekLabel, 3));
      for (const day of week.days) {
        body.push([
          { content: day.dateLabel, styles: { fontStyle: "bold" as const, textColor: [30, 30, 30] as [number,number,number] } },
          timeInCell(day.timeIn),
          timeOutCell(day.timeOut),
        ]);
      }
    }
  }

  autoTable(doc, {
    startY,
    head: [[
      { content: "DATE & DAY", styles: { halign: "left",   fillColor: BLUE_DARK, textColor: WHITE, fontStyle: "bold", fontSize: 8 } },
      { content: "TIME IN",   styles: { halign: "center", fillColor: BLUE_DARK, textColor: WHITE, fontStyle: "bold", fontSize: 8 } },
      { content: "TIME OUT",  styles: { halign: "center", fillColor: BLUE_DARK, textColor: WHITE, fontStyle: "bold", fontSize: 8 } },
    ]],
    body,
    columnStyles: {
      0: { cellWidth: 72 },
      1: { cellWidth: 55, halign: "center" },
      2: { cellWidth: 55, halign: "center" },
    },
    styles: {
      cellPadding: { top: 2.5, bottom: 2.5, left: 4, right: 4 },
      overflow:    "ellipsize",
      lineColor:   [210, 215, 230],
      lineWidth:   0.2,
      fontSize:    8.5,
    },
    alternateRowStyles: { fillColor: ALT_ROW },
    margin: { left: MARGIN, right: MARGIN, top: HEADER_BLOCK_H },
    didDrawPage: (data) => {
      if (data.pageNumber > 1) {
        renderPageHeader(doc, logo, titleLabel, periodStr, dept, employees.length, generatedAt);
      }
    },
  });
}

// ── Page post-processing: add footers on every page ──────────────────────────

function applyFooters(doc: jsPDF) {
  const total = doc.getNumberOfPages();
  for (let p = 1; p <= total; p++) {
    renderFooterOnPage(doc, p);
  }
  doc.putTotalPages(TOTAL_PG);
}

// ── Core orchestrator — renders one department section onto doc ───────────────

async function renderDeptSection(
  doc: jsPDF,
  dept: string,
  logs: AttendanceLog[],
  period: SummaryPeriod,
  titleLabel: string,
  periodStr: string,
  logo: { url: string; w: number; h: number } | null,
  dateStr: string,
  generatedAt: Date,
) {
  if (period === "daily" || period === "custom") {
    const rows    = buildDailyRows(logs);
    const startY  = renderPageHeader(doc, logo, titleLabel, periodStr, dept, rows.length, generatedAt);
    renderDailyBody(doc, rows, startY, logo, titleLabel, periodStr, dept, generatedAt);

  } else if (period === "weekly") {
    const employees = buildWeeklyData(logs, dateStr);
    const startY    = renderPageHeader(doc, logo, titleLabel, periodStr, dept, employees.length, generatedAt);
    renderWeeklyBody(doc, employees, startY, logo, titleLabel, periodStr, dept, generatedAt);

  } else if (period === "monthly") {
    const employees = buildMonthlyData(logs, dateStr);
    const startY    = renderPageHeader(doc, logo, titleLabel, periodStr, dept, employees.length, generatedAt);
    renderMonthlyBody(doc, employees, startY, logo, titleLabel, periodStr, dept, generatedAt);
  }
}

// ── Public: export single department from real log data ───────────────────────

export async function exportDeptPDF(
  dept: string,
  deptLogs: AttendanceLog[],
  period: SummaryPeriod,
  dateStr: string,
  customStart?: string,
  customEnd?: string,
): Promise<void> {
  const logo       = await loadLogo();
  const label      = periodSummaryTitle(period);
  const pStr       = periodLabel(period, dateStr, customStart, customEnd);
  const doc        = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  const now        = new Date();

  await renderDeptSection(doc, dept, deptLogs, period, label, pStr, logo, dateStr, now);
  applyFooters(doc);

  const slug = sanitizeDeptForFilename(dept);
  doc.save(`${slug}_${periodSuffix(period)}_${filenamePeriod(period, dateStr, customStart)}.pdf`);
}

// ── Public: export all departments in one PDF ─────────────────────────────────

export async function exportAllDeptsPDF(
  deptGroups: Record<string, AttendanceLog[]>,
  period: SummaryPeriod,
  dateStr: string,
  customStart?: string,
  customEnd?: string,
): Promise<void> {
  const depts = Object.keys(deptGroups).sort();
  if (depts.length === 0) return;

  const logo  = await loadLogo();
  const label = periodSummaryTitle(period);
  const pStr  = periodLabel(period, dateStr, customStart, customEnd);
  const doc   = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  const now   = new Date();

  for (let i = 0; i < depts.length; i++) {
    if (i > 0) doc.addPage();
    await renderDeptSection(doc, depts[i], deptGroups[depts[i]], period, label, pStr, logo, dateStr, now);
  }

  applyFooters(doc);
  doc.save(`ALL_Departments_${periodSuffix(period)}_${filenamePeriod(period, dateStr, customStart)}.pdf`);
}

// ── Public: sample PDF exports ────────────────────────────────────────────────

/** Daily sample — accepts pre-built EmployeeTimeRow[]. */
export async function exportSamplePDF(
  dept: string,
  rows: EmployeeTimeRow[],
  period: SummaryPeriod,
  periodStr: string,
  filename: string,
): Promise<void> {
  const logo = await loadLogo();
  const label = periodSummaryTitle(period);
  const doc   = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  const now   = new Date();
  const startY = renderPageHeader(doc, logo, label, periodStr, dept, rows.length, now);
  renderDailyBody(doc, rows, startY, logo, label, periodStr, dept, now);
  applyFooters(doc);
  doc.save(filename);
}

/** Weekly sample — accepts pre-built WeeklyEmployeeRecord[]. */
export async function exportSampleWeeklyPDF(
  dept: string,
  employees: WeeklyEmployeeRecord[],
  periodStr: string,
  filename: string,
): Promise<void> {
  const logo   = await loadLogo();
  const label  = "Weekly Attendance Summary";
  const doc    = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  const now    = new Date();
  const startY = renderPageHeader(doc, logo, label, periodStr, dept, employees.length, now);
  renderWeeklyBody(doc, employees, startY, logo, label, periodStr, dept, now);
  applyFooters(doc);
  doc.save(filename);
}

/** Monthly sample — accepts pre-built MonthlyEmployeeRecord[]. */
export async function exportSampleMonthlyPDF(
  dept: string,
  employees: MonthlyEmployeeRecord[],
  periodStr: string,
  filename: string,
): Promise<void> {
  const logo   = await loadLogo();
  const label  = "Monthly Attendance Summary";
  const doc    = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  const now    = new Date();
  const startY = renderPageHeader(doc, logo, label, periodStr, dept, employees.length, now);
  renderMonthlyBody(doc, employees, startY, logo, label, periodStr, dept, now);
  applyFooters(doc);
  doc.save(filename);
}
