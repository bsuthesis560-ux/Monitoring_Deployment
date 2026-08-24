import React, { useState } from "react";
import { AppLayout } from "@/components/layout/AppLayout";
import { Download, FileText, Calendar, CalendarDays, CalendarRange } from "lucide-react";
import { exportSamplePDF, type EmployeeTimeRow, type SummaryPeriod } from "@/lib/pdf-export";

// ── Sample employee data ──────────────────────────────────────────────────────

const DEPT = "College of Engineering Technology";

// Daily — May 16, 2025
const DAILY_ROWS: EmployeeTimeRow[] = [
  { name: "Christian Lloyd Brito",    timeIn1: "6:50 AM",  timeOut1: "11:40 AM", timeIn2: "1:30 PM",  timeOut2: "5:00 PM"  },
  { name: "Norly Boy Bisso",           timeIn1: "7:00 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "5:30 PM"  },
  { name: "Patrick Falcutila",         timeIn1: "8:10 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "6:00 PM"  },
  { name: "Althea Fadriquela",         timeIn1: "6:45 AM",  timeOut1: "12:00 PM", timeIn2: "1:30 PM",  timeOut2: "4:00 PM"  },
  { name: "Wayne Cedric Utanes",       timeIn1: "9:00 AM",  timeOut1: "1:35 PM",  timeIn2: "2:45 PM",  timeOut2: "6:00 PM"  },
  { name: "Maria Santos",              timeIn1: "7:05 AM",  timeOut1: "12:10 PM", timeIn2: "1:10 PM",  timeOut2: "5:00 PM"  },
  { name: "Juan Carlo Dela Cruz",      timeIn1: "7:30 AM",  timeOut1: "11:30 AM", timeIn2: "12:30 PM", timeOut2: "5:30 PM"  },
  { name: "Ana Marie Villanueva",      timeIn1: "7:15 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "5:00 PM"  },
  { name: "Jose Antonio Reyes",        timeIn1: "8:00 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "5:30 PM"  },
  { name: "Maricel Cruz",              timeIn1: "7:50 AM",  timeOut1: "11:50 AM", timeIn2: "12:50 PM", timeOut2: "5:00 PM"  },
  { name: "Ramon Eduardo Mendoza",     timeIn1: "7:20 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "6:00 PM"  },
  { name: "Liza Mae Torres",           timeIn1: "6:55 AM",  timeOut1: "11:55 AM", timeIn2: "12:55 PM", timeOut2: "5:00 PM"  },
  { name: "Roberto James Dela Torre",  timeIn1: "8:05 AM",  timeOut1: "12:00 PM", timeIn2: "1:05 PM",  timeOut2: "5:30 PM"  },
  { name: "Carmela Ann Aquino",        timeIn1: "7:10 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "5:00 PM"  },
  { name: "Fernando Luis Ramos",       timeIn1: "7:45 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "5:45 PM"  },
  { name: "Jenny Rose Castillo",       timeIn1: "7:25 AM",  timeOut1: "11:45 AM", timeIn2: "12:45 PM", timeOut2: "5:00 PM"  },
  { name: "Mark Anthony Florendo",     timeIn1: "8:15 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "6:00 PM"  },
  { name: "Rowena Grace Espiritu",     timeIn1: "7:00 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "5:30 PM"  },
];

// Weekly — May 5–11, 2025
const WEEKLY_ROWS: EmployeeTimeRow[] = [
  { name: "Christian Lloyd Brito",    timeIn1: "7:02 AM",  timeOut1: "12:05 PM", timeIn2: "1:05 PM",  timeOut2: "5:15 PM"  },
  { name: "Norly Boy Bisso",           timeIn1: "6:58 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "5:10 PM"  },
  { name: "Patrick Falcutila",         timeIn1: "8:00 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "5:55 PM"  },
  { name: "Althea Fadriquela",         timeIn1: "7:10 AM",  timeOut1: "11:50 AM", timeIn2: "12:50 PM", timeOut2: "4:45 PM"  },
  { name: "Wayne Cedric Utanes",       timeIn1: "8:30 AM",  timeOut1: "12:30 PM", timeIn2: "1:30 PM",  timeOut2: "5:30 PM"  },
  { name: "Maria Santos",              timeIn1: "7:00 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "5:00 PM"  },
  { name: "Juan Carlo Dela Cruz",      timeIn1: "7:45 AM",  timeOut1: "11:45 AM", timeIn2: "12:45 PM", timeOut2: "5:45 PM"  },
  { name: "Ana Marie Villanueva",      timeIn1: "7:20 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "5:20 PM"  },
  { name: "Jose Antonio Reyes",        timeIn1: "7:55 AM",  timeOut1: "11:55 AM", timeIn2: "12:55 PM", timeOut2: "5:00 PM"  },
  { name: "Maricel Cruz",              timeIn1: "8:10 AM",  timeOut1: "12:10 PM", timeIn2: "1:10 PM",  timeOut2: "5:30 PM"  },
  { name: "Ramon Eduardo Mendoza",     timeIn1: "7:30 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "6:15 PM"  },
  { name: "Liza Mae Torres",           timeIn1: "7:05 AM",  timeOut1: "11:45 AM", timeIn2: "12:45 PM", timeOut2: "4:50 PM"  },
  { name: "Roberto James Dela Torre",  timeIn1: "8:20 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "5:40 PM"  },
  { name: "Carmela Ann Aquino",        timeIn1: "7:15 AM",  timeOut1: "11:50 AM", timeIn2: "12:50 PM", timeOut2: "5:00 PM"  },
  { name: "Fernando Luis Ramos",       timeIn1: "7:40 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "5:35 PM"  },
  { name: "Jenny Rose Castillo",       timeIn1: "7:00 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "5:00 PM"  },
  { name: "Mark Anthony Florendo",     timeIn1: "8:05 AM",  timeOut1: "12:05 PM", timeIn2: "1:05 PM",  timeOut2: "5:55 PM"  },
  { name: "Rowena Grace Espiritu",     timeIn1: "6:55 AM",  timeOut1: "11:55 AM", timeIn2: "12:55 PM", timeOut2: "5:10 PM"  },
];

// Monthly — May 2025
const MONTHLY_ROWS: EmployeeTimeRow[] = [
  { name: "Christian Lloyd Brito",    timeIn1: "7:01 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "5:05 PM"  },
  { name: "Norly Boy Bisso",           timeIn1: "6:50 AM",  timeOut1: "11:50 AM", timeIn2: "12:50 PM", timeOut2: "5:00 PM"  },
  { name: "Patrick Falcutila",         timeIn1: "8:05 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "6:00 PM"  },
  { name: "Althea Fadriquela",         timeIn1: "7:00 AM",  timeOut1: "12:00 PM", timeIn2: "1:15 PM",  timeOut2: "4:30 PM"  },
  { name: "Wayne Cedric Utanes",       timeIn1: "8:45 AM",  timeOut1: "12:45 PM", timeIn2: "1:45 PM",  timeOut2: "5:45 PM"  },
  { name: "Maria Santos",              timeIn1: "7:10 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "5:10 PM"  },
  { name: "Juan Carlo Dela Cruz",      timeIn1: "7:35 AM",  timeOut1: "11:35 AM", timeIn2: "12:35 PM", timeOut2: "5:35 PM"  },
  { name: "Ana Marie Villanueva",      timeIn1: "7:10 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "5:10 PM"  },
  { name: "Jose Antonio Reyes",        timeIn1: "8:00 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "5:20 PM"  },
  { name: "Maricel Cruz",              timeIn1: "7:55 AM",  timeOut1: "11:45 AM", timeIn2: "12:45 PM", timeOut2: "5:00 PM"  },
  { name: "Ramon Eduardo Mendoza",     timeIn1: "7:25 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "6:00 PM"  },
  { name: "Liza Mae Torres",           timeIn1: "7:00 AM",  timeOut1: "11:55 AM", timeIn2: "12:55 PM", timeOut2: "5:00 PM"  },
  { name: "Roberto James Dela Torre",  timeIn1: "8:10 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "5:30 PM"  },
  { name: "Carmela Ann Aquino",        timeIn1: "7:05 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "5:05 PM"  },
  { name: "Fernando Luis Ramos",       timeIn1: "7:50 AM",  timeOut1: "11:50 AM", timeIn2: "12:50 PM", timeOut2: "5:50 PM"  },
  { name: "Jenny Rose Castillo",       timeIn1: "7:15 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "5:00 PM"  },
  { name: "Mark Anthony Florendo",     timeIn1: "8:00 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "6:00 PM"  },
  { name: "Rowena Grace Espiritu",     timeIn1: "7:00 AM",  timeOut1: "12:00 PM", timeIn2: "1:00 PM",  timeOut2: "5:25 PM"  },
];

// ── PDF cards config ──────────────────────────────────────────────────────────

interface PDFCard {
  title:      string;
  subtitle:   string;
  period:     SummaryPeriod;
  periodStr:  string;
  rows:       EmployeeTimeRow[];
  filename:   string;
  icon:       React.ElementType;
  gradient:   string;
  accent:     string;
  badge:      string;
}

const PDF_CARDS: PDFCard[] = [
  {
    title:     "Daily Summary",
    subtitle:  "May 16, 2025",
    period:    "daily",
    periodStr: "May 16, 2025",
    rows:      DAILY_ROWS,
    filename:  "CET_DailySummary_May16_2025.pdf",
    icon:      Calendar,
    gradient:  "from-blue-500 to-blue-700",
    accent:    "bg-blue-50 border-blue-200 text-blue-700",
    badge:     "bg-blue-100 text-blue-800",
  },
  {
    title:     "Weekly Summary",
    subtitle:  "May 5 – May 11, 2025",
    period:    "weekly",
    periodStr: "May 5 \u2013 May 11, 2025",
    rows:      WEEKLY_ROWS,
    filename:  "CET_WeeklySummary_May5-11_2025.pdf",
    icon:      CalendarDays,
    gradient:  "from-emerald-500 to-emerald-700",
    accent:    "bg-emerald-50 border-emerald-200 text-emerald-700",
    badge:     "bg-emerald-100 text-emerald-800",
  },
  {
    title:     "Monthly Summary",
    subtitle:  "May 2025",
    period:    "monthly",
    periodStr: "May 2025",
    rows:      MONTHLY_ROWS,
    filename:  "CET_MonthlySummary_May2025.pdf",
    icon:      CalendarRange,
    gradient:  "from-purple-500 to-purple-700",
    accent:    "bg-purple-50 border-purple-200 text-purple-700",
    badge:     "bg-purple-100 text-purple-800",
  },
];

// ── Component ─────────────────────────────────────────────────────────────────

export default function SamplePDFs() {
  const [generating, setGenerating] = useState<string | null>(null);

  const handleDownload = async (card: PDFCard) => {
    setGenerating(card.period);
    try {
      await exportSamplePDF(DEPT, card.rows, card.period, card.periodStr, card.filename);
    } finally {
      setGenerating(null);
    }
  };

  return (
    <AppLayout>
      <div className="max-w-4xl mx-auto">

        {/* Page header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-1">
            <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
              <FileText className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900">Sample Attendance PDFs</h2>
              <p className="text-sm text-gray-500 mt-0.5">
                Ready-to-print attendance summary samples — Daily, Weekly, and Monthly formats
              </p>
            </div>
          </div>
        </div>

        {/* Info notice */}
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 mb-8 flex items-start gap-3">
          <span className="text-amber-500 text-lg mt-0.5">ℹ</span>
          <div className="text-sm text-amber-800">
            <p className="font-semibold">Sample data — Department: {DEPT}</p>
            <p className="mt-0.5 text-amber-700">
              These PDFs use 18 randomized personnel names with realistic schedules for May 2025.
              Each row shows an employee's morning session (TIME IN / TIME OUT) and afternoon session (TIME IN / TIME OUT).
            </p>
          </div>
        </div>

        {/* PDF Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {PDF_CARDS.map((card) => {
            const Icon = card.icon;
            const isGenerating = generating === card.period;
            return (
              <div
                key={card.period}
                className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden flex flex-col"
              >
                {/* Gradient top bar */}
                <div className={`bg-gradient-to-r ${card.gradient} h-1.5 w-full`} />

                <div className="p-6 flex flex-col gap-4 flex-1">
                  {/* Icon + title */}
                  <div className="flex items-start gap-3">
                    <div className={`w-11 h-11 rounded-xl bg-gradient-to-br ${card.gradient} flex items-center justify-center shadow-md flex-shrink-0`}>
                      <Icon className="w-5 h-5 text-white" />
                    </div>
                    <div className="flex-1">
                      <h3 className="font-bold text-gray-900">{card.title}</h3>
                      <p className="text-sm font-medium text-gray-500 mt-0.5">{card.subtitle}</p>
                    </div>
                  </div>

                  {/* Stats */}
                  <div className="flex flex-wrap gap-2">
                    <span className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${card.accent}`}>
                      {card.rows.length} employees
                    </span>
                    <span className={`text-xs font-semibold px-2.5 py-1 rounded-full ${card.badge}`}>
                      A4 · Portrait
                    </span>
                  </div>

                  {/* Preview table (mini) */}
                  <div className="border border-gray-200 rounded-lg overflow-hidden text-[9px] flex-1">
                    <div className="bg-gray-800 text-white grid grid-cols-5 text-center">
                      <div className="col-span-2 px-1 py-1 text-left pl-2 font-bold">NAME</div>
                      <div className="px-1 py-1 font-bold">TI</div>
                      <div className="px-1 py-1 font-bold">TO</div>
                      <div className="px-1 py-1 font-bold">TO</div>
                    </div>
                    {card.rows.slice(0, 5).map((r, i) => (
                      <div
                        key={i}
                        className={`grid grid-cols-5 text-center ${i % 2 === 1 ? "bg-blue-50/60" : "bg-white"}`}
                      >
                        <div className="col-span-2 px-1.5 py-0.5 text-left text-gray-700 font-medium truncate">{r.name.split(" ")[0]} {r.name.split(" ")[1] ?? ""}</div>
                        <div className="px-0.5 py-0.5 text-green-700">{r.timeIn1.replace(" AM","").replace(" PM","")}</div>
                        <div className="px-0.5 py-0.5 text-red-700">{r.timeOut1.replace(" AM","").replace(" PM","")}</div>
                        <div className="px-0.5 py-0.5 text-red-700">{r.timeOut2.replace(" AM","").replace(" PM","")}</div>
                      </div>
                    ))}
                    <div className="bg-gray-50 text-center text-gray-400 py-0.5">+ {card.rows.length - 5} more rows</div>
                  </div>

                  {/* Download button */}
                  <button
                    onClick={() => handleDownload(card)}
                    disabled={!!generating}
                    className={`w-full flex items-center justify-center gap-2 py-2.5 rounded-xl font-semibold text-sm transition-all ${
                      isGenerating
                        ? "bg-gray-100 text-gray-500 cursor-not-allowed"
                        : `bg-gradient-to-r ${card.gradient} text-white hover:opacity-90 shadow-md hover:shadow-lg`
                    }`}
                  >
                    {isGenerating ? (
                      <>
                        <div className="w-4 h-4 border-2 border-gray-400 border-t-transparent rounded-full animate-spin" />
                        Generating PDF…
                      </>
                    ) : (
                      <>
                        <Download className="w-4 h-4" />
                        Download {card.title}
                      </>
                    )}
                  </button>
                </div>
              </div>
            );
          })}
        </div>

        {/* Format description */}
        <div className="mt-8 bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
          <h3 className="font-bold text-gray-800 mb-3">PDF Table Format</h3>
          <div className="border border-gray-200 rounded-xl overflow-hidden text-sm">
            <div className="grid grid-cols-5 bg-[#0a2463] text-white font-bold text-center text-xs">
              <div className="col-span-2 py-2 px-3 text-left">NAME</div>
              <div className="py-2 px-2">TIME IN</div>
              <div className="py-2 px-2">TIME OUT</div>
              <div className="py-2 px-2">TIME IN</div>
            </div>
            {[
              { name: "Christian Lloyd Brito", ti1: "6:50 AM", to1: "11:40 AM", ti2: "1:30 PM", to2: "5:00 PM" },
              { name: "Patrick Falcutila",      ti1: "8:10 AM", to1: "12:00 PM", ti2: "1:00 PM", to2: "6:00 PM" },
              { name: "Ana Marie Villanueva",   ti1: "7:15 AM", to1: "12:00 PM", ti2: "1:00 PM", to2: "5:00 PM" },
            ].map((r, i) => (
              <div key={i} className={`grid grid-cols-5 text-center text-xs ${i % 2 === 1 ? "bg-blue-50/60" : "bg-white"}`}>
                <div className="col-span-2 py-2 px-3 text-left font-semibold text-gray-800">{r.name}</div>
                <div className="py-2 px-2 bg-green-50 text-green-700 font-bold">{r.ti1}</div>
                <div className="py-2 px-2 bg-red-50 text-red-700 font-bold">{r.to1}</div>
                <div className="py-2 px-2 bg-green-50 text-green-700 font-bold">{r.ti2}</div>
              </div>
            ))}
          </div>
          <p className="text-xs text-gray-500 mt-3">
            First pair = morning session &nbsp;·&nbsp; Second pair = afternoon session &nbsp;·&nbsp;
            Green cells = TIME IN &nbsp;·&nbsp; Red cells = TIME OUT
          </p>
        </div>

      </div>
    </AppLayout>
  );
}
